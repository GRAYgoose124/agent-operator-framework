"""Layered vault graph: the structure a reader (or a report writer) can actually traverse.

The first structure pass linked every claim to its ~6 most similar claims. On a 4.5k-claim vault that produced one
5,000-node hairball (every claim within two hops of hundreds of others), hubs of 200+ claims, and ~1,400 isolated
archive notes. Similarity alone is not structure. This module rebuilds claim links from typed layers instead:

    topic hub (tree, <= max_hub_size direct members, unique titles)
      └─ claim ── concept notes (the 1-3 specific terms it mentions: "place cells", "TRN")
               ── source note (the paper it is quoted from; claims of one paper are one neighbourhood)
               ── 0-2 mutual nearest-neighbour claims from *other* papers (strong semantic relation only)

Every claim link is derived, so the pass is idempotent: re-running replaces claim links wholesale. Concept and
source notes are extractive (they list claims; no model prose). Stale concept/source notes are archived, not deleted.
"""

from __future__ import annotations

import hashlib
import logging
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from aof.memory.store import MemoryStore
from aof.memory.zettel import ZettelNote
from aof.refine.assess import live_claims
from aof.refine.claims import Embedder
from aof.refine.concepts import Concept, extract_concepts, related_concepts
from aof.refine.hubs import build_hubs
from aof.refine.vectors import unit_rows
from aof.refine.verify import claim_text, url_work_key
from aof.specialists import SpecialistRegistry

logger = logging.getLogger(__name__)

GRAPH_AGENT = "refine"
_EVIDENCE_LINE = re.compile(r'^- "(?P<quote>.*)" — (?P<cite>.*?)(?: \((?P<year>\d{4})\))?, (?P<url>https?://\S+)\s*$')


@dataclass
class GraphOptions:
    link_k: int = 2  # mutual nearest neighbours kept per claim (0 disables claim-claim links)
    link_min_sim: float = 0.62
    link_max_sim: float = 0.92  # above this the pair is a near-duplicate, not a relation
    concept_min_df: int = 4
    concept_max_share: float = 0.06
    max_concepts: int = 0  # 0 = scale with the vault (one per ~12 claims, 60..800)
    concepts_per_claim: int = 3
    concept_key_claims: int = 8  # "key statements" quoted on a concept note
    concept_list_max: int = 80  # claims listed on a concept note (the rest are counted)
    hub_threshold: float = 0.5
    hub_min_size: int = 3
    max_hub_size: int = 40


@dataclass
class GraphReport:
    claims: int = 0
    hubs: int = 0
    concepts: int = 0
    sources: int = 0
    claim_links: int = 0  # claim-claim edges (undirected)
    claims_changed: int = 0
    retired: dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}


# ---------------------------------------------------------------- sources

@dataclass
class SourceInfo:
    key: str  # work identity (doi:… or url:…)
    url: str
    title: str = ""
    year: str = ""
    claim_ids: list[str] = field(default_factory=list)

    @property
    def note_id(self) -> str:
        return "src-" + hashlib.sha1(self.key.encode()).hexdigest()[:10]

    @property
    def label(self) -> str:
        title = self.title or self.url
        return f"{title} ({self.year})" if self.year else title


def parse_evidence(note: ZettelNote) -> list[dict]:
    """The note's evidence lines as dicts (quote, cite, year, url)."""
    out = []
    for line in note.content.splitlines():
        m = _EVIDENCE_LINE.match(line.strip())
        if m:
            out.append(m.groupdict())
    return out


def collect_sources(notes: Sequence[ZettelNote]) -> dict[str, SourceInfo]:
    """Works cited by live claims, keyed by work identity, with the claims that cite them."""
    works: dict[str, SourceInfo] = {}
    for n in notes:
        meta = {e["url"]: e for e in parse_evidence(n)}
        for url in dict.fromkeys(n.sources):
            key = url_work_key(url)
            info = works.get(key)
            if info is None:
                e = meta.get(url, {})
                info = works[key] = SourceInfo(key, url, (e.get("cite") or "").strip(), e.get("year") or "")
            elif not info.title and url in meta:
                info.title, info.year = meta[url].get("cite", ""), meta[url].get("year") or ""
            if n.id not in info.claim_ids:
                info.claim_ids.append(n.id)
    return works


# ---------------------------------------------------------------- claim-claim links

def mutual_knn(
    notes: Sequence[ZettelNote],
    vectors: dict[str, Sequence[float]],
    *,
    k: int = 2,
    min_sim: float = 0.62,
    max_sim: float = 0.92,
    same_work: dict[str, set[str]] | None = None,
    block: int = 2048,
) -> dict[str, set[str]]:
    """Edges between claims that are among each other's `k` nearest neighbours (and not near-duplicates).

    Claims quoting the same work are skipped (`same_work`: id -> work keys); they already meet at the source note.
    Computed in row blocks so a 10k-claim vault never needs a dense n x n matrix in memory at once.
    """
    n = len(notes)
    edges: dict[str, set[str]] = {x.id: set() for x in notes}
    if n < 2 or k <= 0:
        return edges
    x = unit_rows([vectors[m.id] for m in notes])
    works = [same_work.get(m.id, set()) if same_work else set() for m in notes]
    top: list[list[int]] = []
    for start in range(0, n, block):
        sims = x[start:start + block] @ x.T
        for r in range(sims.shape[0]):
            i = start + r
            row = sims[r]
            row[i] = -1.0
            row[(row < min_sim) | (row >= max_sim)] = -1.0
            cand = np.argpartition(-row, min(n - 1, k * 4))[: k * 4]
            cand = [int(j) for j in cand[np.argsort(-row[cand])] if row[j] > -1.0 and not (works[i] & works[int(j)])]
            top.append(cand[:k])
    sets = [set(t) for t in top]
    for i, nbrs in enumerate(top):
        for j in nbrs:
            if i in sets[j]:
                edges[notes[i].id].add(notes[j].id)
                edges[notes[j].id].add(notes[i].id)
    return edges


# ---------------------------------------------------------------- concept notes

def broader_concepts(concepts: Sequence[Concept]) -> dict[str, str]:
    """{narrow key: broader key} when a concept's phrase contains another concept's phrase
    ("medial entorhinal cortex" -> "entorhinal cortex"). The longest contained phrase wins."""
    keys = {c.key for c in concepts}
    out: dict[str, str] = {}
    for c in concepts:
        words = c.key.split()
        best = ""
        for size in range(len(words) - 1, 0, -1):
            for i in range(len(words) - size + 1):
                sub = " ".join(words[i:i + size])
                if sub in keys and len(sub) > len(best):
                    best = sub
            if best:
                break
        if best:
            out[c.key] = best
    return out


def _concept_content(
    c: Concept,
    key_ids: Sequence[str],
    listed: Sequence[str],
    by_id: dict[str, ZettelNote],
    slug: dict[str, str],
    label: dict[str, str],
    broader: str | None,
    narrower: Sequence[str],
    related: Sequence[str],
    hubs: Sequence[tuple[str, int]],
    n_sources: int,
) -> str:
    also = f" (also: {', '.join(sorted(c.aliases))})" if c.aliases else ""
    lines = [f"Concept: **{c.label}**{also}. {len(c.claim_ids)} claims from {n_sources} sources.", ""]
    if broader:
        lines.append(f"Broader: [[{slug[broader]}]]")
    if narrower:
        lines.append("Narrower: " + ", ".join(f"[[{slug[k]}]]" for k in narrower))
    if related:
        lines.append("Related: " + ", ".join(f"[[{slug[k]}]]" for k in related))
    if hubs:
        lines.append("Topics: " + ", ".join(f"[[{h}]] ({cnt})" for h, cnt in hubs))
    lines += ["", "**Key statements**"]
    lines += [f"- [[{i}]] {claim_text(by_id[i])}" for i in key_ids]
    rest = [i for i in listed if i not in set(key_ids)]
    if rest:
        lines += ["", "**More claims**"]
        lines += [f"- [[{i}]] {by_id[i].title}" for i in rest]
    hidden = len(c.claim_ids) - len(key_ids) - len(rest)
    if hidden > 0:
        lines += ["", f"_{hidden} further claims mention {c.label}; see the topic hubs above._"]
    return "\n".join(lines)


def _source_content(info: SourceInfo, by_id: dict[str, ZettelNote]) -> str:
    lines = [f"Source: **{info.label}**", "", info.url, "", f"**Claims quoted from this work** ({len(info.claim_ids)})"]
    lines += [f"- [[{i}]] {claim_text(by_id[i])}" for i in info.claim_ids if i in by_id]
    return "\n".join(lines)


async def _upsert(store: MemoryStore, note_id: str, title: str, content: str, tags: list[str], links: list[str],
                  kind: str, sources: list[str]) -> bool:
    """Create or refresh a derived note; returns True if anything changed."""
    existing = await store.get_note(note_id)
    if existing is None:
        await store.create_note(title, content, tags, links=links, agent_id=GRAPH_AGENT, note_id=note_id,
                                kind=kind, status="canonical", sources=sources)
        return True
    if (existing.title, existing.content, existing.tags, existing.links, existing.sources, existing.status) == (
        title, content, tags, links, sources, "canonical"
    ):
        return False
    existing.title, existing.content, existing.tags, existing.links = title, content, tags, links
    existing.sources, existing.status, existing.kind = sources, "canonical", kind
    await store.update_note(existing)
    return True


async def _retire(store: MemoryStore, tag: str, keep: set[str]) -> int:
    retired = 0
    for note in await store.get_notes_by_tags([tag], limit=100000, exclude_tags=["archived"]):
        if note.id in keep:
            continue
        note.status = "archived"
        note.tags = [*note.tags, "archived"]
        await store.update_note(note)
        retired += 1
    return retired


# ---------------------------------------------------------------- the pass

async def build_graph(
    store: MemoryStore,
    registry: SpecialistRegistry | None,
    embed: Embedder,
    options: GraphOptions | None = None,
    *,
    progress=None,
) -> GraphReport:
    """Rebuild the vault's derived structure (hubs, concepts, sources, claim links). Safe to re-run."""
    opt = options or GraphOptions()
    say = progress or (lambda msg: logger.info("graph: %s", msg))
    report = GraphReport()
    notes = [n for n in await live_claims(store) if n.kind == "claim"]
    report.claims = len(notes)
    if not notes:
        return report
    by_id = {n.id: n for n in notes}
    say(f"embedding {len(notes)} claims")
    vectors = dict(zip((n.id for n in notes), await embed([claim_text(n) for n in notes])))

    say("building topic hubs")
    hubs = await build_hubs(
        store, notes, vectors, registry=registry, threshold=opt.hub_threshold, min_size=opt.hub_min_size,
        max_hub_size=opt.max_hub_size, link_members=False,
    )
    report.hubs = len(hubs)
    hub_of: dict[str, str] = {}
    hub_title = {h.id: h.title for h in hubs}
    for h in hubs:  # parents first, so the most specific (deepest) hub listing a claim wins
        for i in h.links:
            if i in by_id:
                hub_of[i] = h.id

    say("extracting concepts")
    items = [(n.id, claim_text(n)) for n in notes]
    max_concepts = opt.max_concepts or max(60, min(800, len(notes) // 12))
    concepts, assigned = extract_concepts(
        items, min_df=opt.concept_min_df, max_share=opt.concept_max_share, max_concepts=max_concepts,
        per_claim=opt.concepts_per_claim,
    )
    slug = {c.key: c.slug for c in concepts}
    label = {c.key: c.label for c in concepts}
    related = related_concepts(concepts, assigned)
    broader = broader_concepts(concepts)
    narrower: dict[str, list[str]] = defaultdict(list)
    for k, b in broader.items():
        narrower[b].append(k)
    works = collect_sources(notes)
    work_of: dict[str, set[str]] = defaultdict(set)
    for w in works.values():
        for i in w.claim_ids:
            work_of[i].add(w.key)

    say(f"writing {len(concepts)} concept notes")
    for c in concepts:
        members = [i for i in c.claim_ids if i in by_id]
        if not members:
            continue
        x = unit_rows([vectors[i] for i in members])
        centre = unit_rows(x.sum(axis=0))[0]
        corroborated = np.array([by_id[i].status == "canonical" or "grade:core" in by_id[i].tags for i in members])
        order = np.argsort(-(x @ centre + 0.05 * corroborated), kind="stable")
        ranked = [members[int(j)] for j in order]
        key_ids = ranked[: opt.concept_key_claims]
        listed = ranked[: opt.concept_list_max]
        hub_counts = Counter(hub_of[i] for i in members if i in hub_of).most_common(4)
        n_sources = len({w for i in members for w in work_of[i]})
        narrow = narrower.get(c.key, [])[:12]
        rel = [k for k in related.get(c.key, []) if k != broader.get(c.key) and k not in narrow][:6]
        content = _concept_content(
            c, key_ids, listed, by_id, slug, label, broader.get(c.key), narrow, rel, hub_counts, n_sources,
        )
        links = list(dict.fromkeys(
            ([slug[broader[c.key]]] if c.key in broader else []) + [slug[k] for k in narrow]
            + [slug[k] for k in rel] + [h for h, _ in hub_counts] + listed
        ))
        tags = ["concept", *(f"alias:{a.lower()}" for a in sorted(c.aliases))]
        await _upsert(store, c.slug, c.label, content, tags, links, "concept",
                      list(dict.fromkeys(u for i in key_ids for u in by_id[i].sources))[:20])
    report.concepts = len(concepts)

    say(f"writing {len(works)} source notes")
    for info in works.values():
        await _upsert(store, info.note_id, info.label[:180], _source_content(info, by_id), ["source"],
                      [i for i in info.claim_ids if i in by_id], "source", [info.url])
    report.sources = len(works)

    say("linking claims")
    knn = mutual_knn(notes, vectors, k=opt.link_k, min_sim=opt.link_min_sim, max_sim=opt.link_max_sim,
                     same_work=work_of)
    report.claim_links = sum(len(v) for v in knn.values()) // 2
    source_note = {w.key: w.note_id for w in works.values()}
    for n in notes:
        links = list(dict.fromkeys(
            ([hub_of[n.id]] if n.id in hub_of else [])
            + [slug[k] for k in assigned.get(n.id, []) if k in slug]
            + [source_note[w] for w in sorted(work_of[n.id])]
            + sorted(knn[n.id])
        ))
        if links != n.links:
            n.links = links
            await store.update_note(n)
            report.claims_changed += 1

    report.retired = {
        "concept": await _retire(store, "concept", {c.slug for c in concepts}),
        "source": await _retire(store, "source", {w.note_id for w in works.values()}),
    }
    say(f"done: {report.as_dict()}")
    return report
