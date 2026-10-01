"""Vault quality metrics: is the vault deduplicated, connected, sourced and verified?"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Sequence

import numpy as np
from urllib.parse import urlparse

from aof.memory.store import MemoryStore
from aof.memory.zettel import ZettelNote
from aof.refine.assess import live_claims
from aof.refine.vectors import cosine_matrix
from aof.refine.pipeline import registry_embedder
from aof.refine.verify import claim_text
from aof.specialists import SpecialistRegistry


@dataclass
class VaultMetrics:
    live_claims: int = 0
    archived: int = 0  # merged duplicates and curated-out claims, all retained
    hubs: int = 0
    duplicate_rate: float = 0.0  # live claim pairs with cosine >= threshold, per live claim
    orphan_rate: float = 0.0  # live claims with no links at all
    avg_links: float = 0.0
    hub_coverage: float = 0.0  # live claims belonging to at least one hub
    sourced_rate: float = 0.0  # live claims with >= 1 source (should be 1.0)
    corroborated_rate: float = 0.0  # live claims backed by >= 2 sources
    verified_rate: float = 0.0  # live claims with a recorded corroboration lookup
    conflicts: int = 0
    source_mix: dict[str, int] = field(default_factory=dict)  # source host -> claim count
    graph: dict[str, float] = field(default_factory=dict)  # see graph_shape()

    def as_dict(self) -> dict:
        return asdict(self)

    def render(self) -> str:
        rows = [
            ("live claims", self.live_claims), ("archived (retained)", self.archived), ("hubs", self.hubs),
            ("duplicate rate", f"{self.duplicate_rate:.2f}"), ("orphan rate", f"{self.orphan_rate:.0%}"),
            ("avg links / claim", f"{self.avg_links:.1f}"), ("hub coverage", f"{self.hub_coverage:.0%}"),
            ("sourced", f"{self.sourced_rate:.0%}"), ("corroborated (>=2 sources)", f"{self.corroborated_rate:.0%}"),
            ("verified by lookup", f"{self.verified_rate:.0%}"), ("open conflicts", self.conflicts),
        ]
        rows += [(f"graph: {k.replace('_', ' ')}", f"{v:.2f}" if isinstance(v, float) else v) for k, v in self.graph.items()]
        out = ["| metric | value |", "|---|---|", *[f"| {k} | {v} |" for k, v in rows]]
        if self.source_mix:
            out += ["", "Sources: " + ", ".join(f"{h} ({c})" for h, c in sorted(self.source_mix.items(), key=lambda kv: -kv[1]))]
        return "\n".join(out)


def _host(url: str) -> str:
    host = urlparse(url).netloc.lower()
    return host[4:] if host.startswith("www.") else host or "unknown"


def measure(
    live: Sequence[ZettelNote],
    archived: int,
    hubs: Sequence[ZettelNote],
    vectors: dict[str, Sequence[float]] | None = None,
    *,
    duplicate_threshold: float = 0.9,
) -> VaultMetrics:
    n = len(live)
    m = VaultMetrics(live_claims=n, archived=archived, hubs=len(hubs))
    if n == 0:
        return m
    hub_members = {i for h in hubs for i in h.links}  # includes claims listed by any hub, at any level
    m.orphan_rate = sum(1 for x in live if not x.links) / n
    m.avg_links = sum(len(x.links) for x in live) / n
    m.hub_coverage = sum(1 for x in live if x.id in hub_members) / n
    m.sourced_rate = sum(1 for x in live if x.sources) / n
    m.corroborated_rate = sum(1 for x in live if len(x.sources) >= 2) / n
    m.verified_rate = sum(1 for x in live if "**Corroboration**" in x.content) / n
    m.conflicts = sum(1 for x in live if "conflict" in x.tags)
    for x in live:
        for host in {_host(u) for u in x.sources}:
            m.source_mix[host] = m.source_mix.get(host, 0) + 1
    if vectors and n > 1:
        sims = cosine_matrix([vectors[x.id] for x in live])
        m.duplicate_rate = int(np.triu(sims >= duplicate_threshold, k=1).sum()) / n
    return m


def graph_shape(notes: Sequence[ZettelNote]) -> dict[str, float]:
    """Shape of the link graph over the given notes (links to notes outside the set are ignored).

    What a reader feels when traversing: isolated notes (orphans), whether everything collapses into one blob
    (largest component share, claim-claim degree), and whether hubs are walls of claims (max direct members) or
    share titles (duplicate titles make the map ambiguous).
    """
    ids = {n.id for n in notes}
    adj: dict[str, set[str]] = {n.id: set() for n in notes}
    for n in notes:
        for t in n.links:
            if t in ids and t != n.id:
                adj[n.id].add(t)
                adj[t].add(n.id)
    seen: set[str] = set()
    sizes = []
    for start in adj:
        if start in seen:
            continue
        stack, size = [start], 0
        seen.add(start)
        while stack:
            x = stack.pop()
            size += 1
            for y in adj[x] - seen:
                seen.add(y)
                stack.append(y)
        sizes.append(size)
    kind = {n.id: n.kind for n in notes}
    claims = [n for n in notes if n.kind == "claim"]
    hubs = [n for n in notes if n.kind == "hub"]
    titles = [h.title.lower() for h in hubs]
    out: dict[str, float] = {
        "nodes": len(notes),
        "isolated_nodes": sum(1 for v in adj.values() if not v),
        "components": len(sizes),
        "largest_component_share": max(sizes) / len(notes) if notes else 0.0,
        "claim_claim_degree": (
            sum(sum(1 for t in adj[c.id] if kind.get(t) == "claim") for c in claims) / len(claims) if claims else 0.0
        ),
        "max_hub_direct_claims": max((sum(1 for t in h.links if kind.get(t) == "claim") for h in hubs), default=0),
        "duplicate_hub_titles": len(titles) - len(set(titles)),
    }
    for k in ("concept", "source", "report"):
        out[f"{k}_notes"] = sum(1 for n in notes if n.kind == k)
    if claims:
        out["claims_with_concept"] = sum(1 for c in claims if any(kind.get(t) == "concept" for t in c.links)) / len(claims)
    return out


async def compute_metrics(store: MemoryStore, registry: SpecialistRegistry | None = None) -> VaultMetrics:
    live = [n for n in await live_claims(store) if n.kind == "claim"]
    everything = await store.get_notes_by_tags(["claim"], limit=100000)
    hubs = await store.get_notes_by_tags(["hub"], limit=100000, exclude_tags=["archived"])
    vectors = None
    embed = registry_embedder(registry) if registry is not None else None
    if embed and live:
        vectors = dict(zip((n.id for n in live), await embed([claim_text(n) for n in live])))
    m = measure(live, len(everything) - len(live), hubs, vectors)
    derived = await store.get_notes_by_tags(["hub", "concept", "source", "report"], limit=100000, exclude_tags=["archived"])
    m.graph = graph_shape([*live, *(n for n in derived if n.status != "archived")])
    return m
