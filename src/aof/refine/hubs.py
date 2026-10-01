"""Hub (map-of-content) notes: group related claims and give each group a navigable entry point.

Hubs are extractive on purpose: a hub lists its member claims as links and quotes them; it does not add prose that
could drift from the sources. Only the *title* is model-written, and it must reuse vocabulary from its members.

Large topics are split into subtopic hubs (recursively, with a stricter similarity threshold), so no hub is a
wall of hundreds of claims: a parent hub lists its subtopic hubs, and only claims that fit no subtopic.
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from aof.memory.store import MemoryStore
from aof.memory.zettel import ZettelNote
from aof.refine.text import content_tokens
from aof.refine.vectors import unit_rows
from aof.refine.verify import claim_text
from aof.specialists import GENERATE, SpecialistExhausted, SpecialistRegistry

logger = logging.getLogger(__name__)

HUB_AGENT = "refine"

_TITLE_SYSTEM = (
    "You name groups of related scientific statements. Reply with a short topic title of 2-6 words, "
    "in title case, with no quotes and no punctuation at the end."
)


def cluster_notes(
    notes: Sequence[ZettelNote],
    vectors: dict[str, Sequence[float]],
    *,
    threshold: float = 0.5,
    min_size: int = 3,
) -> list[list[ZettelNote]]:
    """Greedy leader clustering on cosine similarity to running centroids; small groups fold into the nearest.

    Deterministic: notes in dense neighbourhoods lead (most close neighbours, cosine >= threshold + 0.15), then by
    id. The order must not depend on links, which the graph pass itself derives, or re-running it would reshuffle
    the hubs. (Counting neighbours at `threshold` itself favours bridging notes, which then absorb several topics.)
    """
    unit = {n.id: unit_rows(vectors[n.id])[0] for n in notes}
    rank: dict[str, int] = {}
    if notes:
        x = np.stack([unit[n.id] for n in notes])
        for i in range(0, len(x), 2048):  # row blocks: no dense n x n matrix for large vaults
            counts = ((x[i:i + 2048] @ x.T) >= threshold + 0.15).sum(axis=1)
            rank.update((notes[i + j].id, int(c)) for j, c in enumerate(counts))
    order = sorted(notes, key=lambda n: (-rank[n.id], n.id))
    groups: list[list[ZettelNote]] = []
    sums: list[np.ndarray] = []  # running vector sum per group; direction of the sum is the centroid direction

    def centroids_unit(sum_list: list[np.ndarray]) -> np.ndarray:
        return unit_rows(np.stack(sum_list))

    for note in order:
        best = -1
        if sums:
            sims = centroids_unit(sums) @ unit[note.id]
            top = int(np.argmax(sims))
            if sims[top] >= threshold:
                best = top
        if best >= 0:
            groups[best].append(note)
            sums[best] = sums[best] + unit[note.id]
        else:
            groups.append([note])
            sums.append(unit[note.id].copy())

    keep = [gi for gi, g in enumerate(groups) if len(g) >= min_size]
    if not keep:
        return []
    big = [groups[gi] for gi in keep]
    big_units = centroids_unit([sums[gi] for gi in keep])
    for g in groups:
        if len(g) < min_size:  # fold stragglers into the closest real hub rather than losing them
            for note in g:
                sims = big_units @ unit[note.id]
                nearest = int(np.argmax(sims))
                if sims[nearest] >= threshold * 0.8:
                    big[nearest].append(note)
    return big


@dataclass
class HubNode:
    """One hub in the tree: all its claims, its subtopic hubs, and the claims that fit no subtopic."""

    members: list[ZettelNote]
    children: list["HubNode"] = field(default_factory=list)
    leftovers: list[ZettelNote] = field(default_factory=list)


def _kmeans_rows(x: np.ndarray, k: int, iters: int = 20) -> np.ndarray:
    """Spherical k-means over unit rows; returns the group index per row.

    Initialised from quantile bins along the first principal component (deterministic and balanced; farthest-point
    seeding picks outliers and leaves one giant group plus fragments)."""
    centred = x - x.mean(axis=0)
    _, _, vt = np.linalg.svd(centred, full_matrices=False)
    bins = np.array_split(np.argsort(centred @ vt[0], kind="stable"), k)
    c = unit_rows(np.stack([x[b].sum(axis=0) for b in bins]))
    assign = np.full(len(x), -1)
    for _ in range(iters):
        new = np.argmax(x @ c.T, axis=1)
        if np.array_equal(new, assign):
            break
        assign = new
        for j in range(k):
            pts = x[assign == j]
            if len(pts):
                c[j] = unit_rows(pts.sum(axis=0))[0]
    return assign


def kmeans_split(
    members: Sequence[ZettelNote], vectors: dict[str, Sequence[float]], k: int, iters: int = 20,
) -> list[list[ZettelNote]]:
    """Split `members` into up to `k` groups by spherical k-means. Used when threshold clustering cannot split a
    large group: a hub must never stay a wall of hundreds of claims."""
    k = max(2, min(k, len(members)))
    assign = _kmeans_rows(unit_rows([vectors[m.id] for m in members]), k, iters)
    groups = [[members[i] for i in np.flatnonzero(assign == j)] for j in range(k)]
    return [g for g in groups if g]


def group_roots(roots: list["HubNode"], vectors: dict[str, Sequence[float]], max_top: int) -> list["HubNode"]:
    """Too many top-level hubs make a flat, unbrowsable index: gather them under ~sqrt(n) area hubs."""
    if len(roots) <= max_top:
        return roots
    k = max(2, min(max_top, int(round(len(roots) ** 0.5))))
    cents = unit_rows(np.stack([unit_rows([vectors[m.id] for m in r.members]).sum(axis=0) for r in roots]))
    assign = _kmeans_rows(cents, k)
    areas = []
    for j in range(k):
        kids = [roots[i] for i in np.flatnonzero(assign == j)]
        if len(kids) == 1:
            areas.append(kids[0])
        elif kids:
            areas.append(HubNode(members=[m for c in kids for m in c.members], children=kids))
    return areas


def hub_tree(
    notes: Sequence[ZettelNote],
    vectors: dict[str, Sequence[float]],
    *,
    threshold: float = 0.5,
    min_size: int = 3,
    max_size: int = 60,
    step: float = 0.12,
    max_depth: int = 4,
    max_top: int = 12,
) -> list[HubNode]:
    """Cluster into hubs, then split any hub larger than `max_size` into subtopic hubs.

    A split first tries threshold clustering at a stricter threshold; when that finds fewer than two subtopics it
    falls back to k-means, so a hub keeps at most ~`max_size` direct members (within `max_depth` levels). Claims
    that fit no subtopic join the nearest one unless only a few are left over.
    """

    def split(node: HubNode, thr: float, depth: int) -> None:
        group = node.members
        children = build(group, thr + step, depth + 1)
        # threshold clustering often peels a few claims off one dominant group, which makes a chain of hubs
        # (791 -> 327 -> 160 -> 114 ...), not a map; treat an unbalanced split as a failed one
        if len(children) < 2 or max(len(c.members) for c in children) > 0.45 * len(group):
            parts = kmeans_split(group, vectors, max(2, min(10, -(-len(group) // max_size))))
            children = [HubNode(members=p) for p in parts if len(p) >= min_size]
            for c in children:
                if len(c.members) > max_size and depth + 1 < max_depth:
                    split(c, thr + step, depth + 1)
        if len(children) < 2:
            return  # cannot be split meaningfully: keep it flat
        in_child = {n.id for c in children for n in c.members}
        leftovers = [n for n in group if n.id not in in_child]
        if len(leftovers) > max(min_size, max_size // 4):
            cents = unit_rows(np.stack([unit_rows([vectors[m.id] for m in c.members]).sum(axis=0) for c in children]))
            for n in leftovers:
                children[int(np.argmax(cents @ unit_rows(vectors[n.id])[0]))].members.append(n)
            leftovers = []
        node.children, node.leftovers = children, leftovers

    def build(members: Sequence[ZettelNote], thr: float, depth: int) -> list[HubNode]:
        nodes = []
        for group in cluster_notes(members, vectors, threshold=thr, min_size=min_size):
            node = HubNode(members=group)
            if len(group) > max_size and depth < max_depth:
                split(node, thr, depth)
            nodes.append(node)
        return nodes

    return group_roots(build(list(notes), threshold, 0), vectors, max_top)


def keyword_title(members: Sequence[ZettelNote], limit: int = 4) -> str:
    """Fallback title from the most frequent content words across a group's claims."""
    counts: Counter[str] = Counter()
    for n in members:
        counts.update({t.lower() for t in content_tokens(claim_text(n)) if len(t) > 3})
    top = [w for w, _ in counts.most_common(limit)]
    return " ".join(w.capitalize() for w in top) or "Related claims"


def distinct_word(members: Sequence[ZettelNote], background: Counter[str], exclude: str = "") -> str:
    """The word most characteristic of `members` relative to `background` (tells same-titled hubs apart)."""
    counts: Counter[str] = Counter()
    for n in members:
        counts.update({t.lower() for t in content_tokens(claim_text(n)) if len(t) > 3})
    skip = {w.lower().strip(":,") for w in exclude.split()}
    total = sum(background.values()) or 1
    size = len(members) or 1
    best = max(
        (w for w, c in counts.items() if c >= 2 and w not in skip),
        key=lambda w: (counts[w] / size) / ((background[w] + 1) / total) * min(counts[w], 5),
        default="",
    )
    return best.upper() if best.upper() in {t for n in members for t in content_tokens(claim_text(n))} else best.capitalize()


async def name_hub(
    members: Sequence[ZettelNote], registry: SpecialistRegistry | None, avoid: Sequence[str] = (),
) -> str:
    fallback = keyword_title(members)
    if registry is None or not registry.providers_for(GENERATE):
        return fallback
    spread = list(members)[:: max(1, len(members) // 8)][:8]  # across the whole group (areas span many hubs)
    sample = "\n".join(f"- {claim_text(n)[:200]}" for n in spread)
    taken = ""
    if avoid:
        taken = "\n\nThese titles are already taken by other groups; choose a more specific one:\n" + "\n".join(
            f"- {t}" for t in avoid
        )
    try:
        raw = (await registry.call(
            GENERATE, _TITLE_SYSTEM, f"Statements:\n{sample}{taken}\n\nTitle:", max_tokens=24,
        )).value
    except SpecialistExhausted:
        return fallback
    return _valid_title(raw, members, fallback)


_PARENT_SYSTEM = (
    "You name a research area that groups several subtopics. Reply with a short area title of 2-6 words, in title "
    "case, broad enough to cover every subtopic, with no quotes and no punctuation at the end."
)


async def name_parent(
    child_titles: Sequence[str], members: Sequence[ZettelNote], registry: SpecialistRegistry | None,
    avoid: Sequence[str] = (),
) -> str:
    """Name a hub from its subtopics' titles: 8 sample claims cannot represent hundreds, their subtopics can."""
    fallback = keyword_title(members)
    if registry is None or not registry.providers_for(GENERATE):
        return fallback
    taken = ("\n\nAlready used elsewhere (do not reuse):\n" + "\n".join(f"- {t}" for t in avoid)) if avoid else ""
    listing = "\n".join(f"- {t}" for t in child_titles[:15])
    try:
        raw = (await registry.call(
            GENERATE, _PARENT_SYSTEM, f"Subtopics:\n{listing}{taken}\n\nArea title:", max_tokens=24,
        )).value
    except SpecialistExhausted:
        return fallback
    return _valid_title(raw, members, fallback, extra_vocabulary=" ".join(child_titles))


def _valid_title(raw: str, members: Sequence[ZettelNote], fallback: str, extra_vocabulary: str = "") -> str:
    """The model's title if it is short and shares vocabulary with the group, else the keyword fallback."""
    title = re.sub(r"[\"'\n].*$", "", raw.strip().splitlines()[0] if raw.strip() else "").strip(" .:*#")
    vocabulary = {t.lower() for n in members for t in content_tokens(claim_text(n))}
    vocabulary |= {t.lower() for t in content_tokens(extra_vocabulary)}
    words = title.split()
    if not (2 <= len(words) <= 8) or not any(w.lower().strip(",") in vocabulary for w in words):
        return fallback  # a title that shares no vocabulary with its members is not trustworthy
    return title


def _claims_block(members: Sequence[ZettelNote]) -> list[str]:
    lines = []
    for n in sorted(members, key=lambda n: (n.status != "canonical", n.id)):
        badge = " (corroborated)" if n.status == "canonical" else ""
        lines.append(f"- [[{n.id}]]{badge} {claim_text(n)}")
    return lines


def _hub_content(title: str, members: Sequence[ZettelNote], children: Sequence[ZettelNote] = (), sizes: dict[str, int] | None = None) -> str:
    lines = [f"Map of related claims: {title}.", ""]
    if children:
        lines += ["**Subtopics**", *[f"- [[{c.id}]] ({(sizes or {}).get(c.id, len(c.links))} claims)" for c in children], ""]
        lines += ["**Claims not in a subtopic**"] if members else []
    lines += _claims_block(members)
    return "\n".join(lines)


def _hub_id(members: Sequence[ZettelNote], depth: int) -> str:
    return f"hub-{depth}-" + re.sub(r"[^a-z0-9]+", "-", min(m.id for m in members).lower()).strip("-")[:40]


async def build_hubs(
    store: MemoryStore,
    notes: Sequence[ZettelNote],
    vectors: dict[str, Sequence[float]],
    *,
    registry: SpecialistRegistry | None = None,
    threshold: float = 0.5,
    min_size: int = 3,
    max_hub_size: int = 60,
    link_members: bool = True,
) -> list[ZettelNote]:
    """Create hub notes (idempotent: existing hubs for the same members are refreshed) and link each claim back to
    its most specific hub. Returns every hub note, parents before their subtopics.

    Hubs are named bottom-up: a leaf from sample claims, a parent from its subtopics' titles. Titles are unique
    across the vault: the namer sees its siblings' titles, and a clash that remains is disambiguated with the word
    most characteristic of the hub. With `link_members=False` claims are
    not touched (the caller writes claim links itself, as `aof.refine.graph` does).
    """
    tree = hub_tree(notes, vectors, threshold=threshold, min_size=min_size, max_size=max_hub_size)
    made: list[ZettelNote] = []
    used: set[str] = set()  # lowercased titles already given
    background: Counter[str] = Counter()
    for n in notes:
        background.update({t.lower() for t in content_tokens(claim_text(n)) if len(t) > 3})

    async def write(node: HubNode, depth: int, siblings: Sequence[str] = ()) -> ZettelNote:
        hub_id = _hub_id(node.members, depth)
        children: list[ZettelNote] = []
        for c in node.children:  # bottom-up: siblings see each other's titles, so they name different aspects
            children.append(await write(c, depth + 1, [h.title for h in children]))
        avoid = list(siblings)[-12:]
        if children:
            title = await name_parent([h.title for h in children], node.members, registry, avoid=avoid)
        else:
            title = await name_hub(node.members, registry, avoid=avoid)
        if title.lower() in used:
            word = distinct_word(node.members, background, exclude=title)
            title = f"{title}: {word}" if word else title
        base, k = title, 2
        while title.lower() in used:
            title, k = f"{base} ({k})", k + 1
        used.add(title.lower())
        direct = node.leftovers if node.children else node.members
        sizes = {c.id: len(child.members) for c, child in zip(children, node.children)}
        content = _hub_content(title, direct, children, sizes)
        links = [c.id for c in children] + [m.id for m in direct]
        sources = list(dict.fromkeys(u for m in node.members for u in m.sources))
        hub = await store.get_note(hub_id)
        tags = ["hub", f"hub-level:{depth}"]
        if hub is None:
            hub = await store.create_note(
                title, content, tags, links=links, agent_id=HUB_AGENT, note_id=hub_id,
                kind="hub", status="canonical", sources=sources,
            )
        else:
            hub.title, hub.content, hub.links, hub.sources, hub.tags = title, content, links, sources, tags
            hub.status = "canonical"
            await store.update_note(hub)
        if link_members:
            for m in direct:  # a claim links back to the most specific hub that lists it
                if hub_id not in m.links:
                    m.links.append(hub_id)
                    await store.update_note(m)
        made.append(hub)
        return hub

    tops: list[ZettelNote] = []
    for node in tree:
        tops.append(await write(node, 0, [h.title for h in tops]))
    made.sort(key=lambda h: h.tags[1] if len(h.tags) > 1 else "")  # parents (level 0) first, stable within a level
    await _retire_stale_hubs(store, {h.id for h in made}, unlink_members=link_members)
    return made


async def _retire_stale_hubs(store: MemoryStore, keep: set[str], *, unlink_members: bool = True) -> None:
    """Archive hubs left over from an earlier structure and unlink their claims (nothing is deleted)."""
    for hub in await store.get_notes_by_tags(["hub"], limit=100000, exclude_tags=["archived"]):
        if hub.id in keep:
            continue
        for member_id in hub.links if unlink_members else ():
            member = await store.get_note(member_id)
            if member is not None and hub.id in member.links:
                member.links = [i for i in member.links if i != hub.id]
                await store.update_note(member)
        hub.status = "archived"
        if "archived" not in hub.tags:
            hub.tags.append("archived")
        await store.update_note(hub)
