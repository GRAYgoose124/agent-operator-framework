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

    Deterministic for a given input order (notes are processed most-linked first, then by id).
    """
    order = sorted(notes, key=lambda n: (-len(n.links), n.id))
    unit = {n.id: unit_rows(vectors[n.id])[0] for n in notes}
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


def hub_tree(
    notes: Sequence[ZettelNote],
    vectors: dict[str, Sequence[float]],
    *,
    threshold: float = 0.5,
    min_size: int = 3,
    max_size: int = 60,
    step: float = 0.12,
    max_depth: int = 3,
) -> list[HubNode]:
    """Cluster into hubs, then split any hub larger than `max_size` into subtopic hubs at a stricter threshold."""

    def build(members: Sequence[ZettelNote], thr: float, depth: int) -> list[HubNode]:
        nodes = []
        for group in cluster_notes(members, vectors, threshold=thr, min_size=min_size):
            node = HubNode(members=group)
            if len(group) > max_size and depth < max_depth:
                node.children = build(group, thr + step, depth + 1)
                in_child = {n.id for c in node.children for n in c.members}
                node.leftovers = [n for n in group if n.id not in in_child]
                if len(node.children) < 2:  # splitting achieved nothing: keep it flat
                    node.children, node.leftovers = [], []
            nodes.append(node)
        return nodes

    return build(list(notes), threshold, 0)


def keyword_title(members: Sequence[ZettelNote], limit: int = 4) -> str:
    """Fallback title from the most frequent content words across a group's claims."""
    counts: Counter[str] = Counter()
    for n in members:
        counts.update({t.lower() for t in content_tokens(claim_text(n)) if len(t) > 3})
    top = [w for w, _ in counts.most_common(limit)]
    return " ".join(w.capitalize() for w in top) or "Related claims"


async def name_hub(members: Sequence[ZettelNote], registry: SpecialistRegistry | None) -> str:
    fallback = keyword_title(members)
    if registry is None or not registry.providers_for(GENERATE):
        return fallback
    sample = "\n".join(f"- {claim_text(n)[:200]}" for n in members[:8])
    try:
        raw = (await registry.call(GENERATE, _TITLE_SYSTEM, f"Statements:\n{sample}\n\nTitle:", max_tokens=24)).value
    except SpecialistExhausted:
        return fallback
    title = re.sub(r"[\"'\n].*$", "", raw.strip().splitlines()[0] if raw.strip() else "").strip(" .:")
    vocabulary = {t.lower() for n in members for t in content_tokens(claim_text(n))}
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
) -> list[ZettelNote]:
    """Create hub notes (idempotent: existing hubs for the same members are refreshed) and link each claim back to
    its most specific hub. Returns every hub note, parents before their subtopics."""
    tree = hub_tree(notes, vectors, threshold=threshold, min_size=min_size, max_size=max_hub_size)
    made: list[ZettelNote] = []

    async def write(node: HubNode, depth: int) -> ZettelNote:
        title = await name_hub(node.members, registry)
        hub_id = _hub_id(node.members, depth)
        children = [await write(c, depth + 1) for c in node.children]
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
            await store.update_note(hub)
        for m in direct:  # a claim links back to the most specific hub that lists it
            if hub_id not in m.links:
                m.links.append(hub_id)
                await store.update_note(m)
        made.append(hub)
        return hub

    for node in tree:
        await write(node, 0)
    made.sort(key=lambda h: h.tags[1] if len(h.tags) > 1 else "")  # parents (level 0) first, stable within a level
    await _retire_stale_hubs(store, {h.id for h in made})
    return made


async def _retire_stale_hubs(store: MemoryStore, keep: set[str]) -> None:
    """Archive hubs left over from an earlier structure and unlink their claims (nothing is deleted)."""
    for hub in await store.get_notes_by_tags(["hub"], limit=100000, exclude_tags=["archived"]):
        if hub.id in keep:
            continue
        for member_id in hub.links:
            member = await store.get_note(member_id)
            if member is not None and hub.id in member.links:
                member.links = [i for i in member.links if i != hub.id]
                await store.update_note(member)
        hub.status = "archived"
        if "archived" not in hub.tags:
            hub.tags.append("archived")
        await store.update_note(hub)
