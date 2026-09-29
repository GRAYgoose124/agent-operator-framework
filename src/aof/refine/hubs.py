"""Hub (map-of-content) notes: group related claims and give each group a navigable entry point.

Hubs are extractive on purpose: a hub lists its member claims as links and quotes them; it does not add prose that
could drift from the sources. Only the *title* is model-written, and it must reuse vocabulary from its members.
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from typing import Sequence

from aof.memory.store import MemoryStore
from aof.memory.zettel import ZettelNote
from aof.refine.claims import cosine
from aof.refine.text import content_tokens
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
    groups: list[list[ZettelNote]] = []
    centroids: list[list[float]] = []

    def centroid(members: list[ZettelNote]) -> list[float]:
        dim = len(vectors[members[0].id])
        return [sum(vectors[m.id][d] for m in members) / len(members) for d in range(dim)]

    for note in order:
        best, best_sim = -1, threshold
        for gi, c in enumerate(centroids):
            sim = cosine(vectors[note.id], c)
            if sim >= best_sim:
                best, best_sim = gi, sim
        if best >= 0:
            groups[best].append(note)
            centroids[best] = centroid(groups[best])
        else:
            groups.append([note])
            centroids.append(list(vectors[note.id]))

    big = [g for g in groups if len(g) >= min_size]
    if not big:
        return []
    big_centroids = [centroid(g) for g in big]
    for g in groups:
        if len(g) < min_size:  # fold stragglers into the closest real hub rather than losing them
            for note in g:
                sims = [cosine(vectors[note.id], c) for c in big_centroids]
                nearest = max(range(len(big)), key=lambda i: sims[i])
                if sims[nearest] >= threshold * 0.8:
                    big[nearest].append(note)
    return big


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


def _hub_content(title: str, members: Sequence[ZettelNote]) -> str:
    lines = [f"Map of related claims: {title}.", ""]
    for n in sorted(members, key=lambda n: (n.status != "canonical", n.id)):
        badge = " (corroborated)" if n.status == "canonical" else ""
        lines.append(f"- [[{n.id}]]{badge} {claim_text(n)}")
    return "\n".join(lines)


async def build_hubs(
    store: MemoryStore,
    notes: Sequence[ZettelNote],
    vectors: dict[str, Sequence[float]],
    *,
    registry: SpecialistRegistry | None = None,
    threshold: float = 0.5,
    min_size: int = 3,
) -> list[ZettelNote]:
    """Create one hub note per cluster (idempotent: an existing hub for the same members is refreshed) and link
    each member back to its hub. Returns the hub notes."""
    hubs: list[ZettelNote] = []
    for members in cluster_notes(notes, vectors, threshold=threshold, min_size=min_size):
        title = await name_hub(members, registry)
        hub_id = "hub-" + re.sub(r"[^a-z0-9]+", "-", min(m.id for m in members).lower()).strip("-")[:40]
        hub = await store.get_note(hub_id)
        member_ids = [m.id for m in members]
        if hub is None:
            hub = await store.create_note(
                title, _hub_content(title, members), ["hub"], links=member_ids, agent_id=HUB_AGENT,
                note_id=hub_id, kind="hub", status="canonical",
                sources=list(dict.fromkeys(u for m in members for u in m.sources)),
            )
        else:
            hub.title, hub.content, hub.links = title, _hub_content(title, members), member_ids
            await store.update_note(hub)
        for m in members:
            if hub_id not in m.links:
                m.links.append(hub_id)
                await store.update_note(m)
        hubs.append(hub)
    return hubs
