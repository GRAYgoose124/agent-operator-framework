"""Link proposals between claim notes: semantic proximity plus shared, *specific* entities.

Links are mutual and only join live (non-archived) notes. Near-duplicates are not linked: those were already
merged (or would be flagged by the duplicate metric). A link means "related", nothing stronger.
"""

from __future__ import annotations

from collections import Counter
from typing import Sequence

from aof.memory.store import MemoryStore
from aof.memory.zettel import ZettelNote
from aof.refine.claims import cosine


def entities_of(note: ZettelNote) -> set[str]:
    return {t.split(":", 1)[1] for t in note.tags if t.startswith("entity:")}


def specific_entities(notes: Sequence[ZettelNote], max_share: float = 0.3) -> set[str]:
    """Entities that appear in at least two notes but not in more than `max_share` of them.

    An entity present everywhere (the vault's own subject, e.g. "trn") connects everything, which is no signal.
    """
    counts: Counter[str] = Counter()
    for n in notes:
        counts.update(entities_of(n))
    limit = max(2, int(len(notes) * max_share))
    return {e for e, c in counts.items() if 2 <= c <= limit}


def propose_links(
    notes: Sequence[ZettelNote],
    vectors: dict[str, Sequence[float]],
    *,
    min_sim: float = 0.55,
    max_sim: float = 0.9,
    entity_bonus: float = 0.1,
    per_note: int = 3,
) -> dict[str, set[str]]:
    """Mutual link sets {note id: {linked ids}} with at most ~`per_note` outgoing links proposed per note."""
    shared_ok = specific_entities(notes)
    ents = {n.id: entities_of(n) & shared_ok for n in notes}
    candidates: dict[str, list[tuple[float, str]]] = {n.id: [] for n in notes}
    for i, a in enumerate(notes):
        for b in notes[i + 1:]:
            sim = cosine(vectors[a.id], vectors[b.id])
            if sim >= max_sim:
                continue  # a near-duplicate, not a relation
            score = sim + entity_bonus * len(ents[a.id] & ents[b.id])
            if score >= min_sim:
                candidates[a.id].append((score, b.id))
                candidates[b.id].append((score, a.id))
    links: dict[str, set[str]] = {n.id: set() for n in notes}
    for nid, options in candidates.items():
        for _, other in sorted(options, reverse=True)[:per_note]:
            links[nid].add(other)
            links[other].add(nid)
    return links


async def apply_links(store: MemoryStore, notes: Sequence[ZettelNote], links: dict[str, set[str]]) -> int:
    """Merge proposed links into the notes (existing links are kept). Returns the number of notes changed."""
    changed = 0
    for note in notes:
        merged = list(dict.fromkeys([*note.links, *sorted(links.get(note.id, ()))]))
        if merged != note.links:
            note.links = merged
            await store.update_note(note)
            changed += 1
    return changed
