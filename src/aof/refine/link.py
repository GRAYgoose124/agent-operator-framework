"""Link proposals between claim notes: semantic proximity plus shared, *specific* entities.

Links are mutual and only join live (non-archived) notes. Near-duplicates are not linked: those were already
merged (or would be flagged by the duplicate metric). A link means "related", nothing stronger.
"""

from __future__ import annotations

from collections import Counter
from typing import Sequence

import numpy as np

from aof.memory.store import MemoryStore
from aof.memory.zettel import ZettelNote
from aof.refine.vectors import cosine_matrix


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
    n = len(notes)
    if n < 2:
        return {x.id: set() for x in notes}
    sims = cosine_matrix([vectors[x.id] for x in notes])
    # Shared *specific* entities per pair, as a matrix product over an entity-incidence matrix.
    shared_ok = sorted(specific_entities(notes))
    entity_index = {e: i for i, e in enumerate(shared_ok)}
    incidence = np.zeros((n, max(1, len(shared_ok))), dtype=np.float32)
    for row, note in enumerate(notes):
        for e in entities_of(note) & set(shared_ok):
            incidence[row, entity_index[e]] = 1.0
    score = sims + entity_bonus * (incidence @ incidence.T)
    pairs = np.argwhere(np.triu((score >= min_sim) & (sims < max_sim), k=1))  # < max_sim: near-duplicates are not relations
    candidates: dict[str, list[tuple[float, str]]] = {x.id: [] for x in notes}
    for i, j in pairs:
        a_id, b_id, sc = notes[int(i)].id, notes[int(j)].id, float(score[i, j])
        candidates[a_id].append((sc, b_id))
        candidates[b_id].append((sc, a_id))
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
