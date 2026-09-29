"""Near-duplicate clustering and canonical selection for claims."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from aof.refine.claims import Claim
from aof.refine.vectors import cosine_matrix
from aof.refine.text import lexical_overlap, normalize

_SECTION_RANK = {"conclusions": 3, "conclusion": 3, "results": 2, "result": 2, "discussion": 1}


class _UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, i: int) -> int:
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def union(self, a: int, b: int) -> None:
        self.parent[self.find(a)] = self.find(b)


def cluster_claims(
    claims: Sequence[Claim],
    vectors: Sequence[Sequence[float]] | None = None,
    *,
    threshold: float = 0.88,
    lexical_threshold: float = 0.7,
) -> list[list[int]]:
    """Group indices of claims that say (nearly) the same thing.

    Exact duplicates (after normalisation) always merge. Otherwise cosine >= `threshold` on `vectors`,
    or, without vectors, token overlap >= `lexical_threshold`. Clusters keep input order.
    """
    n = len(claims)
    uf = _UnionFind(n)
    by_key: dict[str, int] = {}
    for i, c in enumerate(claims):
        key = normalize(c.text)
        if key in by_key:
            uf.union(i, by_key[key])
        else:
            by_key[key] = i
    if vectors is not None:
        sims = cosine_matrix(vectors)
        for i, j in np.argwhere(np.triu(sims >= threshold, k=1)):
            uf.union(int(i), int(j))
    else:
        for i in range(n):
            for j in range(i + 1, n):
                if uf.find(i) != uf.find(j) and lexical_overlap(claims[i].text, claims[j].text) >= lexical_threshold:
                    uf.union(i, j)
    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(uf.find(i), []).append(i)
    return sorted(groups.values(), key=lambda g: g[0])


def pick_canonical(claims: Sequence[Claim], members: Sequence[int]) -> int:
    """Best representative of a cluster: authority, then conclusion/result sections, then relevance."""
    return max(
        members,
        key=lambda i: (
            claims[i].authority,
            _SECTION_RANK.get(claims[i].section, 0),
            claims[i].relevance,
            -len(claims[i].text),  # prefer the tighter statement on a tie
        ),
    )
