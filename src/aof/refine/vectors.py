"""Vector maths for similarity, as NumPy matrix operations (no pure-Python inner loops)."""

from __future__ import annotations

from typing import Sequence

import numpy as np


def unit_rows(vectors: Sequence[Sequence[float]]) -> np.ndarray:
    """Rows scaled to unit length (zero rows stay zero)."""
    m = np.asarray(vectors, dtype=np.float32)
    if m.ndim == 1:
        m = m[None, :]
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return m / norms


def cosine_matrix(vectors: Sequence[Sequence[float]]) -> np.ndarray:
    """All-pairs cosine similarity, shape (n, n)."""
    u = unit_rows(vectors)
    return u @ u.T


def cosine_to(query: Sequence[float], vectors: Sequence[Sequence[float]]) -> np.ndarray:
    """Cosine similarity of one vector against many, shape (n,)."""
    if len(vectors) == 0:
        return np.zeros(0, dtype=np.float32)
    return unit_rows(vectors) @ unit_rows(query)[0]


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    return float(cosine_to(a, [b])[0])
