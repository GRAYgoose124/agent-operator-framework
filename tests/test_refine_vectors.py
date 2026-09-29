"""Vector maths: correctness against a reference, and scale (this used to crash CPython in a pure-Python loop)."""

from __future__ import annotations

import math
import random
import time

from aof.refine.claims import Claim
from aof.refine.dedupe import cluster_claims
from aof.refine.vectors import cosine, cosine_matrix, cosine_to, unit_rows


def _ref_cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    return dot / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)))


def test_cosine_functions_match_a_reference_implementation():
    rng = random.Random(0)
    vecs = [[rng.gauss(0, 1) for _ in range(16)] for _ in range(6)]
    m = cosine_matrix(vecs)
    for i in range(6):
        assert abs(m[i, i] - 1.0) < 1e-5
        for j in range(6):
            assert abs(m[i, j] - _ref_cosine(vecs[i], vecs[j])) < 1e-5
    assert abs(cosine(vecs[0], vecs[1]) - _ref_cosine(vecs[0], vecs[1])) < 1e-5
    got = cosine_to(vecs[0], vecs[1:])
    assert [round(float(x), 4) for x in got] == [round(_ref_cosine(vecs[0], v), 4) for v in vecs[1:]]


def test_zero_and_empty_inputs_are_safe():
    assert cosine([0.0, 0.0], [1.0, 2.0]) == 0.0
    assert len(cosine_to([1.0, 0.0], [])) == 0
    assert unit_rows([[0.0, 0.0]]).tolist() == [[0.0, 0.0]]


def test_clustering_hundreds_of_384d_vectors_is_fast_and_correct():
    rng = random.Random(1)
    centers = [[rng.gauss(0, 1) for _ in range(384)] for _ in range(3)]
    vectors, claims = [], []
    for k in range(300):
        c = centers[k % 3]
        vectors.append([x + rng.gauss(0, 0.05) for x in c])  # tight blobs around 3 centres
        claims.append(Claim(text=f"distinct claim number {k}", url=f"u{k}", title="t", source="pubmed", authority=3))
    started = time.time()
    clusters = cluster_claims(claims, vectors, threshold=0.9)
    assert time.time() - started < 5
    assert sorted(len(c) for c in clusters) == [100, 100, 100]
    assert sorted(i for c in clusters for i in c) == list(range(300))
