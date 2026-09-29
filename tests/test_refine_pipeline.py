"""Refine pipeline: lossless merge logic (deterministic) and an end-to-end run on real scholarly sources."""

from __future__ import annotations

import os
import re

import pytest

from aof.config import MemoryConfig, load_config, resolve_role_path
from aof.memory.store import MemoryStore
from aof.refine.claims import Claim, candidate_sentences, extract_claims
from aof.refine.dedupe import cluster_claims, pick_canonical
from aof.refine.pipeline import refine_question
from aof.refine.sources import SourceDoc, build_sources
from aof.refine.vault import store_clusters
from aof.specialists import build_registry
from aof.specialists.backends import RoleBackends


def _claim(text, url, *, authority=3, section="", relevance=0.5, title="T", year="2020"):
    return Claim(text=text, url=url, title=title, source="pubmed", authority=authority,
                 year=year, section=section, relevance=relevance)


@pytest.fixture
async def store(tmp_path):
    s = MemoryStore(MemoryConfig(db_path=str(tmp_path / "m.db"), notes_dir=str(tmp_path / "notes")))
    await s.initialize()
    yield s
    await s.close()


# -- extraction and clustering -----------------------------------------------------------------------

def test_candidate_sentences_drop_methods_questions_and_fragments():
    doc = SourceDoc("u", "t", (
        "Background: The TRN is a thin shell of GABAergic neurons surrounding the dorsal thalamus. "
        "Methods: We recorded from 42 mice using patch clamp electrodes in acute slices. "
        "Results: TRN neurons fire bursts that pace sleep spindles in thalamocortical circuits. "
        "Is it so? Yes. Conclusions: TRN dysfunction may contribute to absence seizures in rodent models."
    ), "pubmed")
    got = candidate_sentences(doc)
    assert [s for s, _ in got] == [
        "The TRN is a thin shell of GABAergic neurons surrounding the dorsal thalamus.",
        "TRN neurons fire bursts that pace sleep spindles in thalamocortical circuits.",
        "TRN dysfunction may contribute to absence seizures in rodent models.",
    ]
    assert [sec for _, sec in got] == ["background", "results", "conclusions"]


async def test_extract_claims_lexical_fallback_ranks_relevant_first():
    doc = SourceDoc("u", "t", (
        "The thalamic reticular nucleus is a GABAergic structure that gates thalamocortical transmission. "
        "The city council approved a new budget for road maintenance during the winter months."
    ), "pubmed")
    claims = await extract_claims(doc, "What is the thalamic reticular nucleus?", embed=None, top_k=2, min_relevance=0.1)
    assert [c.text.split()[1] for c in claims] == ["thalamic"]  # the off-topic sentence is filtered out


def test_cluster_exact_and_lexical_duplicates():
    claims = [
        _claim("The TRN is a GABAergic shell around the thalamus.", "a"),
        _claim("The TRN is a GABAergic shell around the thalamus!", "b"),  # same after normalisation
        _claim("Grid cells fire in hexagonal patterns in entorhinal cortex.", "c"),
    ]
    assert cluster_claims(claims) == [[0, 1], [2]]


def test_pick_canonical_prefers_authority_then_section_then_relevance():
    claims = [
        _claim("low authority", "a", authority=1, section="conclusions", relevance=0.9),
        _claim("high authority background", "b", authority=3, section="background", relevance=0.2),
        _claim("high authority conclusion", "c", authority=3, section="conclusions", relevance=0.1),
    ]
    assert pick_canonical(claims, [0, 1, 2]) == 2


# -- lossless storage --------------------------------------------------------------------------------

async def test_store_clusters_is_lossless_and_links_both_ways(store):
    same = "The TRN is a GABAergic shell around the dorsal thalamus."
    claims = [
        _claim(same, "https://a.org/1", section="results"),
        _claim(same, "https://b.org/2", section="background"),  # identical sentence, different source
        _claim("Spindles are 11-16 Hz oscillations generated in thalamocortical loops.", "https://c.org/3"),
    ]
    canonical, archived = await store_clusters(store, "What is the TRN?", claims, [[0, 1], [2]])
    assert len(canonical) == 2 and len(archived) == 1  # 3 claims -> nothing dropped

    merged = next(n for n in canonical if same in n.content)
    assert merged.status == "canonical" and set(merged.sources) == {"https://a.org/1", "https://b.org/2"}
    assert merged.supersedes == [archived[0].id]
    assert "https://b.org/2" in merged.content  # every source's quote survives in the merged note

    loser = await store.get_note(archived[0].id)
    assert loser.status == "archived" and loser.superseded_by == merged.id
    assert loser.sources == ["https://b.org/2"] and "archived" in loser.tags

    single = next(n for n in canonical if n is not merged)
    assert single.status == "refined" and single.supersedes == []


# -- real end-to-end ---------------------------------------------------------------------------------

def _squash(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


async def test_real_pipeline_on_scholarly_sources_is_grounded_and_lossless(store):
    config = load_config("config.toml")
    backends = RoleBackends(config)
    registry = build_registry(config, backends.get)
    question = "What is the thalamic reticular nucleus and how does it generate sleep spindles?"
    try:
        report = await refine_question(
            question, store=store, registry=registry, sources=build_sources(["pubmed", "openalex"]),
            topics=["hippocampus", "thalamus", "cortex", "sleep"], per_source=4, claims_per_doc=4,
        )
    finally:
        await registry.close()
        await backends.close()
    if not report.documents:
        pytest.skip("no evidence source reachable")

    assert report.canonical, report.summary()
    # Partition: every extracted claim is in exactly one cluster
    assert sorted(i for c in report.clusters for i in c) == list(range(len(report.claims)))

    corpus = " ".join(_squash(d.text) for d in report.documents)
    urls = {d.url for d in report.documents}
    for note in [*report.canonical, *report.archived]:
        if note.status == "archived":
            quotes = [note.content]
        else:  # the headline may be a verified rewrite; the evidence quotes must be verbatim source text
            quotes = re.findall(r'^- "(.*)" — ', note.content, flags=re.MULTILINE)
            assert quotes, f"canonical note has no evidence quotes: {note.content!r}"
        for quote in quotes:
            assert _squash(quote) in corpus, f"evidence is not a verbatim source quote: {quote!r}"
        assert note.sources and set(note.sources) <= urls

    canon_ids = {n.id for n in report.canonical}
    for note in report.archived:
        assert note.superseded_by in canon_ids
    listed = {i for n in report.canonical for i in n.supersedes}
    assert listed == {n.id for n in report.archived}

    got = await store.get_note(report.canonical[0].id)
    assert got.kind == "claim" and got.status in ("refined", "canonical")
    assert got.sources == report.canonical[0].sources


def test_paper_meta_sentences_are_not_claims():
    from aof.refine.text import is_paper_meta

    for meta in (
        "In this review we focus on the thalamic reticular nucleus providing evidence for a hub.",
        "In this manuscript, we review the development of the thalamocortical system.",
        "[HYPOTHESES]: The present study hypothesizes that the TRN plays a pivotal role.",
        "Summary: We speculate that deficits in the thalamocortical system are reflected in spindles.",
        "Recent Findings: Recent work in humans has shown alterations in sleep spindles.",
        "This review summarizes recent studies on cholinergic transmission in the TRN.",
    ):
        assert is_paper_meta(meta), meta
    for finding in (
        "We found that, in all nuclei, the TRN provides GABAergic input primarily to relay cells (93-100%).",
        "Our results suggest that TRN terminals modulate thalamocortical transmission.",
        "Sleep spindles are characteristic EEG signatures of stage 2 NREM sleep.",
    ):
        assert not is_paper_meta(finding), finding


def test_claim_quota_scales_with_document_length_but_is_capped():
    from aof.refine.claims import claim_quota

    abstract = SourceDoc("u", "t", "x" * 1500, "pubmed")
    article = SourceDoc("u", "t", "x" * 9000, "wikipedia")
    huge = SourceDoc("u", "t", "x" * 200000, "wikipedia")
    assert claim_quota(abstract, 6) == 6
    assert claim_quota(article, 6) == 7
    assert claim_quota(huge, 6) == 24  # capped at 4x the base
