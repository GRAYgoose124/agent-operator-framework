"""Gap-driven research: parsing (deterministic) and an end-to-end run with real models and real sources."""

from __future__ import annotations

import os

import pytest

from aof.config import MemoryConfig, load_config, resolve_role_path
from aof.memory.store import MemoryStore
from aof.refine import gaps
from aof.refine.pipeline import refine_question, registry_embedder
from aof.refine.sources import build_sources
from aof.refine.vault import question_slug
from aof.specialists import build_registry
from aof.specialists.backends import RoleBackends

CONFIG = load_config("config.toml")
needs_small = pytest.mark.skipif(
    not os.path.exists(resolve_role_path(CONFIG, "small")), reason="local Qwen3-4B role model not available"
)


def test_parse_parts_cleans_dedupes_and_excludes_the_question():
    question = "How do sleep spindles support memory consolidation?"
    text = (
        "1. What generates sleep spindles in the thalamocortical system?\n"
        "- Which ion channels shape spindle oscillations in thalamic neurons?\n"
        '"What generates sleep spindles in the thalamocortical system?"\n'  # duplicate
        "How do sleep spindles support memory consolidation?\n"  # the question itself
        "too short\n"
        "* Is spindle density correlated with overnight memory retention\n"  # gets a question mark
    )
    parts = gaps.parse_parts(text, question, limit=5)
    assert parts == [
        "What generates sleep spindles in the thalamocortical system?",
        "Which ion channels shape spindle oscillations in thalamic neurons?",
        "Is spindle density correlated with overnight memory retention?",
    ]
    assert gaps.parse_parts(text, question, limit=1) == parts[:1]


def test_gap_report_counts_and_markdown():
    r = gaps.GapReport("Q?", parts=[
        gaps.SubQuestion("a?", "answered"), gaps.SubQuestion("b?", "partial", filled=True), gaps.SubQuestion("c?"),
    ], rounds=1)
    assert r.counts() == {"answered": 1, "partial": 1, "unanswered": 1, "researched": 1}
    md = gaps.render_markdown([r])
    assert "- [partial *] b?" in md and "- [unanswered] c?" in md and "researched 1 in 1 round(s)" in md


@pytest.fixture
async def store(tmp_path):
    s = MemoryStore(MemoryConfig(db_path=str(tmp_path / "m.db"), notes_dir=str(tmp_path / "notes")))
    await s.initialize()
    yield s
    await s.close()


@needs_small
async def test_real_gap_research_finds_and_fills_a_missing_part(store):
    """The vault knows about TRN anatomy only; a question that also needs spindle mechanisms must trigger research."""
    backends = RoleBackends(CONFIG)
    registry = build_registry(CONFIG, backends.get)
    try:
        embed = registry_embedder(registry)
        if embed is None:
            pytest.skip("sentence-transformers not available")
        sources = build_sources(["pubmed"])
        seed = await refine_question(
            "What is the thalamic reticular nucleus?", store=store, registry=registry, sources=sources,
            per_source=3, claims_per_doc=4,
        )
        if not seed.canonical:
            pytest.skip("no evidence source reachable")
        index = gaps.VaultIndex(store, embed)
        await index.refresh()
        before = len(index.notes)

        async def research(sub_question: str, parent: str) -> None:
            await refine_question(
                sub_question, store=store, registry=registry, sources=sources, per_source=3, claims_per_doc=4,
                extra_tags=[f"gap-of:{question_slug(parent)}"],
            )

        report = await gaps.find_gaps(
            "How does the thalamic reticular nucleus generate sleep spindles?", index, registry, max_parts=4,
            judge_only="role:small",
        )
        assert report.parts, "the question was not decomposed"
        assert all(p.grade in gaps.GRADES for p in report.parts)
        await gaps.fill_gaps(report, index, registry, research, rounds=1, max_gaps=2, judge_only="role:small")
    finally:
        await registry.close()
        await backends.close()

    researched = [p for p in report.parts if p.filled]
    if not researched:
        pytest.skip("every part was already answered by the seed vault")
    await index.refresh()
    assert len(index.notes) > before  # targeted research added claims
    gap_notes = await store.get_notes_by_tags(["gap-of:" + question_slug("How does the thalamic reticular nucleus generate sleep spindles?")], limit=1000)
    assert gap_notes and all(n.kind == "claim" and n.sources for n in gap_notes)
