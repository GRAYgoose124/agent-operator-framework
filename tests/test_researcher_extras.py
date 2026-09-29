"""Tests for seed expansion, lateral thinking, and reflection filtering helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

from aof.agent.base import _is_meta_reflection
from aof.config import load_config, config_with_workspace
from aof.researcher.lateral_thinking import _parse_questions
from aof.researcher.seed_expansion import (
    _is_meta_question,
    _parse_questions_from_response,
    _topical_summary,
)


def test_seed_parse_json():
    text = '{"questions": ["What is consciousness?", "How does memory work?"]}'
    assert _parse_questions_from_response(text) == [
        "What is consciousness?",
        "How does memory work?",
    ]


def test_seed_parse_numbered_fallback():
    text = "1. What is the hard problem of consciousness?\n2) How do we measure awareness?\nnot a q"
    qs = _parse_questions_from_response(text)
    assert qs == [
        "What is the hard problem of consciousness?",
        "How do we measure awareness?",
    ]


def test_meta_question_detection():
    assert _is_meta_question("Okay, the user wants me to list questions")
    assert _is_meta_question("x" * 201)
    assert not _is_meta_question("What drives emergent behaviour in swarms?")


def test_topical_summary():
    assert "Philosophical" in _topical_summary(["A philosophical angle?", "b", "c"])
    assert _topical_summary(["one?"]) == "Follow-up questions generated from seed."


def test_lateral_parse_json_and_lines():
    assert _parse_questions('{"questions": ["Why is the sky blue?"]}') == ["Why is the sky blue?"]
    lines = "- What if entropy reversed?\n2. Could ants teach us routing?\nplain text"
    assert _parse_questions(lines) == ["What if entropy reversed?", "Could ants teach us routing?"]


def test_meta_reflection():
    assert _is_meta_reflection("Okay, the user wants me to summarise the result of the step.")
    assert not _is_meta_reflection("Searched three sources and stored citations.")
    assert not _is_meta_reflection("short")


def test_new_research_config_defaults_and_workspace_passthrough():
    config = load_config(None)
    r = config.research
    assert r.seed_expansion_role == "medium"
    assert r.seed_expansion_count == 30
    assert r.branch_connector_enabled is False
    assert r.lateral_thinking_enabled is False
    scoped = config_with_workspace(config, Path("data/workspaces/x").resolve())
    assert scoped.research.lateral_min_backlog == r.lateral_min_backlog
    assert scoped.research.branch_connector_interval == r.branch_connector_interval


def _local_model_available() -> bool:
    import os
    return os.path.exists(load_config("config.toml").model.path)


@pytest.mark.skipif(not _local_model_available(), reason="local GGUF model not available")
async def test_seed_expansion_is_idempotent_with_real_model(tmp_path):
    """Real-model run: seed expansion adds questions once and never duplicates on re-run."""
    from dataclasses import replace
    from aof.research import ResearchQueue
    from aof.researcher.seed_expansion import run_seed_expansion

    config = load_config("config.toml")
    config = replace(
        config,
        research=replace(config.research, seed_expansion_role="micro", seed_expansion_count=15),
        memory=replace(config.memory, db_path=str(tmp_path / "m.db"), notes_dir=str(tmp_path / "notes")),
    )
    queue_path = str(tmp_path / "queue.json")
    seed = "What limits the scaling of small language models?"

    await run_seed_expansion(config, queue_path, seed)
    first = [i.question.lower() for g in ResearchQueue(queue_path).list_all().values() for i in g]
    added_again = await run_seed_expansion(config, queue_path, seed)
    second = [i.question.lower() for g in ResearchQueue(queue_path).list_all().values() for i in g]

    assert seed.lower() in first
    assert len(first) == len(set(first))
    assert len(second) == len(set(second))
    assert added_again == len(second) - len(first)
