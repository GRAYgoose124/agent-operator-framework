"""Tests for curated question sets (queue files)."""

from __future__ import annotations

from pathlib import Path

import pytest

from aof.research import ResearchQueue
from aof.research.queue_file import enqueue_question_set, load_question_set

EXAMPLES = Path(__file__).resolve().parent.parent / "examples" / "queues"


@pytest.mark.parametrize("path", sorted(EXAMPLES.glob("*.toml")), ids=lambda p: p.name)
def test_example_queue_files_are_valid(path):
    qset = load_question_set(path)
    assert qset.name and qset.description
    assert len(qset.questions) >= 10
    for spec in qset.questions:
        assert spec.question.endswith(("?", ".")), spec.id
        assert spec.key_facts, f"{spec.id} has no rubric key_facts"


def test_enqueue_skips_existing_and_is_idempotent(tmp_path):
    qset = load_question_set(EXAMPLES / "neuro_hippocampal_thalamic.toml")
    queue = ResearchQueue(tmp_path / "queue.json")
    queue.add(qset.questions[0].question.upper())  # differs only by case
    first = enqueue_question_set(queue, qset)
    second = enqueue_question_set(queue, qset)
    assert first == len(qset.questions) - 1
    assert second == 0
    assert sum(len(v) for v in queue.list_all().values()) == len(qset.questions)


def test_duplicate_ids_rejected(tmp_path):
    f = tmp_path / "q.toml"
    f.write_text('[[questions]]\nid="a"\nquestion="One?"\n[[questions]]\nid="a"\nquestion="Two?"\n')
    with pytest.raises(ValueError, match="duplicate"):
        load_question_set(f)
