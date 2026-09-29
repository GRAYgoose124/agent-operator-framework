"""End-to-end tests for research queue."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from aof.research import ResearchQueue


def test_research_queue_add_list_run(research_queue_fixture: ResearchQueue):
    """Add item, list (see in backlog), run with --limit 1 and mocked model."""
    item_id = research_queue_fixture.add("What is Python?")
    assert item_id

    grouped = research_queue_fixture.list_all()
    assert len(grouped["backlog"]) >= 1
    backlog_ids = [i.id for i in grouped["backlog"]]
    assert item_id in backlog_ids

    item = research_queue_fixture.pop_next()
    assert item is not None
    assert item.question == "What is Python?"
    research_queue_fixture.mark_done(item.id)

    grouped_after = research_queue_fixture.list_all()
    done_ids = [i.id for i in grouped_after["done"]]
    assert item.id in done_ids


def test_research_queue_persistence(temp_config):
    """Add to queue, create new queue instance from same path, verify item exists."""
    path = temp_config.research.queue_path
    queue1 = ResearchQueue(path)
    item_id = queue1.add("Persistence test question")
    assert item_id

    queue2 = ResearchQueue(path)
    grouped = queue2.list_all()
    assert len(grouped["backlog"]) >= 1
    found = any(i.id == item_id for i in grouped["backlog"])
    assert found


def test_restart_done_item(research_queue_fixture: ResearchQueue):
    """restart_done moves a done item to backlog and can add a refinement note."""
    item_id = research_queue_fixture.add("Question to re-run")
    item = research_queue_fixture.pop_next()
    assert item is not None
    research_queue_fixture.mark_done(item.id)
    grouped = research_queue_fixture.list_all()
    assert item_id in [i.id for i in grouped["done"]]

    ok = research_queue_fixture.restart_done(item_id, refinement_note="Focus on recent papers")
    assert ok is True
    grouped = research_queue_fixture.list_all()
    assert item_id in [i.id for i in grouped["backlog"]]
    restarted = research_queue_fixture._items[item_id]
    assert "Focus on recent papers" in restarted.human_notes

    # Idempotent on non-done: no-op, returns False
    research_queue_fixture.pop_next()
    research_queue_fixture.mark_done(item_id)
    ok2 = research_queue_fixture.restart_done(item_id)
    assert ok2 is True
    ok3 = research_queue_fixture.restart_done(item_id)
    assert ok3 is False  # already backlog


def test_restart_all_done(research_queue_fixture: ResearchQueue):
    """restart_all_done moves all done items to backlog."""
    a = research_queue_fixture.add("A")
    b = research_queue_fixture.add("B")
    for _ in range(2):
        item = research_queue_fixture.pop_next()
        if item:
            research_queue_fixture.mark_done(item.id)
    grouped = research_queue_fixture.list_all()
    assert len(grouped["done"]) == 2

    n = research_queue_fixture.restart_all_done(refinement_note="Re-run with updated sources")
    assert n == 2
    grouped = research_queue_fixture.list_all()
    assert len(grouped["done"]) == 0
    assert len(grouped["backlog"]) == 2
    assert "Re-run with updated sources" in research_queue_fixture._items[a].human_notes
    assert "Re-run with updated sources" in research_queue_fixture._items[b].human_notes
