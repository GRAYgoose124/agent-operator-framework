"""E2E tests for researcher runner: queue reload, consumption, artifact and synthesis note."""

from __future__ import annotations

from pathlib import Path

import pytest

from aof.config import config_with_workspace
from aof.memory.store import MemoryStore
from aof.research import ResearchQueue
from aof.researcher.runner import _make_item_done_callback
from aof.researcher.workspace import ResearcherWorkspace


def test_queue_reload_sees_external_updates(temp_dir):
    """Runner sees queue updates from REPL: second queue instance reloads and gets item added by first."""
    queue_path = str(temp_dir / "queue.json")
    queue1 = ResearchQueue(queue_path)
    item_id = queue1.add("Question from REPL")
    queue1._save()

    queue2 = ResearchQueue(queue_path)
    queue2.reload()
    item = queue2.pop_next()
    assert item is not None
    assert item.id == item_id
    assert item.question == "Question from REPL"


def test_queue_reload_sees_restart_done(temp_dir):
    """After restart_all_done in one instance, another instance sees backlog after reload."""
    queue_path = str(temp_dir / "queue.json")
    queue1 = ResearchQueue(queue_path)
    a = queue1.add("A")
    queue1.pop_next()
    queue1.mark_done(a)
    grouped = queue1.list_all()
    assert len(grouped["done"]) == 1

    queue2 = ResearchQueue(queue_path)
    queue2.reload()
    grouped2 = queue2.list_all()
    assert len(grouped2["done"]) == 1

    queue1.restart_all_done()
    queue2.reload()
    item = queue2.pop_next()
    assert item is not None
    assert item.id == a


def test_item_done_callback_writes_artifact_with_title_and_sources(temp_dir):
    """on_item_done with question and sources produces one artifact file with title and ## Sources."""
    from aof.researcher.runner import _make_item_done_callback

    workspace_root = Path(temp_dir) / "ws"
    workspace_root.mkdir(parents=True, exist_ok=True)
    (workspace_root / "artifacts").mkdir(exist_ok=True)
    workspace = ResearcherWorkspace(workspace_root)
    workspace.ensure_dirs()

    on_done = _make_item_done_callback(workspace)
    item_id = "abc123"
    content = "Summary: Python is a language.\n\nKey findings: It is popular."
    sources = [
        {"url": "https://example.com", "title": "Example", "snippet": "Snippet text"},
    ]
    on_done(item_id, content, sources, question="What is Python?")

    artifact_path = workspace.artifacts_dir / f"{item_id}.md"
    assert artifact_path.exists()
    text = artifact_path.read_text()
    assert text.startswith("# What is Python?")
    assert "## Sources" in text
    assert "https://example.com" in text
    assert "Example" in text
    assert "Snippet text" in text


@pytest.mark.asyncio
async def test_synthesis_note_created_with_expected_tags(temp_config, temp_dir):
    """Runner-style synthesis note creation yields a note with tags research, synthesis, item_id."""
    workspace_root = Path(temp_dir) / "researcher_workspace"
    workspace_root.mkdir(parents=True, exist_ok=True)
    (workspace_root / "memory").mkdir(exist_ok=True)
    config = config_with_workspace(temp_config, workspace_root)
    workspace_memory = MemoryStore(config.memory)
    await workspace_memory.initialize()
    try:
        item_id = "test-item-1"
        cohesive = "# What is Python?\n\nSummary: Python is a language.\n\nKey findings: It is popular."
        await workspace_memory.create_note(
            title="Research: What is Python?",
            content=cohesive,
            tags=["research", "synthesis", item_id],
            source=f"research:{item_id}",
            agent_id="pipeline",
        )
        notes = await workspace_memory.search("research synthesis", limit=20)
        synthesis_notes = [n for n in notes if "synthesis" in n.tags and item_id in n.tags]
        assert len(synthesis_notes) >= 1
    finally:
        await workspace_memory.close()
