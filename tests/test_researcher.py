"""Tests for the continuous researcher module."""

from pathlib import Path

from aof.researcher.artifacts import log_activity, write_artifact, write_sources_artifact
from aof.researcher.workspace import ResearcherWorkspace, ensure_workspace


def test_researcher_workspace_paths(tmp_path):
    ws = ResearcherWorkspace(tmp_path / "foo")
    ws.ensure_dirs()
    assert ws.queue_path == ws.root / "queue.json"
    assert ws.artifacts_dir == ws.root / "artifacts"
    assert ws.activity_log_path == ws.root / "activity.log"
    assert ws.artifacts_dir.exists()


def test_ensure_workspace_creates_dirs(tmp_path):
    ws = ensure_workspace("test", root=tmp_path)
    assert ws.root == tmp_path / "test"
    assert ws.root.exists()
    assert ws.notes_dir.exists()
    assert ws.artifacts_dir.exists()


def test_write_artifact(tmp_path):
    write_artifact(tmp_path, "abc123", "# Report\n\nContent here.")
    p = tmp_path / "abc123.md"
    assert p.exists()
    assert "Report" in p.read_text()


def test_write_sources_artifact(tmp_path):
    sources = [
        {"url": "https://a.com", "title": "A", "snippet": "Snippet A"},
        {"url": "https://b.com", "title": "B", "snippet": "Snippet B"},
    ]
    write_sources_artifact(tmp_path, "abc123", sources)
    p = tmp_path / "abc123_sources.md"
    assert p.exists()
    text = p.read_text()
    assert "https://a.com" in text
    assert "Snippet A" in text


def test_log_activity(tmp_path):
    log_path = tmp_path / "activity.log"
    log_activity(log_path, "test_agent", "item_start", "Processing X")
    log_activity(log_path, "test_agent", "item_done", "Done")
    lines = log_path.read_text().strip().split("\n")
    assert len(lines) == 2
    assert "item_start" in lines[0]
    assert "Processing X" in lines[0]
    assert "item_done" in lines[1]
