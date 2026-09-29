"""Tests for citation tracking in research pipeline."""

from __future__ import annotations

import pytest

from aof.research.context import current_research_item_id
from aof.research.models import Citation, ResearchItem, ResearchItemStatus
from aof.research.queue import ResearchQueue


def test_citation_dataclass():
    """Citation stores url, title, snippet."""
    c = Citation(url="https://example.com", title="Example", snippet="A brief excerpt.")
    assert c.url == "https://example.com"
    assert c.title == "Example"
    assert c.snippet == "A brief excerpt."


def test_citation_to_dict():
    """Citation serializes to dict."""
    c = Citation(url="https://a.com", title="A", snippet="S")
    d = c.to_dict()
    assert d == {"url": "https://a.com", "title": "A", "snippet": "S"}


def test_citation_from_dict():
    """Citation deserializes from dict."""
    d = {"url": "https://b.com", "title": "B", "snippet": "B snippet"}
    c = Citation.from_dict(d)
    assert c.url == "https://b.com"
    assert c.title == "B"
    assert c.snippet == "B snippet"


def test_research_item_human_notes_serialization():
    """ResearchItem human_notes round-trip in to_dict/from_dict."""
    item = ResearchItem(
        id="abc123",
        question="Q?",
        human_notes=["Focus on X", "Skip Y"],
    )
    d = item.to_dict()
    assert d["human_notes"] == ["Focus on X", "Skip Y"]
    loaded = ResearchItem.from_dict(d)
    assert loaded.human_notes == ["Focus on X", "Skip Y"]


def test_research_item_citation_entries_serialization(temp_config):
    """ResearchItem citation_entries round-trip in to_dict/from_dict."""
    queue = ResearchQueue(temp_config.research.queue_path)
    item_id = queue.add("Test question")
    queue.add_citation_to_item(item_id, "https://x.com", "X", "Snippet X")
    queue.add_citation_to_item(item_id, "https://y.com", "Y", "Snippet Y")

    item = queue._items[item_id]
    assert len(item.citation_entries) == 2
    assert item.citation_entries[0].url == "https://x.com"
    assert item.citation_entries[1].title == "Y"

    d = item.to_dict()
    assert "citation_entries" in d
    assert len(d["citation_entries"]) == 2

    loaded = ResearchItem.from_dict(d)
    assert len(loaded.citation_entries) == 2
    assert loaded.citation_entries[0].url == "https://x.com"


def test_queue_add_citation_to_item(research_queue_fixture: ResearchQueue):
    """ResearchQueue.add_citation_to_item appends and persists."""
    queue = research_queue_fixture
    item_id = queue.add("What is Foobar?")
    item = queue.pop_next()
    assert item is not None

    queue.add_citation_to_item(item_id, "https://foo.com", "Foo", "Foo content")
    queue.add_citation_to_item(item_id, "https://bar.com", "Bar", "Bar content")

    item = queue._items[item_id]
    assert len(item.citation_entries) == 2
    assert item.citation_entries[0].url == "https://foo.com"
    assert item.citation_entries[1].title == "Bar"

    # Reload from disk
    queue2 = ResearchQueue(queue._path)
    item2 = queue2._items.get(item_id)
    assert item2 is not None
    assert len(item2.citation_entries) == 2


def test_queue_add_human_note(research_queue_fixture: ResearchQueue):
    """ResearchQueue.add_human_note appends and persists."""
    queue = research_queue_fixture
    item_id = queue.add("What is Foobar?")
    item = queue.pop_next()
    assert item is not None

    queue.add_human_note(item_id, "Focus on recent papers")
    queue.add_human_note(item_id, "Skip commercial sites")

    item = queue._items[item_id]
    assert item.human_notes == ["Focus on recent papers", "Skip commercial sites"]

    # Reload from disk
    queue2 = ResearchQueue(queue._path)
    item2 = queue2._items.get(item_id)
    assert item2 is not None
    assert item2.human_notes == ["Focus on recent papers", "Skip commercial sites"]


def test_queue_add_citation_to_nonexistent_item(research_queue_fixture: ResearchQueue):
    """add_citation_to_item on nonexistent item does nothing (no crash)."""
    queue = research_queue_fixture
    queue.add_citation_to_item("nonexistent", "https://x.com", "X", "S")
    assert "nonexistent" not in queue._items


def test_queue_save_checkpoint_and_mark_done_clears(research_queue_fixture: ResearchQueue):
    """save_checkpoint persists; mark_done clears checkpoint."""
    queue = research_queue_fixture
    item_id = queue.add("Checkpoint test")
    item = queue.pop_next()
    assert item is not None

    queue.save_checkpoint(
        item_id,
        sources_gathered=[{"url": "https://a.com", "title": "A", "snippet": "S"}],
        partial_findings="Partial findings here",
        step_index=1,
    )
    item = queue._items[item_id]
    assert item.checkpoint is not None
    assert item.checkpoint["partial_findings"] == "Partial findings here"
    assert len(item.checkpoint["sources_gathered"]) == 1

    queue.mark_done(item_id)
    item = queue._items[item_id]
    assert item.checkpoint is None


@pytest.mark.asyncio
async def test_add_citation_tool_with_context(temp_config, temp_memory, research_queue_fixture):
    """add_citation tool adds to item when context is set."""
    from aof.tools.builtin import citation
    from aof.tools.registry import ToolRegistry

    tools = ToolRegistry()
    citation.register(tools, research_queue_fixture)

    item_id = research_queue_fixture.add("Test")
    item = research_queue_fixture.pop_next()
    assert item is not None

    token = current_research_item_id.set(item_id)
    try:
        result = await tools.invoke(
            "add_citation",
            {"url": "https://test.com", "title": "Test Source", "snippet": "Test snippet"},
        )
        assert result.get("success") is True
        assert "Test Source" in str(result.get("message", ""))
    finally:
        current_research_item_id.reset(token)

    item = research_queue_fixture._items[item_id]
    assert len(item.citation_entries) == 1
    assert item.citation_entries[0].url == "https://test.com"


@pytest.mark.asyncio
async def test_add_citation_tool_without_context(research_queue_fixture):
    """add_citation tool returns error when no active research item."""
    from aof.tools.builtin import citation
    from aof.tools.registry import ToolRegistry

    tools = ToolRegistry()
    citation.register(tools, research_queue_fixture)

    # Do not set current_research_item_id - default is None
    result = await tools.invoke(
        "add_citation",
        {"url": "https://x.com", "title": "X", "snippet": "S"},
    )
    assert "error" in result
    assert "No active research item" in result["error"]
