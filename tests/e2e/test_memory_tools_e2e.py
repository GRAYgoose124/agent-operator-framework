"""End-to-end tests for memory tools (refine_memory, search_memory)."""

from __future__ import annotations

import pytest


@pytest.mark.asyncio
async def test_refine_memory_merge(temp_memory, temp_tools):
    """Create 2 notes, call refine_memory(action='merge', ...), verify new note exists with links."""
    n1 = await temp_memory.create_note(
        title="Note A",
        content="Content A",
        tags=["a"],
        agent_id="test",
    )
    n2 = await temp_memory.create_note(
        title="Note B",
        content="Content B",
        tags=["b"],
        agent_id="test",
    )

    result = await temp_tools.invoke(
        "refine_memory",
        {
            "action": "merge",
            "source_ids": [n1.id, n2.id],
            "title": "Merged AB",
            "content": "Combined content from A and B.",
        },
    )
    assert result.get("success") is True
    assert "note_id" in result
    merged_id = result["note_id"]

    merged = await temp_memory.get_note(merged_id)
    assert merged is not None
    assert merged.title == "Merged AB"
    assert "Combined content from A and B" in merged.content
    assert f"[[{n1.id}]]" in merged.content
    assert f"[[{n2.id}]]" in merged.content
    assert "refined" in merged.tags or "merged" in merged.tags
    assert n1.id in merged.source
    assert n2.id in merged.source


@pytest.mark.asyncio
async def test_refine_memory_update_tags(temp_memory, temp_tools):
    """Create note, call refine_memory(action='update_tags', ...), verify tags updated."""
    note = await temp_memory.create_note(
        title="Tagged Note",
        content="Some content",
        tags=["old"],
        agent_id="test",
    )

    result = await temp_tools.invoke(
        "refine_memory",
        {
            "action": "update_tags",
            "note_id": note.id,
            "tags": ["new", "updated"],
        },
    )
    assert result.get("success") is True
    assert result.get("tags") == ["new", "updated"]

    updated = await temp_memory.get_note(note.id)
    assert updated is not None
    assert set(updated.tags) == {"new", "updated"}


@pytest.mark.asyncio
async def test_search_memory(temp_memory, temp_tools):
    """Create notes, call search_memory; verify results."""
    await temp_memory.create_note(
        title="Python Notes",
        content="Python is a programming language.",
        tags=["python"],
        agent_id="test",
    )
    await temp_memory.create_note(
        title="Rust Notes",
        content="Rust is a systems language.",
        tags=["rust"],
        agent_id="test",
    )
    await temp_memory.create_note(
        title="Python Tips",
        content="Python best practices and tips.",
        tags=["python", "tips"],
        agent_id="test",
    )

    results = await temp_tools.invoke("search_memory", {"query": "Python", "limit": 5})
    assert isinstance(results, list)
    assert len(results) >= 2
    assert any("Python" in r.get("content", "") for r in results)
    for r in results:
        assert "id" in r
        assert "title" in r
        assert "content" in r
        assert "tags" in r


@pytest.mark.asyncio
async def test_search_memory_returns_most_relevant(temp_memory, temp_tools):
    """Create notes with different relevance; search and assert most relevant appears first."""
    await temp_memory.create_note(
        title="Python tutorial",
        content="Python tutorial for beginners. Learn Python step by step.",
        tags=["python", "tutorial"],
        agent_id="test",
    )
    await temp_memory.create_note(
        title="Rust basics",
        content="Rust programming basics and ownership model.",
        tags=["rust"],
        agent_id="test",
    )

    results = await temp_tools.invoke("search_memory", {"query": "Python tutorial", "limit": 5})
    assert isinstance(results, list)
    assert len(results) >= 1
    first_content = results[0].get("content", "") + results[0].get("title", "")
    assert "Python tutorial" in first_content or "tutorial" in first_content
