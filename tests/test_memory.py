"""Tests for Zettelkasten memory system."""

import asyncio
import tempfile
from pathlib import Path

import pytest

from aof.config import MemoryConfig
from aof.memory.store import MemoryStore
from aof.memory.zettel import ZettelNote


def test_zettel_note_roundtrip():
    """Test that a note serializes to markdown and parses back correctly."""
    note = ZettelNote(
        id="20260211-test-abc123",
        title="Test Note",
        content="This is a test note with some content.",
        tags=["test", "unit"],
        links=["20260211-test-def456"],
        source="test",
        agent_id="agent01",
    )

    md = note.to_markdown()
    assert "---" in md
    assert "Test Note" in md
    assert "[[20260211-test-def456]]" in md

    parsed = ZettelNote.from_markdown(md)
    assert parsed.id == note.id
    assert parsed.title == note.title
    assert parsed.content == note.content
    assert parsed.tags == note.tags
    assert parsed.links == note.links
    assert parsed.agent_id == note.agent_id


def test_zettel_note_id_generation():
    id1 = ZettelNote.new_id("agent01", "content A")
    id2 = ZettelNote.new_id("agent01", "content B")
    assert id1 != id2
    assert "agen" in id1  # First 4 chars of agent_id


@pytest.mark.asyncio
async def test_memory_store_create_and_search():
    with tempfile.TemporaryDirectory() as tmpdir:
        config = MemoryConfig(
            db_path=str(Path(tmpdir) / "test.db"),
            notes_dir=str(Path(tmpdir) / "notes"),
        )
        store = MemoryStore(config)
        await store.initialize()

        note = await store.create_note(
            title="Python Tutorial",
            content="Python is a programming language used for web development.",
            tags=["python", "tutorial"],
            agent_id="test01",
        )

        assert note.id
        assert (Path(tmpdir) / "notes" / f"{note.id}.md").exists()

        # Search by content
        results = await store.search("programming")
        assert len(results) >= 1
        assert results[0].title == "Python Tutorial"

        # Search by tag
        results = await store.search_by_tag("python")
        assert len(results) >= 1

        # Search with tags filter
        results = await store.search("programming", limit=5, tags=["python"])
        assert len(results) >= 1
        assert all("python" in n.tags for n in results)

        count = await store.count()
        assert count == 1

        await store.close()


@pytest.mark.asyncio
async def test_memory_search_with_tags(tmp_path):
    """Search with tags filter returns only notes that have any of the given tags."""
    config = MemoryConfig(
        db_path=str(tmp_path / "test.db"),
        notes_dir=str(tmp_path / "notes"),
    )
    store = MemoryStore(config)
    await store.initialize()

    await store.create_note(
        title="Research Note",
        content="Findings about AI and machine learning.",
        tags=["research", "ai"],
        agent_id="test",
    )
    await store.create_note(
        title="Unrelated Note",
        content="Meeting notes and scheduling.",
        tags=["meeting"],
        agent_id="test",
    )

    results = await store.search("findings", limit=5, tags=["research"])
    assert len(results) == 1
    assert "research" in results[0].tags

    results = await store.search("findings", limit=5, tags=["meeting"])
    assert len(results) == 0

    results = await store.search("and", limit=5, tags=["research", "meeting"])
    assert len(results) == 2
    assert all(
        any(t in n.tags for t in ["research", "meeting"])
        for n in results
    )

    await store.close()


@pytest.mark.asyncio
async def test_search_citations_returns_empty_when_not_hybrid():
    """search_citations returns [] when memory strategy is not hybrid."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = MemoryConfig(
            db_path=str(Path(tmpdir) / "test.db"),
            notes_dir=str(Path(tmpdir) / "notes"),
            memory_strategy="search",
        )
        store = MemoryStore(config)
        await store.initialize()
        results = await store.search_citations("Python", limit=5)
        assert results == []
        await store.close()


@pytest.mark.asyncio
async def test_search_citations_hybrid(tmp_path):
    """search_citations returns citations when hybrid memory and citations are stored."""
    config = MemoryConfig(
        db_path=str(tmp_path / "test.db"),
        notes_dir=str(tmp_path / "notes"),
        memory_strategy="hybrid",
    )
    store = MemoryStore(config)
    await store.initialize()

    from aof.research.models import Citation

    citations = [
        Citation(url="https://a.com", title="Article A", snippet="About Python."),
        Citation(url="https://b.com", title="Article B", snippet="About Rust."),
    ]
    try:
        await store.add_citations_to_vector_store(citations, source_id="note1")
    except RuntimeError as e:  # embedding model must be downloaded from Hugging Face on first use
        await store.close()
        pytest.skip(f"embedding model unavailable (offline?): {e}")

    results = await store.search_citations("Python", limit=5)
    assert len(results) >= 1
    assert any("a.com" in r.get("url", "") or "Python" in r.get("snippet", "") for r in results)

    await store.close()


@pytest.mark.asyncio
async def test_research_note_with_citations_searchable():
    """Research notes with citations in content are searchable (Option A: citations embedded with note)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = MemoryConfig(
            db_path=str(Path(tmpdir) / "test.db"),
            notes_dir=str(Path(tmpdir) / "notes"),
        )
        store = MemoryStore(config)
        await store.initialize()

        content = (
            "Key findings: Python is widely used.\n\n"
            "## Sources\n"
            "- [Python.org](https://python.org): Official Python website."
        )
        await store.create_note(
            title="Research: Python",
            content=content,
            tags=["research"],
            agent_id="director",
        )

        results = await store.search("python.org")
        assert len(results) >= 1
        assert "Sources" in results[0].content or "python.org" in results[0].content

        await store.close()


@pytest.mark.asyncio
async def test_memory_store_recent():
    with tempfile.TemporaryDirectory() as tmpdir:
        config = MemoryConfig(
            db_path=str(Path(tmpdir) / "test.db"),
            notes_dir=str(Path(tmpdir) / "notes"),
        )
        store = MemoryStore(config)
        await store.initialize()

        for i in range(5):
            await store.create_note(
                title=f"Note {i}",
                content=f"Content for note {i}",
                tags=[f"tag{i}"],
                agent_id="test01",
            )

        recent = await store.get_recent(limit=3)
        assert len(recent) == 3

        await store.close()


@pytest.mark.asyncio
async def test_vector_store_add_and_search(tmp_path):
    """VectorStore add and search return expected results."""
    from aof.memory.vector_store import VectorStore

    vs = VectorStore(persist_directory=tmp_path, embedding_model="all-MiniLM-L6-v2")
    vs.add("n1", "Python is a programming language", {"title": "Python"})
    vs.add("n2", "Rust is a systems programming language", {"title": "Rust"})
    results = vs.search("Python programming", limit=2)
    assert len(results) >= 1
    assert results[0][0] == "n1"
    assert results[0][1] > 0


@pytest.mark.asyncio
async def test_memory_store_hybrid_search(tmp_path):
    """When memory_strategy=hybrid, search merges FTS and vector results."""
    config = MemoryConfig(
        db_path=str(tmp_path / "test.db"),
        notes_dir=str(tmp_path / "notes"),
        memory_strategy="hybrid",
        memory_search_limit=5,
    )
    store = MemoryStore(config)
    await store.initialize()

    await store.create_note(
        title="Machine Learning",
        content="Machine learning uses neural networks for pattern recognition.",
        tags=["ml"],
        agent_id="test",
    )
    await store.create_note(
        title="Deep Learning",
        content="Deep learning involves multiple layers of neural networks.",
        tags=["dl"],
        agent_id="test",
    )

    results = await store.search("neural networks", limit=5)
    assert len(results) >= 1
    assert any("neural" in n.content or "Neural" in n.title for n in results)

    await store.close()
