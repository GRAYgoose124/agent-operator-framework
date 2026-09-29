"""Tests for LangChain adapters: models, tools, memory_store."""

import asyncio
import tempfile
from pathlib import Path

import pytest

from aof.config import MemoryConfig, load_config
from aof.langchain_adapters.models import create_chat_model
from aof.langchain_adapters.tools import tool_registry_to_langchain
from aof.memory.store import MemoryStore
from aof.tools.registry import ToolDefinition, ToolRegistry


def test_create_chat_model():
    """create_chat_model returns a LangChain chat model."""
    config = load_config()
    model = create_chat_model(config)
    assert model is not None
    assert hasattr(model, "invoke")


def test_tool_registry_to_langchain():
    """tool_registry_to_langchain converts ToolRegistry to LangChain tools."""
    registry = ToolRegistry()

    async def dummy(q: str) -> str:
        return f"result: {q}"

    registry.register(
        ToolDefinition(
            name="dummy_tool",
            description="A dummy tool",
            parameters={
                "type": "object",
                "properties": {"q": {"type": "string"}},
                "required": ["q"],
            },
            handler=dummy,
        )
    )

    tools = tool_registry_to_langchain(registry)
    assert len(tools) == 1
    assert tools[0].name == "dummy_tool"
    assert tools[0].description == "A dummy tool"


def test_tool_registry_to_langchain_filtered():
    """tool_registry_to_langchain with tool_names filters correctly."""
    registry = ToolRegistry()
    for name in ["a", "b", "c"]:
        registry.register(
            ToolDefinition(name=name, description=name, parameters={}, handler=lambda: None)
        )

    tools = tool_registry_to_langchain(registry, tool_names=["a", "c"])
    assert len(tools) == 2
    assert {t.name for t in tools} == {"a", "c"}


@pytest.mark.asyncio
async def test_zettel_store_adapter():
    """ZettelStoreAdapter get/put/search works with MemoryStore."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = MemoryConfig(
            db_path=str(Path(tmpdir) / "test.db"),
            notes_dir=str(Path(tmpdir) / "notes"),
        )
        memory = MemoryStore(config)
        await memory.initialize()

        from aof.langchain_adapters.memory_store import ZettelStoreAdapter

        store = ZettelStoreAdapter(memory)

        # Put a note
        value = {
            "id": "test-note-1",
            "title": "Test",
            "content": "Hello world",
            "tags": ["test"],
            "links": [],
            "created_at": "2026-02-11T00:00:00+00:00",
            "updated_at": "2026-02-11T00:00:00+00:00",
            "source": "",
            "agent_id": "",
        }
        await store.aput(("zettel",), "test-note-1", value)

        # Get it back
        item = await store.aget(("zettel",), "test-note-1")
        assert item is not None
        assert item.value["title"] == "Test"
        assert "Hello world" in item.value["content"]

        # Search
        results = await store.asearch(("zettel",), query="Hello", limit=5)
        assert len(results) >= 1

        await memory.close()
