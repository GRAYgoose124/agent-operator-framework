"""Tests for agent memory retrieval and context building."""

import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from aof.agent.base import BaseAgent
from aof.config import MemoryConfig
from aof.memory.store import MemoryStore
from aof.tools.registry import ToolDefinition, ToolRegistry


@pytest.fixture
def mock_backend():
    backend = MagicMock()
    backend.model_info.return_value = {"n_ctx": 2048, "max_tokens": 512, "model_path": "qwen3"}
    return backend


@pytest.fixture
async def memory_with_notes():
    with tempfile.TemporaryDirectory() as tmpdir:
        config = MemoryConfig(
            db_path=str(Path(tmpdir) / "test.db"),
            notes_dir=str(Path(tmpdir) / "notes"),
            memory_strategy="search",
            memory_search_limit=5,
        )
        store = MemoryStore(config)
        await store.initialize()

        await store.create_note(
            title="Python Research",
            content="Python is used for web development and data science.",
            tags=["python", "research"],
            agent_id="test01",
        )
        await store.create_note(
            title="Agent Tips",
            content="Use tools when they help accomplish the task.",
            tags=["tips"],
            agent_id="test01",
        )

        yield store
        await store.close()


@pytest.mark.asyncio
async def test_build_messages_includes_memory(mock_backend, memory_with_notes):
    """Agent _build_messages retrieves and includes relevant memory notes."""
    tools = ToolRegistry()
    agent = BaseAgent(
        backend=mock_backend,
        memory=memory_with_notes,
        tools=tools,
    )
    agent.ctx.goal = "Research Python"

    messages = await agent._build_messages("Summarize Python research.")

    # Should have system message with memory context
    content = " ".join(m.get("content", "") for m in messages)
    assert "Python" in content
    assert "web development" in content or "Python Research" in content


@pytest.mark.asyncio
async def test_execute_step_tool_loop():
    """Execute step loops back to model when tool calls are returned."""
    from unittest.mock import AsyncMock

    call_count = 0

    async def mock_complete(messages, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return type("R", (), {"text": '<tool_call>{"name": "add", "arguments": {"a": 1, "b": 2}}</tool_call>', "tokens_used": 10})()
        return type("R", (), {"text": "The sum is 3.", "tokens_used": 5})()

    with tempfile.TemporaryDirectory() as tmpdir:
        config = MemoryConfig(
            db_path=str(Path(tmpdir) / "test.db"),
            notes_dir=str(Path(tmpdir) / "notes"),
        )
        store = MemoryStore(config)
        await store.initialize()

        backend = MagicMock()
        backend.model_info.return_value = {"n_ctx": 2048, "max_tokens": 512, "model_path": "qwen3"}
        backend.complete = AsyncMock(side_effect=mock_complete)

        tools = ToolRegistry()
        def add(a: int, b: int) -> int:
            return a + b
        tools.register(
            ToolDefinition(
                name="add",
                description="Add two numbers",
                parameters={"type": "object", "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}}},
                handler=add,
            )
        )

        agent = BaseAgent(backend=backend, memory=store, tools=tools, max_tool_rounds=5)
        agent.ctx.goal = "Compute 1+2"

        result = await agent.execute_step("Add 1 and 2")

        assert result.text == "The sum is 3."
        assert call_count == 2  # First call returned tool call, second returned final text
        assert agent.ctx.metrics.tool_calls_made == 1
        assert agent.ctx.metrics.tool_calls_succeeded == 1

        await store.close()


@pytest.mark.asyncio
async def test_build_messages_recent_strategy():
    """Agent _build_messages with recent strategy includes recent notes."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = MemoryConfig(
            db_path=str(Path(tmpdir) / "test.db"),
            notes_dir=str(Path(tmpdir) / "notes"),
            memory_strategy="recent",
            memory_search_limit=2,
        )
        store = MemoryStore(config)
        await store.initialize()

        await store.create_note(
            title="Recent A",
            content="Most recent note content.",
            tags=[],
            agent_id="a",
        )

        tools = ToolRegistry()
        backend = MagicMock()
        backend.model_info.return_value = {"n_ctx": 2048, "max_tokens": 512, "model_path": "qwen3"}
        agent = BaseAgent(backend=backend, memory=store, tools=tools)

        messages = await agent._build_messages("Any task")

        content = " ".join(m.get("content", "") for m in messages)
        assert "Recent A" in content or "recent" in content.lower()

        await store.close()
