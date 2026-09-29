"""End-to-end tests for agent lifecycle (plan -> execute -> reflect -> store)."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from aof.agent.base import BaseAgent
from aof.agent.lifecycle import AgentLifecycle
from aof.inference.backend import CompletionResult


@pytest.mark.asyncio
async def test_agent_lifecycle_stores_reflection(temp_memory, temp_tools, mock_backend):
    """Run AgentLifecycle; assert memory has reflection note with goal source and reflection content."""
    mock_backend._complete_json = AsyncMock(return_value={"steps": ["Research topic X"]})
    mock_backend._complete = AsyncMock(
        return_value=CompletionResult(
            text="Reflection: accomplished the research task.",
            tokens_used=5,
            finish_reason="stop",
        )
    )

    agent = BaseAgent(
        backend=mock_backend,
        memory=temp_memory,
        tools=temp_tools,
        system_prompt="Test agent.",
    )
    goal = "Research topic X"
    lifecycle = AgentLifecycle(agent, max_steps=1)
    await lifecycle.run(goal)

    assert await temp_memory.count() >= 1
    notes = await temp_memory.get_recent(limit=10)
    reflection_notes = [n for n in notes if "reflection" in n.tags and "goal:" in n.source]
    assert len(reflection_notes) >= 1
    assert any("accomplished" in n.content for n in reflection_notes)
