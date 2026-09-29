"""End-to-end tests for deep agent."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from aof.orchestrator.deep_agent_factory import create_aof_deep_agent


def test_agent_deep_creates(temp_config, temp_memory, temp_tools):
    """Create deep agent with mocked model; verify structure (invoke not run - FakeListChatModel lacks bind_tools)."""
    from langchain_core.language_models import FakeListChatModel

    with patch(
        "aof.orchestrator.deep_agent_factory.create_chat_model",
        return_value=FakeListChatModel(responses=["Done."]),
    ):
        agent = create_aof_deep_agent(temp_config, temp_tools, temp_memory)
        assert agent is not None
        assert hasattr(agent, "invoke")


@pytest.mark.asyncio
async def test_agent_deep_with_memory(temp_config, temp_memory, temp_tools):
    """Pre-populate memory, create agent; verify agent creation succeeds with memory store."""
    await temp_memory.create_note(
        title="Test Note",
        content="Python is used for agents.",
        tags=["test"],
        agent_id="e2e",
    )

    from langchain_core.language_models import FakeListChatModel

    with patch(
        "aof.orchestrator.deep_agent_factory.create_chat_model",
        return_value=FakeListChatModel(responses=["Done."]),
    ):
        agent = create_aof_deep_agent(temp_config, temp_tools, temp_memory)
        assert agent is not None
        assert hasattr(agent, "invoke")

    # Verify note exists in memory
    results = await temp_memory.search("Python", limit=5)
    assert len(results) >= 1
    assert any("Python" in n.content for n in results)
