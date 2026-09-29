"""Tests for discovery queue and tools."""

from __future__ import annotations

import pytest

from aof.discovery import DiscoveryQueue
from aof.research import ResearchQueue


@pytest.fixture
def discovery_queue_fixture(temp_config):
    """DiscoveryQueue with temp path."""
    return DiscoveryQueue(temp_config.discovery.queue_path)


def test_discovery_queue_add_list(discovery_queue_fixture: DiscoveryQueue):
    """Add seed, list, pop, mark done."""
    queue = discovery_queue_fixture
    item_id = queue.add("https://example.com")
    assert item_id

    grouped = queue.list_all()
    assert len(grouped["backlog"]) >= 1
    assert any(i.seed == "https://example.com" for i in grouped["backlog"])

    item = queue.pop_next()
    assert item is not None
    assert item.seed == "https://example.com"
    queue.mark_done(item.id)

    grouped = queue.list_all()
    assert len(grouped["done"]) >= 1


def test_discovery_queue_persistence(temp_config):
    """Add to queue, create new instance, verify item exists."""
    queue1 = DiscoveryQueue(temp_config.discovery.queue_path)
    item_id = queue1.add("Topic: Python 3.13")
    assert item_id

    queue2 = DiscoveryQueue(temp_config.discovery.queue_path)
    grouped = queue2.list_all()
    assert len(grouped["backlog"]) >= 1
    assert any(i.id == item_id for i in grouped["backlog"])


@pytest.mark.asyncio
async def test_add_to_research_queue_tool(temp_config, research_queue_fixture):
    """add_to_research_queue tool adds to research backlog."""
    from aof.tools.builtin import discovery_tools
    from aof.tools.registry import ToolRegistry

    tools = ToolRegistry()
    discovery_tools.register(tools, research_queue_fixture)

    result = await tools.invoke(
        "add_to_research_queue",
        {"question": "What is new in Python 3.13?"},
    )
    assert result.get("success") is True
    assert "item_id" in result

    grouped = research_queue_fixture.list_all()
    assert len(grouped["backlog"]) >= 1
    assert any("Python 3.13" in i.question for i in grouped["backlog"])
