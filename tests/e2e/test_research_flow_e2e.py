"""End-to-end tests for research add + run flow producing memory."""

from __future__ import annotations

from pathlib import Path

import pytest

from aof.pipeline.compose import Pipeline


EXAMPLES_DIR = Path(__file__).resolve().parent.parent.parent / "examples"


@pytest.mark.asyncio
async def test_research_add_run_produces_memory(
    temp_config,
    temp_memory,
    temp_tools,
    research_queue_fixture,
    mock_backend,
    mock_web_search,
):
    """Add item to queue, run pipeline (legacy), assert item done and memory has note."""
    queue = research_queue_fixture
    item_id = queue.add("What is Python?")
    assert item_id

    with mock_web_search:
        pipeline = Pipeline.from_toml(
            EXAMPLES_DIR / "research.toml",
            mock_backend,
            temp_memory,
            temp_tools,
            max_tool_rounds=2,
        )
        item = queue.pop_next()
        assert item is not None
        result = await pipeline.run(item.question)
        queue.mark_done(item.id)

    assert result.steps_completed >= 1
    assert await temp_memory.count() >= 1
    item_after = queue._items.get(item_id)
    assert item_after is not None
    assert item_after.status.value == "done"
