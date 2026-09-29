"""End-to-end tests for ResearchDirector."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from aof.agent.pool import AgentPool
from aof.inference.backend import CompletionResult, InferenceBackend
from aof.research.director import ResearchDirector
from aof.research.queue import ResearchQueue


def _make_director_backend() -> InferenceBackend:
    """Mock backend for director: plan returns steps, synthesize returns report."""
    from aof.inference.backend import InferenceBackend

    class DirectorBackend(InferenceBackend):
        def __init__(self):
            self._plan_json = AsyncMock(return_value={"steps": ["Crawl X", "Search Y"]})
            self._plan_complete = AsyncMock(
                return_value=CompletionResult(
                    text='{"steps": ["Crawl X", "Search Y"]}',
                    tokens_used=5,
                    finish_reason="stop",
                )
            )
            self._synthesize_complete = AsyncMock(
                return_value=CompletionResult(
                    text="Synthesized report with key findings from X and Y.",
                    tokens_used=10,
                    finish_reason="stop",
                )
            )
            self._call_count = 0

        def model_info(self):
            return {"n_ctx": 2048, "max_tokens": 512, "model_path": "mock"}

        async def complete(self, messages, **kwargs):
            self._call_count += 1
            if self._call_count <= 2:
                return await self._plan_complete(messages, **kwargs)
            return await self._synthesize_complete(messages, **kwargs)

        async def complete_json(self, messages, schema=None):
            return await self._plan_json(messages, schema=schema)

        async def start(self):
            pass

        async def shutdown(self):
            pass

    return DirectorBackend()


def _make_expert_backend() -> InferenceBackend:
    """Mock backend for experts: returns canned findings."""
    from aof.inference.backend import InferenceBackend

    class ExpertBackend(InferenceBackend):
        def __init__(self):
            self._complete = AsyncMock(
                return_value=CompletionResult(
                    text="Expert findings: X and Y.",
                    tokens_used=5,
                    finish_reason="stop",
                )
            )
            self._complete_json = AsyncMock(
                return_value={"steps": ["Gather information"]}
            )

        def model_info(self):
            return {"n_ctx": 2048, "max_tokens": 512, "model_path": "mock"}

        async def complete(self, messages, **kwargs):
            return await self._complete(messages, **kwargs)

        async def complete_json(self, messages, schema=None):
            return await self._complete_json(messages, schema=schema)

        async def start(self):
            pass

        async def shutdown(self):
            pass

    return ExpertBackend()


@pytest.mark.asyncio
async def test_research_director_process_item(
    temp_config, temp_memory, temp_tools, research_queue_fixture, mock_web_search
):
    """Director processes item; queue marked done; memory has note with findings."""
    queue = research_queue_fixture
    item_id = queue.add("What is Python?")

    director_backend = _make_director_backend()
    expert_backend = _make_expert_backend()

    pool = AgentPool(
        backend=expert_backend,
        memory=temp_memory,
        tools=temp_tools,
        config=temp_config,
    )
    pool.register_backend("default", expert_backend)
    pool.register_backend(temp_config.research.expert_role, expert_backend)

    interrupt_path = Path(temp_config.research.queue_path).parent / "test_interrupt.flag"
    director = ResearchDirector(
        queue=queue,
        pool=pool,
        director_backend=director_backend,
        memory=temp_memory,
        tools=temp_tools,
        config=temp_config,
        interrupt_path=interrupt_path,
    )

    with mock_web_search:
        item = queue.pop_next()
        assert item is not None
        done = await director.process_item(item)

    assert done is True
    item_after = queue._items.get(item_id)
    assert item_after is not None
    assert item_after.status.value == "done"

    assert await temp_memory.count() >= 1
    notes = await temp_memory.get_recent(limit=10)
    content_combined = " ".join(n.content for n in notes)
    assert "findings" in content_combined or "Synthesized" in content_combined


@pytest.mark.asyncio
async def test_research_director_citations_in_memory_note(
    temp_config, temp_memory, temp_tools, research_queue_fixture, mock_web_search
):
    """Director includes citation_entries in memory note when experts add citations."""
    from aof.tools.builtin import citation

    citation.register(temp_tools, research_queue_fixture)

    queue = research_queue_fixture
    item_id = queue.add("What is Python?")

    class ExpertBackendWithCitation(InferenceBackend):
        """Expert backend that returns add_citation tool call + findings."""

        def __init__(self):
            self._complete = AsyncMock(
                return_value=CompletionResult(
                    text='<tool_call>{"name":"add_citation","arguments":{"url":"https://python.org","title":"Python","snippet":"Python is a programming language."}}</tool_call>\nFindings: Python is widely used.',
                    tokens_used=20,
                    finish_reason="stop",
                )
            )
            self._complete_json = AsyncMock(return_value={"steps": ["Search"]})

        def model_info(self):
            return {"n_ctx": 2048, "max_tokens": 512, "model_path": "mock"}

        async def complete(self, messages, **kwargs):
            return await self._complete(messages, **kwargs)

        async def complete_json(self, messages, schema=None):
            return await self._complete_json(messages, schema=schema)

        async def start(self):
            pass

        async def shutdown(self):
            pass

    director_backend = _make_director_backend()
    expert_backend = ExpertBackendWithCitation()

    pool = AgentPool(
        backend=expert_backend,
        memory=temp_memory,
        tools=temp_tools,
        config=temp_config,
    )
    pool.register_backend("default", expert_backend)
    pool.register_backend(temp_config.research.expert_role, expert_backend)

    interrupt_path = Path(temp_config.research.queue_path).parent / "test_interrupt2.flag"
    director = ResearchDirector(
        queue=queue,
        pool=pool,
        director_backend=director_backend,
        memory=temp_memory,
        tools=temp_tools,
        config=temp_config,
        interrupt_path=interrupt_path,
    )

    with mock_web_search:
        item = queue.pop_next()
        assert item is not None
        done = await director.process_item(item)

    assert done is True
    item_after = queue._items.get(item_id)
    assert item_after is not None
    assert len(item_after.citation_entries) >= 1
    assert any("python.org" in c.url for c in item_after.citation_entries)

    notes = await temp_memory.get_recent(limit=10)
    content_combined = " ".join(n.content for n in notes)
    assert "Sources" in content_combined or "python.org" in content_combined
