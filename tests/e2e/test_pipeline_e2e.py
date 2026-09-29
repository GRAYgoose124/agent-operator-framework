"""End-to-end tests for pipeline execution."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from aof.config import load_config
from aof.inference.backend import CompletionResult
from aof.memory.store import MemoryStore
from aof.pipeline.compose import Pipeline


EXAMPLES_DIR = Path(__file__).resolve().parent.parent.parent / "examples"


@pytest.fixture
def research_toml_path():
    return EXAMPLES_DIR / "research.toml"


@pytest.mark.asyncio
async def test_pipeline_step_with_tools_records_tool_calls(
    research_toml_path, temp_config, temp_memory, temp_tools, mock_backend, mock_web_search
):
    """Run pipeline with a tool-enabled step; mock returns one tool call then text. Assert step_tool_calls > 0."""
    # Step 1 (research): complete (tool call) -> complete (after tool) -> reflect. Step 2 (synthesize): execute -> reflect.
    mock_backend._complete = AsyncMock(
        side_effect=[
            CompletionResult(
                text='<tool_call>{"name": "web_search", "arguments": {"query": "Python programming"}}</tool_call>',
                tokens_used=20,
                finish_reason="stop",
            ),
            CompletionResult(
                text="Findings: Python is widely used for automation and data science.",
                tokens_used=15,
                finish_reason="stop",
            ),
            CompletionResult(text="Reflection: Gathered findings.", tokens_used=5, finish_reason="stop"),
            CompletionResult(
                text="Report: Python is a key language for research and engineering.",
                tokens_used=10,
                finish_reason="stop",
            ),
            CompletionResult(text="Reflection: Wrote report.", tokens_used=5, finish_reason="stop"),
        ]
    )
    with mock_web_search:
        pipeline = Pipeline.from_toml(
            research_toml_path,
            mock_backend,
            temp_memory,
            temp_tools,
            max_tool_rounds=2,
        )
        result = await pipeline.run("Python programming")

    assert result.steps_completed == 2
    step_tool_calls = result.metadata.get("step_tool_calls") or []
    assert len(step_tool_calls) >= 1, "pipeline should record per-step tool call counts"
    assert sum(step_tool_calls) >= 1, "at least one step should have made a tool call"


@pytest.mark.asyncio
async def test_pipeline_legacy_runs(research_toml_path, temp_config, temp_memory, temp_tools, mock_backend, mock_web_search):
    """Load research.toml, run with legacy backend and mocked LLM."""
    with mock_web_search:
        pipeline = Pipeline.from_toml(
            research_toml_path,
            mock_backend,
            temp_memory,
            temp_tools,
            max_tool_rounds=2,
        )
        result = await pipeline.run("Python programming")

    assert result.steps_completed == 2
    assert len(result.results) == 2
    assert len(result.final_output) > 50
    assert "key findings" in result.final_output.lower() or "python" in result.final_output.lower()


@pytest.mark.asyncio
async def test_pipeline_generates_memory_notes(temp_config, temp_memory, temp_tools, mock_backend):
    """Run full lifecycle (plan -> execute -> reflect -> store); assert memory has reflection notes."""
    from aof.agent.base import BaseAgent
    from aof.agent.lifecycle import AgentLifecycle
    from aof.inference.backend import CompletionResult
    from unittest.mock import AsyncMock

    # Custom backend: plan 2 steps, execute returns "Found X" / "Summary: X", reflect returns "Accomplished X"
    mock_backend._complete_json = AsyncMock(return_value={"steps": ["Search for X", "Summarize X"]})
    complete_responses = [
        "Found X",           # execute step 1
        "Accomplished X",    # reflect step 1
        "Summary: X",        # execute step 2
        "Accomplished X",    # reflect step 2
    ]
    mock_backend._complete = AsyncMock(side_effect=[
        CompletionResult(text=t, tokens_used=5, finish_reason="stop") for t in complete_responses
    ])

    agent = BaseAgent(
        backend=mock_backend,
        memory=temp_memory,
        tools=temp_tools,
        system_prompt="Test agent.",
    )
    lifecycle = AgentLifecycle(agent, max_steps=2)
    await lifecycle.run("Research topic X")

    assert await temp_memory.count() >= 1
    notes = await temp_memory.get_recent(limit=10)
    reflection_notes = [n for n in notes if "reflection" in n.tags and "auto" in n.tags]
    assert len(reflection_notes) >= 1
    assert any("Accomplished X" in n.content for n in reflection_notes)


def test_pipeline_deep_creates_agent(temp_config, temp_memory, temp_tools):
    """create_aof_pipeline_agent with each example TOML; verify agent returns compiled graph.
    Uses mocked model to avoid real API calls. Does not invoke (FakeListChatModel lacks bind_tools).
    """
    from langchain_core.language_models import FakeListChatModel

    from aof.orchestrator.deep_agent_factory import create_aof_pipeline_agent

    with patch(
        "aof.orchestrator.deep_agent_factory.create_chat_model",
        return_value=FakeListChatModel(responses=["Done."]),
    ):
        for toml_path in EXAMPLES_DIR.glob("*.toml"):
            agent = create_aof_pipeline_agent(temp_config, temp_tools, temp_memory, toml_path)
            assert agent is not None
            assert hasattr(agent, "invoke")


@pytest.mark.asyncio
async def test_pipeline_lfm2_nanos_runs_with_mock_backends(temp_config, temp_memory, temp_tools, mock_backend, mock_web_search):
    """Run lfm2_nanos.toml pipeline with mock backends for lfm2_tool, lfm2_extract, lfm2_rag."""
    lfm2_path = EXAMPLES_DIR / "lfm2_nanos.toml"
    if not lfm2_path.exists():
        pytest.skip("lfm2_nanos.toml not found")
    backends = {
        "lfm2_tool": mock_backend,
        "lfm2_extract": mock_backend,
        "lfm2_rag": mock_backend,
    }
    with mock_web_search:
        pipeline = Pipeline.from_toml(
            lfm2_path,
            mock_backend,
            temp_memory,
            temp_tools,
            backends=backends,
            max_tool_rounds=2,
        )
        result = await pipeline.run("test query")

    assert result.steps_completed == 3
    assert len(result.results) == 3
    assert len(result.final_output) > 0


@pytest.mark.asyncio
async def test_pipeline_lfm2_full_runs_with_mock_backends(temp_config, temp_memory, temp_tools, mock_web_search):
    """Run lfm2_full.toml (all 6 LFM2 models); each step uses its role-specific backend."""
    from unittest.mock import AsyncMock

    from aof.inference.backend import CompletionResult, InferenceBackend

    lfm2_full_path = EXAMPLES_DIR / "lfm2_full.toml"
    if not lfm2_full_path.exists():
        pytest.skip("lfm2_full.toml not found")

    # Create distinct mock backends so we can verify each step used its role
    step_outputs = [
        "search: found Python adoption data",
        "extract: key entities extracted",
        "analyze: reasoning complete",
        "quantify: numbers summarized",
        "synthesize: report assembled",
        "finalize: final summary",
    ]

    class RoleTrackingBackend(InferenceBackend):
        def __init__(self, step_label: str):
            self._label = step_label
            self._complete = AsyncMock(
                return_value=CompletionResult(
                    text=step_label,
                    tokens_used=5,
                    finish_reason="stop",
                )
            )
            self._complete_json = AsyncMock(
                return_value={"steps": [step_label]}
            )

        def model_info(self):
            return {"n_ctx": 2048, "max_tokens": 512, "model_path": f"mock-{self._label}"}

        async def complete(self, messages, **kwargs):
            return await self._complete(messages, **kwargs)

        async def complete_json(self, messages, schema=None):
            return await self._complete_json(messages, schema=schema)

        async def start(self):
            pass

        async def shutdown(self):
            pass

    backends = {
        "lfm2_tool": RoleTrackingBackend(step_outputs[0]),
        "lfm2_extract": RoleTrackingBackend(step_outputs[1]),
        "reasoning": RoleTrackingBackend(step_outputs[2]),
        "lfm2_math": RoleTrackingBackend(step_outputs[3]),
        "lfm2_rag": RoleTrackingBackend(step_outputs[4]),
        "fast": RoleTrackingBackend(step_outputs[5]),
    }
    default_backend = RoleTrackingBackend("default")

    with mock_web_search:
        pipeline = Pipeline.from_toml(
            lfm2_full_path,
            default_backend,
            temp_memory,
            temp_tools,
            backends=backends,
            max_tool_rounds=2,
        )
        result = await pipeline.run("quantitative analysis of Python adoption")

    assert result.steps_completed == 6
    assert len(result.results) == 6
    assert len(result.final_output) > 0

    # Verify each step used its role-specific backend (output matches step label)
    for i, expected in enumerate(step_outputs):
        step_text = result.results[i]  # list[str] from PipelineResult
        assert expected in step_text, (
            f"Step {i} expected output containing '{expected}', got '{step_text[:80]}'"
        )


@pytest.mark.asyncio
async def test_complete_research_pipeline_runs(temp_config, temp_memory, temp_tools, mock_backend, mock_web_search):
    """Run complete_research.toml (full composability: orchestrator, memory_context, when, multi-role)."""
    complete_path = EXAMPLES_DIR / "complete_research.toml"
    if not complete_path.exists():
        pytest.skip("complete_research.toml not found")
    backends = {
        "orchestrator": mock_backend,
        "lfm2_tool": mock_backend,
        "lfm2_extract": mock_backend,
        "lfm2_rag": mock_backend,
        "micro": mock_backend,
    }
    with mock_web_search:
        pipeline = Pipeline.from_toml(
            complete_path,
            mock_backend,
            temp_memory,
            temp_tools,
            backends=backends,
            max_tool_rounds=2,
        )
        result = await pipeline.run("AI trends 2025")

    assert result.steps_completed >= 4
    assert len(result.final_output) > 0
    notes = await temp_memory.get_recent(limit=5)
    assert len(notes) >= 1


@pytest.mark.asyncio
async def test_all_examples_load():
    """Iterate examples/*.toml, parse with Pipeline.from_dict, assert no errors."""
    import tomllib

    from aof.config import MemoryConfig
    from aof.tools.builtin import file_tools, math_tools, memory_tools, nlp, scraper, search
    from aof.tools.registry import ToolRegistry

    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        mem_cfg = MemoryConfig(db_path=f"{tmpdir}/mem.db", notes_dir=f"{tmpdir}/notes")
        memory = MemoryStore(mem_cfg)
        await memory.initialize()
        try:
            tools = ToolRegistry()
            search.register(tools)
            scraper.register(tools)
            nlp.register(tools)
            math_tools.register(tools)
            file_tools.register(tools)
            memory_tools.register(tools, memory)

            class MockBackend:
                def model_info(self):
                    return {"n_ctx": 2048, "max_tokens": 512}

            for toml_path in EXAMPLES_DIR.glob("*.toml"):
                with open(toml_path, "rb") as f:
                    data = tomllib.load(f)
                if "steps" not in data:
                    continue  # Skip extended_research.toml etc. (different format)
                pipeline = Pipeline.from_dict(data, MockBackend(), memory, tools)
                assert pipeline is not None
                assert len(pipeline.steps) > 0
        finally:
            await memory.close()
