"""Tests for context builder and pipeline composition."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from aof.inference.backend import CompletionResult
from aof.inference.context import ContextBudget, ContextBuilder


def test_context_budget_from_total():
    budget = ContextBudget.from_total(2048, max_tokens=512)
    assert budget.total == 2048
    assert budget.generation == 512
    assert budget.task == 384
    assert (
        budget.system + budget.memory + budget.history + budget.task + budget.generation
        <= 2048
    )


def test_context_builder_basic():
    budget = ContextBudget.from_total(2048, max_tokens=512)
    builder = ContextBuilder(budget)

    messages = builder.build(
        system_prompt="You are a helpful agent.",
        task="Find information about Python.",
    )

    assert len(messages) >= 2
    assert messages[0]["role"] == "system"
    assert messages[-1]["role"] == "user"
    assert "Python" in messages[-1]["content"]


def test_context_builder_with_history():
    budget = ContextBudget.from_total(2048, max_tokens=512)
    builder = ContextBuilder(budget)

    history = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"},
        {"role": "user", "content": "How are you?"},
        {"role": "assistant", "content": "I'm doing well."},
    ]

    messages = builder.build(
        system_prompt="Agent.",
        task="Continue conversation.",
        history=history,
    )

    # Should include system + some history
    assert messages[0]["role"] == "system"
    assert len(messages) > 1


def test_context_builder_with_memory():
    budget = ContextBudget.from_total(2048, max_tokens=512)
    builder = ContextBuilder(budget)

    messages = builder.build(
        system_prompt="Agent.",
        task="Summarize findings.",
        memory_notes=["Note 1: Python is great.", "Note 2: Scrapy is fast."],
    )

    # Should include system + memory message + ack
    assert len(messages) >= 3
    assert "memory" in messages[1]["content"].lower() or "Note 1" in messages[1]["content"]


def test_context_builder_truncation():
    budget = ContextBudget.from_total(256, max_tokens=64)  # Very small
    builder = ContextBuilder(budget)

    long_prompt = "x " * 5000
    messages = builder.build(system_prompt=long_prompt, task="Do something.")

    # Should not exceed budget (system + final user message)
    total_chars = sum(
        len(m["content"]) if isinstance(m["content"], str) else 0
        for m in messages
    )
    assert total_chars < 256 * 4 + 100  # Allow some overhead


def test_estimate_tokens():
    budget = ContextBudget.from_total(2048)
    builder = ContextBuilder(budget)
    assert builder.estimate_tokens("hello world") == 3  # 11 chars / 3


def test_context_builder_with_image_urls():
    """ContextBuilder adds image_url content to user message when image_urls provided."""
    budget = ContextBudget.from_total(2048, max_tokens=512)
    builder = ContextBuilder(budget)

    messages = builder.build(
        system_prompt="You describe images.",
        task="Describe what you see.",
        image_urls=["file:///path/to/image.png"],
    )

    # Should have system + user message with image_url and text
    assert len(messages) >= 2
    user_msg = next((m for m in messages if m["role"] == "user" and isinstance(m.get("content"), list)), None)
    assert user_msg is not None
    content = user_msg["content"]
    parts = [p for p in content if isinstance(p, dict)]
    image_parts = [p for p in parts if p.get("type") == "image_url"]
    text_parts = [p for p in parts if p.get("type") == "text"]
    assert len(image_parts) == 1
    assert image_parts[0]["image_url"]["url"] == "file:///path/to/image.png"
    assert len(text_parts) == 1
    assert "Describe" in text_parts[0]["text"]


@pytest.mark.asyncio
async def test_pipeline_image_urls_passed_to_context(temp_memory, temp_tools):
    """Pipeline with image_input passes image_urls to agent and backend receives multimodal messages."""
    from aof.pipeline.compose import Pipeline

    captured_messages: list[list] = []

    async def capture_complete(messages, **kwargs):
        captured_messages.append(messages)
        return CompletionResult(
            text="The image shows a cat.",
            tokens_used=10,
            finish_reason="stop",
        )

    async def capture_complete_json(messages, schema=None):
        captured_messages.append(messages)
        return {"steps": ["Describe the image"]}

    backend = MagicMock()
    backend.model_info.return_value = {"n_ctx": 2048, "max_tokens": 512, "model_path": "vision"}
    backend.complete = AsyncMock(side_effect=capture_complete)
    backend.complete_json = AsyncMock(side_effect=capture_complete_json)
    backend.start = AsyncMock()
    backend.shutdown = AsyncMock()

    config = {
        "steps": [
            {
                "name": "describe",
                "system_prompt": "You describe images.",
                "goal_template": "Describe what you see in this image.",
                "tools": [],
                "role": "vision",
                "max_steps": 1,
                "image_input": "{input}",
            }
        ]
    }

    pipeline = Pipeline.from_dict(
        config,
        backend=backend,
        memory=temp_memory,
        tools=temp_tools,
        backends={"vision": backend},
    )

    result = await pipeline.run("file:///path/to/image.png")

    assert result.steps_completed == 1
    assert len(captured_messages) >= 1

    # Find any message with image_url content
    found_image = False
    for msgs in captured_messages:
        for msg in msgs:
            content = msg.get("content", [])
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "image_url":
                        found_image = True
                        assert "file:///path/to/image.png" in str(part.get("image_url", {}))
                        break

    assert found_image, "Backend should have received messages with image_url content"


@pytest.mark.asyncio
async def test_pipeline_step_memory_overrides(temp_memory, temp_tools):
    """Step with memory_strategy and memory_search_limit overrides uses them for retrieval."""
    from aof.pipeline.compose import Pipeline

    await temp_memory.create_note(
        title="Python Key",
        content="Python is used for automation and data science.",
        tags=["python"],
        agent_id="test",
    )

    captured_messages: list[list] = []

    async def capture_complete(messages, **kwargs):
        captured_messages.append(messages)
        return CompletionResult(text="Done.", tokens_used=5, finish_reason="stop")

    async def capture_complete_json(messages, schema=None):
        captured_messages.append(messages)
        return {"steps": ["Do task"]}

    backend = MagicMock()
    backend.model_info.return_value = {"n_ctx": 2048, "max_tokens": 512}
    backend.complete = AsyncMock(side_effect=capture_complete)
    backend.complete_json = AsyncMock(side_effect=capture_complete_json)
    backend.start = AsyncMock()
    backend.shutdown = AsyncMock()

    config = {
        "steps": [
            {
                "name": "with_memory",
                "system_prompt": "You help.",
                "goal_template": "Task: {input}",
                "tools": [],
                "role": "micro",
                "max_steps": 1,
                "memory_strategy": "search",
                "memory_search_limit": 10,
            }
        ]
    }

    pipeline = Pipeline.from_dict(
        config,
        backend=backend,
        memory=temp_memory,
        tools=temp_tools,
    )

    result = await pipeline.run("python automation")

    assert result.steps_completed == 1
    content_str = " ".join(str(m.get("content", "")) for m in captured_messages[0])
    assert "Python" in content_str or "automation" in content_str or "data science" in content_str


@pytest.mark.asyncio
async def test_pipeline_step_memory_tags(temp_memory, temp_tools):
    """Step with memory_tags receives only notes matching those tags."""
    from aof.pipeline.compose import Pipeline

    await temp_memory.create_note(
        title="Research A",
        content="Research findings on topic X.",
        tags=["research"],
        agent_id="test",
    )
    await temp_memory.create_note(
        title="Other B",
        content="Unrelated content.",
        tags=["other"],
        agent_id="test",
    )

    captured_messages: list[list] = []

    async def capture_complete(messages, **kwargs):
        captured_messages.append(messages)
        return CompletionResult(text="Done.", tokens_used=5, finish_reason="stop")

    async def capture_complete_json(messages, schema=None):
        captured_messages.append(messages)
        return {"steps": ["Do task"]}

    backend = MagicMock()
    backend.model_info.return_value = {"n_ctx": 2048, "max_tokens": 512}
    backend.complete = AsyncMock(side_effect=capture_complete)
    backend.complete_json = AsyncMock(side_effect=capture_complete_json)
    backend.start = AsyncMock()
    backend.shutdown = AsyncMock()

    config = {
        "steps": [
            {
                "name": "with_tags",
                "system_prompt": "You help.",
                "goal_template": "Task: {input}",
                "tools": [],
                "role": "micro",
                "max_steps": 1,
                "memory_strategy": "search",
                "memory_tags": ["research"],
            }
        ]
    }

    pipeline = Pipeline.from_dict(
        config,
        backend=backend,
        memory=temp_memory,
        tools=temp_tools,
    )

    result = await pipeline.run("findings")

    assert result.steps_completed == 1
    content_str = " ".join(str(m.get("content", "")) for m in captured_messages[0])
    assert "Research A" in content_str or "topic X" in content_str
    assert "Other B" not in content_str


@pytest.mark.asyncio
async def test_pipeline_memory_context_injected(temp_memory, temp_tools):
    """Step with memory_context=true gets {memory_context} filled with retrieved notes."""
    from aof.pipeline.compose import Pipeline

    await temp_memory.create_note(
        title="RAG Note",
        content="Key insight: retrieval augments generation.",
        tags=["rag"],
        agent_id="test",
    )

    captured_content: list[str] = []

    async def capture_complete(messages, **kwargs):
        for m in messages:
            captured_content.append(str(m.get("content", "")))
        return CompletionResult(text="Synthesized.", tokens_used=5, finish_reason="stop")

    async def capture_complete_json(messages, schema=None):
        return {"steps": ["Synthesize"]}

    backend = MagicMock()
    backend.model_info.return_value = {"n_ctx": 2048, "max_tokens": 512}
    backend.complete = AsyncMock(side_effect=capture_complete)
    backend.complete_json = AsyncMock(side_effect=capture_complete_json)
    backend.start = AsyncMock()
    backend.shutdown = AsyncMock()

    config = {
        "steps": [
            {
                "name": "first",
                "system_prompt": "Help.",
                "goal_template": "Gather: {input}",
                "tools": [],
                "role": "micro",
                "max_steps": 1,
            },
            {
                "name": "synthesize",
                "system_prompt": "Synthesize.",
                "goal_template": "Context:\n{memory_context}\n\nContent:\n{prev_result}",
                "tools": [],
                "role": "micro",
                "max_steps": 1,
                "memory_context": True,
            },
        ]
    }

    pipeline = Pipeline.from_dict(
        config,
        backend=backend,
        memory=temp_memory,
        tools=temp_tools,
    )

    result = await pipeline.run("retrieval")

    assert result.steps_completed == 2
    all_content = " ".join(captured_content)
    assert "RAG Note" in all_content or "retrieval augments" in all_content


@pytest.mark.asyncio
async def test_pipeline_workspace_context_injected(temp_memory, temp_tools):
    """Pipeline.run(workspace_context=...) injects {workspace_context} into goal_template."""
    from aof.pipeline.compose import Pipeline

    captured_content: list[str] = []

    async def capture_complete(messages, **kwargs):
        for m in messages:
            captured_content.append(str(m.get("content", "")))
        return CompletionResult(text="Done.", tokens_used=5, finish_reason="stop")

    async def capture_complete_json(messages, schema=None):
        return {"steps": ["Do it"]}

    backend = MagicMock()
    backend.model_info.return_value = {"n_ctx": 2048, "max_tokens": 512}
    backend.complete = AsyncMock(side_effect=capture_complete)
    backend.complete_json = AsyncMock(side_effect=capture_complete_json)
    backend.start = AsyncMock()
    backend.shutdown = AsyncMock()

    config = {
        "steps": [
            {
                "name": "single",
                "system_prompt": "Help.",
                "goal_template": "Task: {input}. Workspace: {workspace_context}",
                "tools": [],
                "role": "micro",
                "max_steps": 1,
            },
        ]
    }

    pipeline = Pipeline.from_dict(
        config,
        backend=backend,
        memory=temp_memory,
        tools=temp_tools,
    )

    result = await pipeline.run("query", workspace_context="Recent artifacts: a1, a2. Recent memory: Note X.")

    assert result.steps_completed == 1
    all_content = " ".join(captured_content)
    assert "Recent artifacts: a1, a2" in all_content
    assert "Recent memory: Note X" in all_content


@pytest.mark.asyncio
async def test_memory_rag_synthesis_pattern(temp_memory, temp_tools):
    """Pipeline with memory_context + lfm2_rag produces synthesized output."""
    from aof.pipeline.compose import Pipeline

    await temp_memory.create_note(
        title="Research: AI trends",
        content="AI adoption is growing. Key factors: compute, data, talent.",
        tags=["research", "synthesis"],
        agent_id="director",
    )

    all_synth_content: list[str] = []

    async def synth_complete(messages, **kwargs):
        for m in messages:
            all_synth_content.append(str(m.get("content", "")))
        return CompletionResult(
            text="Synthesized report.",
            tokens_used=20,
            finish_reason="stop",
        )

    async def synth_complete_json(messages, schema=None):
        return {"steps": ["Synthesize findings"]}

    backend = MagicMock()
    backend.model_info.return_value = {"n_ctx": 2048, "max_tokens": 512}
    backend.complete = AsyncMock(side_effect=synth_complete)
    backend.complete_json = AsyncMock(side_effect=synth_complete_json)
    backend.start = AsyncMock()
    backend.shutdown = AsyncMock()

    config = {
        "steps": [
            {
                "name": "gather",
                "system_prompt": "Gather.",
                "goal_template": "Gather: {input}",
                "tools": [],
                "role": "micro",
                "max_steps": 1,
            },
            {
                "name": "synthesize",
                "system_prompt": "Synthesize using memory context.",
                "goal_template": "Context:\n{memory_context}\n\nContent:\n{prev_result}",
                "tools": [],
                "role": "lfm2_rag",
                "max_steps": 1,
                "memory_context": True,
                "memory_tags": ["research", "synthesis"],
            },
        ]
    }

    pipeline = Pipeline.from_dict(
        config,
        backend=backend,
        memory=temp_memory,
        tools=temp_tools,
    )

    result = await pipeline.run("AI trends")

    assert result.steps_completed == 2
    assert "Synthesized" in result.final_output
    # Verify synthesize step received memory context (research note content)
    combined = " ".join(all_synth_content)
    assert "AI adoption" in combined or "AI trends" in combined or "research" in combined


@pytest.mark.asyncio
async def test_pipeline_parallel_steps(temp_memory, temp_tools):
    """Pipeline with parallel_count > 1 runs N agents and combines results per choose."""
    from aof.pipeline.compose import Pipeline

    complete_calls = 0

    async def mock_complete(messages, **kwargs):
        nonlocal complete_calls
        complete_calls += 1
        return CompletionResult(
            text=f"Agent {complete_calls} result.",
            tokens_used=10,
            finish_reason="stop",
        )

    async def mock_complete_json(messages, schema=None):
        return {"steps": ["Do the task"]}

    backend = MagicMock()
    backend.model_info.return_value = {"n_ctx": 2048, "max_tokens": 512, "model_path": "mock"}
    backend.complete = AsyncMock(side_effect=mock_complete)
    backend.complete_json = AsyncMock(side_effect=mock_complete_json)
    backend.start = AsyncMock()
    backend.shutdown = AsyncMock()

    config = {
        "steps": [
            {
                "name": "parallel",
                "system_prompt": "You help.",
                "goal_template": "Task: {input}",
                "tools": [],
                "role": "micro",
                "max_steps": 1,
                "parallel_count": 3,
                "choose": "first",
            }
        ]
    }

    pipeline = Pipeline.from_dict(
        config,
        backend=backend,
        memory=temp_memory,
        tools=temp_tools,
    )

    result = await pipeline.run("do something")

    assert result.steps_completed == 1
    # 3 agents × (1 complete_json + 1 complete) = 6 calls; we need at least 3 complete calls
    assert complete_calls >= 3, "Should have run 3 parallel agents (each does plan + execute)"
    assert "Agent" in result.final_output and "result" in result.final_output


@pytest.mark.asyncio
async def test_pipeline_parallel_choose_concat(temp_memory, temp_tools):
    """Pipeline with choose=concat concatenates parallel outputs."""
    from aof.pipeline.compose import Pipeline

    async def mock_complete(messages, **kwargs):
        return CompletionResult(
            text="chunk",
            tokens_used=5,
            finish_reason="stop",
        )

    async def mock_complete_json(messages, schema=None):
        return {"steps": ["Step"]}

    backend = MagicMock()
    backend.model_info.return_value = {"n_ctx": 2048, "max_tokens": 512}
    backend.complete = AsyncMock(side_effect=mock_complete)
    backend.complete_json = AsyncMock(side_effect=mock_complete_json)
    backend.start = AsyncMock()
    backend.shutdown = AsyncMock()

    config = {
        "steps": [
            {
                "name": "parallel",
                "system_prompt": "Help.",
                "goal_template": "{input}",
                "tools": [],
                "role": "micro",
                "max_steps": 1,
                "parallel_count": 2,
                "choose": "concat",
            }
        ]
    }

    pipeline = Pipeline.from_dict(
        config,
        backend=backend,
        memory=temp_memory,
        tools=temp_tools,
    )

    result = await pipeline.run("task")

    assert result.steps_completed == 1
    parts = result.final_output.split("\n\n---\n\n")
    assert len(parts) == 2, "concat should join 2 parallel outputs"
    assert all("chunk" in p for p in parts)


@pytest.mark.asyncio
async def test_pipeline_on_error_abort(temp_memory, temp_tools):
    """Step with on_error=abort stops pipeline and returns partial result on failure."""
    from aof.pipeline.compose import Pipeline

    complete_count = 0

    async def fail_after_first_step(messages, **kwargs):
        nonlocal complete_count
        complete_count += 1
        # Step "ok" is call 1; step "fails" is call 2 (skip_plan: no plan() call)
        if complete_count == 1:
            return CompletionResult(text="First done.", tokens_used=5, finish_reason="stop")
        raise RuntimeError("Backend failed")

    async def json_fail_after_first(messages, schema=None):
        return {"steps": ["Do it"]}

    backend = MagicMock()
    backend.model_info.return_value = {"n_ctx": 2048, "max_tokens": 512}
    backend.complete = AsyncMock(side_effect=fail_after_first_step)
    backend.complete_json = AsyncMock(side_effect=json_fail_after_first)
    backend.start = AsyncMock()
    backend.shutdown = AsyncMock()

    config = {
        "steps": [
            {
                "name": "ok",
                "system_prompt": "Help.",
                "goal_template": "{input}",
                "tools": [],
                "role": "micro",
                "max_steps": 1,
            },
            {
                "name": "fails",
                "system_prompt": "Help.",
                "goal_template": "{prev_result}",
                "tools": [],
                "role": "micro",
                "max_steps": 1,
                "on_error": "abort",
            },
        ]
    }

    pipeline = Pipeline.from_dict(
        config,
        backend=backend,
        memory=temp_memory,
        tools=temp_tools,
    )

    result = await pipeline.run("start")

    assert result.steps_completed == 1
    assert result.final_output == "First done."
    assert "Backend failed" in result.metadata.get("error", "")
    assert result.metadata.get("failed_step") == "fails"


@pytest.mark.asyncio
async def test_pipeline_on_error_propagate(temp_memory, temp_tools):
    """Failed step with on_error=propagate passes error message to next step."""
    from aof.pipeline.compose import Pipeline

    # All no-tool steps use skip_plan=True (no plan/complete_json call).
    # Step 1 succeeds, Step 2 fails in execute (complete), Step 3 skipped (failure propagated).
    complete_call_count = 0

    async def complete_with_fail(messages, **kwargs):
        nonlocal complete_call_count
        complete_call_count += 1
        # Calls: 1=first execute, 2=first reflect, 3=fails execute → raise
        if complete_call_count == 3:
            raise RuntimeError("Step 2 failed")
        return CompletionResult(
            text="Step 1.",
            tokens_used=5,
            finish_reason="stop",
        )

    backend = MagicMock()
    backend.model_info.return_value = {"n_ctx": 2048, "max_tokens": 512}
    backend.complete = AsyncMock(side_effect=complete_with_fail)
    backend.complete_json = AsyncMock(return_value={"steps": ["Do it"]})
    backend.start = AsyncMock()
    backend.shutdown = AsyncMock()

    config = {
        "steps": [
            {
                "name": "first",
                "system_prompt": "Help.",
                "goal_template": "{input}",
                "tools": [],
                "role": "micro",
                "max_steps": 1,
            },
            {
                "name": "fails",
                "system_prompt": "Help.",
                "goal_template": "{prev_result}",
                "tools": [],
                "role": "micro",
                "max_steps": 2,
                "on_error": "propagate",
            },
            {
                "name": "third",
                "system_prompt": "Echo the input.",
                "goal_template": "Echo: {prev_result}",
                "tools": [],
                "role": "micro",
                "max_steps": 1,
            },
        ]
    }

    pipeline = Pipeline.from_dict(
        config,
        backend=backend,
        memory=temp_memory,
        tools=temp_tools,
    )

    result = await pipeline.run("start")

    assert result.steps_completed == 3
    # Propagated output must include step name and error summary
    propagated = result.results[1]
    assert "failed" in propagated.lower()
    assert "fails" in propagated  # step name
    assert "Step 2" in propagated  # step index


@pytest.mark.asyncio
async def test_pipeline_step_when_condition(temp_memory, temp_tools):
    """Step with when=non_empty is skipped when prev_result is empty."""
    from aof.pipeline.compose import Pipeline

    complete_call_count = 0

    async def capture_step(messages, **kwargs):
        nonlocal complete_call_count
        complete_call_count += 1
        return CompletionResult(text="", tokens_used=0, finish_reason="stop")

    async def json_step(messages, schema=None):
        return {"steps": ["Do it"]}

    backend = MagicMock()
    backend.model_info.return_value = {"n_ctx": 2048, "max_tokens": 512}
    backend.complete = AsyncMock(side_effect=capture_step)
    backend.complete_json = AsyncMock(side_effect=json_step)
    backend.start = AsyncMock()
    backend.shutdown = AsyncMock()

    config = {
        "steps": [
            {"name": "first", "system_prompt": "Help.", "goal_template": "{input}", "tools": [], "role": "micro", "max_steps": 1, "when": "always"},
            {"name": "second", "system_prompt": "Echo.", "goal_template": "Echo: {prev_result}", "tools": [], "role": "micro", "max_steps": 1, "when": "non_empty"},
        ]
    }

    pipeline = Pipeline.from_dict(config, backend=backend, memory=temp_memory, tools=temp_tools)
    result = await pipeline.run("start")

    assert result.steps_completed == 2
    assert result.results[0] == ""
    assert result.results[1] == ""
    assert complete_call_count == 2
