"""Composable agent pipelines: chain agents sequentially or run in parallel."""

from __future__ import annotations

import asyncio
import logging
import time
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

# Progress callback: (step_index, step_name, event, detail, step_output=None) -> None
# event: "step_start" | "step_done" | "step_skipped"; step_output set only for step_done
OnProgress = Callable[[int, str, str, str, str | None], None]

from aof.agent.base import AgentState, BaseAgent
from aof.agent.lifecycle import AgentLifecycle
from aof.inference.backend import InferenceBackend
from aof.memory.store import MemoryStore
from aof.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


def _is_failure_message(text: str) -> bool:
    """True if text looks like a propagated step failure."""
    return "[Step " in text and " failed:" in text


def _is_overflow_message(text: str) -> bool:
    """True if text is the context-overflow error from the Llama backend."""
    return bool(text) and "[Error: context overflow —" in text and "token window]" in text


def _truncate_prev_result(prev_result: str, max_chars: int = 1500) -> str:
    """Cap previous step output so downstream goal fits context budget."""
    if len(prev_result) <= max_chars:
        return prev_result
    return prev_result[: max_chars - 3] + "..."


def _apply_postprocess(output: str, mode: str) -> str:
    """Apply a built-in postprocess mode to step output."""
    if not mode or not output:
        return output
    if mode == "strip_overflow":
        lines = [ln for ln in output.splitlines() if "[Error: context overflow —" not in ln or "token window]" not in ln]
        return "\n".join(lines).strip()
    if mode.startswith("extract_section:"):
        name = mode.split(":", 1)[1].strip()
        import re
        pattern = rf"(?:^|\n)#+\s*{re.escape(name)}\s*\n(.*?)(?=\n#+\s|\Z)"
        m = re.search(pattern, output, re.DOTALL | re.IGNORECASE)
        return m.group(1).strip() if m else output
    return output


def _check_repeat_condition(condition: str, output: str) -> bool:
    """Check if step output meets a repeat_until condition.

    Conditions:
      - "non_empty": output is not empty/whitespace
      - "min_length:N": output has at least N characters
    """
    if not condition:
        return True
    if condition == "non_empty":
        return bool(output.strip())
    if condition.startswith("min_length:"):
        try:
            min_len = int(condition.split(":")[1])
            return len(output.strip()) >= min_len
        except (ValueError, IndexError):
            return True
    return True


@dataclass
class PipelineStep:
    """Definition for a single step in a pipeline.

    Step name is used in progress files (artifacts/<item_id>_progress.md) and
    in artifact section headers; use short, action-oriented names (e.g. search, analyze, report).
    """

    name: str
    system_prompt: str = ""
    goal_template: str = "{input}"  # Placeholders: {input}, {prev_result}, {memory_context}, {workspace_context}
    tools: list[str] = field(default_factory=list)
    role: str = "general"  # For multi-model routing
    max_steps: int = 5
    image_input: str = ""  # Template for image path(s), e.g. "file://path" or "{prev_result}"
    step_retries: int = 0  # Retry failed steps up to N times
    parallel_count: int = 1  # Run N agents in parallel when > 1
    choose: str = "first"  # "first" | "best" | "concat" — how to combine parallel results
    memory_strategy: str | None = None  # Override: search | recent | agent_notes | hybrid
    memory_search_limit: int | None = None  # Override for memory retrieval limit
    memory_context: bool = False  # Pre-fetch memory and inject as {memory_context} in goal
    on_error: str = "propagate"  # propagate | retry | abort — how to handle step failures
    memory_tags: list[str] | None = None  # Filter memory retrieval to notes with any of these tags
    when: str = "non_empty"  # non_empty | always — skip if prev_result empty or failure when non_empty
    prev_result_max_chars: int | None = None  # Max chars of prev_result in goal (default 1500)
    max_repeats: int = 1  # Max times to run this step (re-run if repeat_until not met)
    repeat_until: str = ""  # Condition: "non_empty", "min_length:N", "" (no repeat)
    context_mode: str = "normal"  # normal | long_input (use for report/synthesis steps)
    overflow_fallback_role: str = ""  # If context overflow, retry once with this role (e.g. report_large)
    postprocess: str = ""  # strip_overflow | extract_section:Name | "" (clean step output)


@dataclass
class PipelineResult:
    """Result of a pipeline execution."""

    steps_completed: int
    results: list[str]
    final_output: str
    step_names: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class Pipeline:
    """Chain agents together: output of step N becomes input to step N+1.

    Pipelines can be defined in code or loaded from TOML config:

        [[steps]]
        name = "research"
        system_prompt = "You are a research agent."
        goal_template = "Research: {input}"
        tools = ["web_search", "web_scrape"]
        role = "micro"

        [[steps]]
        name = "synthesize"
        system_prompt = "You synthesize research into reports."
        goal_template = "Synthesize this: {prev_result}"
        tools = []
        role = "small"
    """

    def __init__(
        self,
        steps: list[PipelineStep],
        backend: InferenceBackend,
        memory: MemoryStore,
        tools: ToolRegistry,
        *,
        backends: dict[str, InferenceBackend] | None = None,
        max_tool_rounds: int = 5,
    ) -> None:
        self.steps = steps
        self.backend = backend
        self.memory = memory
        self.tools = tools
        self._backends = backends or {}
        self._max_tool_rounds = max_tool_rounds

    async def run(
        self,
        initial_input: str,
        *,
        on_progress: OnProgress | None = None,
        workspace_context: str = "",
    ) -> PipelineResult:
        """Execute the pipeline sequentially.

        If on_progress is provided, it is called with (step_index, step_name, event, detail)
        for step_start, step_done, and step_skipped events.
        workspace_context is injected into goal_template as {workspace_context} when present.
        """
        results: list[str] = []
        step_names: list[str] = []
        step_latencies_ms: list[float] = []
        step_tool_calls: list[int] = []
        signals: dict[str, str] = {}
        current_input = initial_input

        for i, step in enumerate(self.steps):
            logger.info("Pipeline step %d/%d: %s", i + 1, len(self.steps), step.name)

            prev_result = results[-1] if results else ""
            if step.when == "non_empty" and (
                (i > 0 and not prev_result.strip())
                or _is_failure_message(prev_result)
            ):
                logger.info("Pipeline step %s skipped (when=non_empty, prev_result empty or failure)", step.name)
                if on_progress:
                    on_progress(i, step.name, "step_skipped", "prev_result empty or failure", None)
                results.append(prev_result)
                step_names.append(step.name)
                current_input = prev_result
                step_tool_calls.append(0)
                step_latencies_ms.append(0.0)
                continue

            if on_progress:
                on_progress(i, step.name, "step_start", "started", None)

            # Resolve memory_context placeholder if step requests it
            memory_context_text = ""
            if step.memory_context:
                search_query = current_input or prev_result or step.name
                limit = step.memory_search_limit or self.memory.config.memory_search_limit
                notes = await self.memory.search(
                    search_query,
                    limit=limit,
                    tags=step.memory_tags,
                )
                memory_context_text = (
                    "\n".join(f"[{n.title}]\n{n.content}" for n in notes)
                    if notes
                    else "(no relevant memory found)"
                )
            # Format goal with template (cap prev_result to avoid context overflow)
            prev_result_max = step.prev_result_max_chars if step.prev_result_max_chars is not None else 1500
            prev_result_bounded = _truncate_prev_result(prev_result, prev_result_max)
            signals_str = "; ".join(f"{k}={v}" for k, v in signals.items()) if signals else ""
            goal = step.goal_template.format(
                input=current_input,
                prev_result=prev_result_bounded,
                memory_context=memory_context_text,
                workspace_context=workspace_context,
                signals=signals_str,
            )

            # Resolve image_input for VL steps
            image_urls: list[str] = []
            if step.image_input:
                resolved = step.image_input.format(
                    input=current_input,
                    prev_result=prev_result,
                )
                if resolved:
                    image_urls = [p.strip() for p in resolved.split(",") if p.strip()]

            # Select backend for this step's role
            backend = self._backends.get(step.role, self.backend)

            # Create filtered tool registry for this step
            step_tools = self.tools.filtered(step.tools) if step.tools else self.tools

            step_output = ""
            elapsed_ms = 0.0
            total_tool_calls = 0
            step_error: Exception | None = None

            for attempt in range(
                (step.step_retries + 1) if step.on_error == "retry" else 1
            ):
                try:
                    if step.parallel_count > 1:
                        step_output, elapsed_ms, total_tool_calls = (
                            await self._run_parallel_step(
                                step, goal, image_urls, backend, step_tools
                            )
                        )
                    else:
                        agent = BaseAgent(
                            backend=backend,
                            memory=self.memory,
                            tools=step_tools,
                            system_prompt=step.system_prompt,
                            role=step.role,
                            max_tool_rounds=self._max_tool_rounds,
                        )
                        if image_urls:
                            agent.ctx.metadata["image_urls"] = image_urls
                        if step.memory_strategy is not None:
                            agent.ctx.metadata["memory_strategy"] = step.memory_strategy
                        if step.memory_search_limit is not None:
                            agent.ctx.metadata["memory_search_limit"] = (
                                step.memory_search_limit
                            )
                        if step.memory_tags is not None:
                            agent.ctx.metadata["memory_tags"] = step.memory_tags

                        lifecycle = AgentLifecycle(
                            agent,
                            max_steps=step.max_steps,
                            step_retries=step.step_retries,
                            skip_plan=(not step.tools),
                        )
                        t0 = time.monotonic()
                        ctx = await lifecycle.run(goal)
                        elapsed_ms = (time.monotonic() - t0) * 1000

                        if ctx.state == AgentState.ERROR:
                            err_msg = ctx.metrics.errors[-1] if ctx.metrics.errors else "Unknown error"
                            raise RuntimeError(err_msg)

                        if ctx.results:
                            last = ctx.results[-1]
                            step_output = (
                                last.text if hasattr(last, "text") else str(last)
                            )
                        total_tool_calls = ctx.metrics.tool_calls_made
                    step_error = None
                    break
                except Exception as e:
                    step_error = e
                    if step.on_error == "abort":
                        logger.error("Pipeline step '%s' failed (abort): %s", step.name, e)
                        return PipelineResult(
                            steps_completed=len(results),
                            results=results,
                            final_output=results[-1] if results else "",
                            step_names=step_names,
                            metadata={
                                "step_latencies_ms": step_latencies_ms,
                                "step_tool_calls": step_tool_calls,
                                "error": str(e),
                                "failed_step": step.name,
                            },
                        )
                    if step.on_error == "retry" and attempt < step.step_retries:
                        logger.warning(
                            "Pipeline step '%s' failed (attempt %d), retrying: %s",
                            step.name,
                            attempt + 1,
                            e,
                        )
                        continue
                    # propagate (default) or retry exhausted: pass error message to next step
                    err_str = str(e)
                    if len(err_str) > 200:
                        err_str = err_str[:200] + "..."
                    step_output = f"[Step {i + 1} ({step.name}) failed: {err_str}]"
                    logger.warning("Pipeline step '%s' failed (propagate): %s", step.name, e)
                    break

            # Overflow-aware retry: re-run once with larger-context role if available
            if _is_overflow_message(step_output) and step.overflow_fallback_role:
                fallback_backend = self._backends.get(step.overflow_fallback_role)
                if fallback_backend:
                    logger.warning(
                        "Step '%s' hit context overflow; retrying with role '%s'",
                        step.name,
                        step.overflow_fallback_role,
                    )
                    try:
                        if step.parallel_count > 1:
                            step_output, _el, total_tool_calls = await self._run_parallel_step(
                                step, goal, image_urls, fallback_backend, step_tools
                            )
                        else:
                            agent = BaseAgent(
                                backend=fallback_backend,
                                memory=self.memory,
                                tools=step_tools,
                                system_prompt=step.system_prompt,
                                role=step.overflow_fallback_role,
                                max_tool_rounds=self._max_tool_rounds,
                            )
                            if image_urls:
                                agent.ctx.metadata["image_urls"] = image_urls
                            if step.memory_strategy is not None:
                                agent.ctx.metadata["memory_strategy"] = step.memory_strategy
                            if step.memory_search_limit is not None:
                                agent.ctx.metadata["memory_search_limit"] = step.memory_search_limit
                            if step.memory_tags is not None:
                                agent.ctx.metadata["memory_tags"] = step.memory_tags
                            lifecycle = AgentLifecycle(
                                agent,
                                max_steps=step.max_steps,
                                step_retries=step.step_retries,
                                skip_plan=(not step.tools),
                            )
                            t0 = time.monotonic()
                            ctx = await lifecycle.run(goal)
                            elapsed_ms = (time.monotonic() - t0) * 1000
                            if ctx.state == AgentState.ERROR and ctx.metrics.errors:
                                raise RuntimeError(ctx.metrics.errors[-1])
                            if ctx.results:
                                last = ctx.results[-1]
                                step_output = last.text if hasattr(last, "text") else str(last)
                            total_tool_calls = ctx.metrics.tool_calls_made
                    except Exception as retry_err:
                        logger.warning(
                            "Overflow retry for step '%s' failed: %s",
                            step.name,
                            retry_err,
                        )
                        # step_output stays the overflow message

            # Step repetition: re-run if repeat_until condition not met
            if step.repeat_until and step.max_repeats > 1 and not _is_failure_message(step_output):
                for repeat_num in range(1, step.max_repeats):
                    if _check_repeat_condition(step.repeat_until, step_output):
                        break
                    logger.info(
                        "Pipeline step '%s' repeat %d/%d (condition '%s' not met)",
                        step.name, repeat_num + 1, step.max_repeats, step.repeat_until,
                    )
                    # Re-run with previous attempt appended to goal
                    repeat_goal = f"{goal}\n\nPrevious attempt (did not meet quality bar):\n{_truncate_prev_result(step_output, 1000)}"
                    try:
                        if step.parallel_count > 1:
                            step_output, rms, rtc = await self._run_parallel_step(
                                step, repeat_goal, image_urls, backend, step_tools,
                            )
                        else:
                            agent = BaseAgent(
                                backend=backend, memory=self.memory, tools=step_tools,
                                system_prompt=step.system_prompt, role=step.role,
                                max_tool_rounds=self._max_tool_rounds,
                            )
                            lifecycle = AgentLifecycle(
                                agent, max_steps=step.max_steps,
                                skip_plan=(not step.tools),
                            )
                            t0 = time.monotonic()
                            ctx = await lifecycle.run(repeat_goal)
                            rms = (time.monotonic() - t0) * 1000
                            rtc = ctx.metrics.tool_calls_made
                            if ctx.results:
                                last = ctx.results[-1]
                                step_output = last.text if hasattr(last, "text") else str(last)
                        elapsed_ms += rms
                        total_tool_calls += rtc
                    except Exception as e:
                        logger.warning("Step '%s' repeat %d failed: %s", step.name, repeat_num + 1, e)
                        break

            step_output = _apply_postprocess(step_output, step.postprocess)
            step_tool_calls.append(total_tool_calls)
            results.append(step_output)
            step_names.append(step.name)
            current_input = step_output
            step_latencies_ms.append(elapsed_ms)

            # Collect signals from step output: [SIGNAL:key=value]
            import re
            for match in re.finditer(r'\[SIGNAL:(\w+)=([^\]]+)\]', step_output):
                signals[match.group(1)] = match.group(2)

            if on_progress:
                if _is_failure_message(step_output):
                    detail = "failed: " + (step_output[:100] + "..." if len(step_output) > 100 else step_output)
                else:
                    detail = f"{len(step_output)} chars, {total_tool_calls} tool calls"
                on_progress(i, step.name, "step_done", detail, step_output)

            logger.info(
                "Pipeline step '%s' completed. Output length: %d",
                step.name,
                len(step_output),
            )

        return PipelineResult(
            steps_completed=len(results),
            results=results,
            final_output=results[-1] if results else "",
            step_names=step_names,
            metadata={
                "step_latencies_ms": step_latencies_ms,
                "step_tool_calls": step_tool_calls,
                "signals": signals,
            },
        )

    async def _run_parallel_step(
        self,
        step: PipelineStep,
        goal: str,
        image_urls: list[str],
        backend: InferenceBackend,
        step_tools: ToolRegistry,
    ) -> tuple[str, float, int]:
        """Run parallel_count agents and combine results per step.choose."""
        agents_and_lifecycles: list[tuple[BaseAgent, AgentLifecycle]] = []
        for _ in range(step.parallel_count):
            agent = BaseAgent(
                backend=backend,
                memory=self.memory,
                tools=step_tools,
                system_prompt=step.system_prompt,
                role=step.role,
                max_tool_rounds=self._max_tool_rounds,
            )
            if image_urls:
                agent.ctx.metadata["image_urls"] = image_urls
            if step.memory_strategy is not None:
                agent.ctx.metadata["memory_strategy"] = step.memory_strategy
            if step.memory_search_limit is not None:
                agent.ctx.metadata["memory_search_limit"] = step.memory_search_limit
            if step.memory_tags is not None:
                agent.ctx.metadata["memory_tags"] = step.memory_tags
            lifecycle = AgentLifecycle(
                agent,
                max_steps=step.max_steps,
                step_retries=step.step_retries,
                skip_plan=(not step.tools),
            )
            agents_and_lifecycles.append((agent, lifecycle))

        t0 = time.monotonic()
        contexts = await asyncio.gather(
            *[lc.run(goal) for _, lc in agents_and_lifecycles]
        )
        elapsed_ms = (time.monotonic() - t0) * 1000

        # Check for failures (lifecycle catches, doesn't raise)
        for ctx in contexts:
            if ctx.state == AgentState.ERROR and ctx.metrics.errors:
                raise RuntimeError(ctx.metrics.errors[-1])

        # Extract outputs
        outputs: list[str] = []
        for ctx in contexts:
            out = ""
            if ctx.results:
                last = ctx.results[-1]
                out = last.text if hasattr(last, "text") else str(last)
            outputs.append(out)

        total_tool_calls = sum(c.metrics.tool_calls_made for c in contexts)

        # Apply choose
        if step.choose == "concat":
            step_output = "\n\n---\n\n".join(outputs)
        elif step.choose == "best":
            best_idx = max(
                range(len(contexts)),
                key=lambda i: contexts[i].metrics.health_score,
            )
            step_output = outputs[best_idx]
        else:
            step_output = outputs[0] if outputs else ""

        return step_output, elapsed_ms, total_tool_calls

    @classmethod
    def from_toml(
        cls,
        path: Path | str,
        backend: InferenceBackend,
        memory: MemoryStore,
        tools: ToolRegistry,
        **kwargs: Any,
    ) -> Pipeline:
        """Load a pipeline definition from a TOML file."""
        with open(path, "rb") as f:
            data = tomllib.load(f)

        steps = []
        for step_data in data.get("steps", []):
            steps.append(
                PipelineStep(
                    name=step_data.get("name", f"step_{len(steps)}"),
                    system_prompt=step_data.get("system_prompt", ""),
                    goal_template=step_data.get("goal_template", "{input}"),
                    tools=step_data.get("tools", []),
                    role=step_data.get("role", "general"),
                    max_steps=step_data.get("max_steps", 5),
                    image_input=step_data.get("image_input", ""),
                    step_retries=step_data.get("step_retries", 0),
                    parallel_count=step_data.get("parallel_count", 1),
                    choose=step_data.get("choose", "first"),
                    memory_strategy=step_data.get("memory_strategy"),
                    memory_search_limit=step_data.get("memory_search_limit"),
                    memory_context=step_data.get("memory_context", False),
                    on_error=step_data.get("on_error", "propagate"),
                    memory_tags=step_data.get("memory_tags"),
                    when=step_data.get("when", "non_empty"),
                    prev_result_max_chars=step_data.get("prev_result_max_chars"),
                    max_repeats=step_data.get("max_repeats", 1),
                    repeat_until=step_data.get("repeat_until", ""),
                    context_mode=step_data.get("context_mode", "normal"),
                    overflow_fallback_role=step_data.get("overflow_fallback_role", ""),
                    postprocess=step_data.get("postprocess", ""),
                )
            )

        return cls(steps, backend, memory, tools, **kwargs)

    @classmethod
    def from_dict(
        cls,
        config: dict,
        backend: InferenceBackend,
        memory: MemoryStore,
        tools: ToolRegistry,
        **kwargs: Any,
    ) -> Pipeline:
        """Create a pipeline from a dictionary config."""
        step_fields = {
            "name", "system_prompt", "goal_template", "tools", "role",
            "max_steps", "image_input", "step_retries", "parallel_count", "choose",
            "memory_strategy", "memory_search_limit", "memory_context", "on_error",
            "memory_tags", "when", "prev_result_max_chars", "max_repeats", "repeat_until",
            "context_mode", "overflow_fallback_role", "postprocess",
        }
        steps = []
        for step_data in config.get("steps", []):
            filtered = {k: step_data.get(k) for k in step_fields if k in step_data}
            steps.append(
                PipelineStep(
                    name=filtered.get("name", f"step_{len(steps)}"),
                    system_prompt=filtered.get("system_prompt", ""),
                    goal_template=filtered.get("goal_template", "{input}"),
                    tools=filtered.get("tools", []),
                    role=filtered.get("role", "general"),
                    max_steps=filtered.get("max_steps", 5),
                    image_input=filtered.get("image_input", ""),
                    step_retries=filtered.get("step_retries", 0),
                    parallel_count=filtered.get("parallel_count", 1),
                    choose=filtered.get("choose", "first"),
                    memory_strategy=filtered.get("memory_strategy"),
                    memory_search_limit=filtered.get("memory_search_limit"),
                    memory_context=filtered.get("memory_context", False),
                    on_error=filtered.get("on_error", "propagate"),
                    memory_tags=filtered.get("memory_tags"),
                    when=filtered.get("when", "non_empty"),
                    prev_result_max_chars=filtered.get("prev_result_max_chars"),
                    max_repeats=filtered.get("max_repeats", 1),
                    repeat_until=filtered.get("repeat_until", ""),
                    context_mode=filtered.get("context_mode", "normal"),
                    overflow_fallback_role=filtered.get("overflow_fallback_role", ""),
                    postprocess=filtered.get("postprocess", ""),
                )
            )
        return cls(steps, backend, memory, tools, **kwargs)
