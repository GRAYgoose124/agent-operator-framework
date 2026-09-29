"""Agent protocol and base implementation."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Protocol

from aof.inference.backend import CompletionResult, InferenceBackend
from aof.inference.context import ContextBuilder, ContextBudget
from aof.inference.parsing import ParsedResponse, detect_model_family, parse_response
from aof.memory.store import MemoryStore
from aof.tools.registry import ToolRegistry


class AgentState(Enum):
    IDLE = auto()
    PLANNING = auto()
    EXECUTING = auto()
    REFLECTING = auto()
    STORING = auto()
    DONE = auto()
    ERROR = auto()


@dataclass
class AgentMetrics:
    """Real-time performance metrics for an agent."""

    steps_attempted: int = 0
    steps_completed: int = 0
    tool_calls_made: int = 0
    tool_calls_succeeded: int = 0
    parse_failures: int = 0
    total_tokens: int = 0
    latencies_ms: list[float] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def completion_rate(self) -> float:
        if self.steps_attempted == 0:
            return 1.0
        return self.steps_completed / self.steps_attempted

    @property
    def tool_accuracy(self) -> float:
        if self.tool_calls_made == 0:
            return 1.0
        return self.tool_calls_succeeded / self.tool_calls_made

    @property
    def avg_latency_ms(self) -> float:
        if not self.latencies_ms:
            return 0.0
        return sum(self.latencies_ms) / len(self.latencies_ms)

    @property
    def health_score(self) -> float:
        """Composite health score 0.0-1.0. Weighted by importance."""
        completion_w = 0.4
        tool_w = 0.3
        parse_w = 0.3

        parse_score = 1.0
        total_outputs = self.steps_attempted
        if total_outputs > 0:
            parse_score = max(0.0, 1.0 - (self.parse_failures / total_outputs))

        return (
            self.completion_rate * completion_w
            + self.tool_accuracy * tool_w
            + parse_score * parse_w
        )


@dataclass
class AgentContext:
    """Mutable state carried through an agent's lifecycle."""

    agent_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    role: str = "general"  # Role for multi-model routing
    goal: str = ""
    plan: list[str] = field(default_factory=list)
    current_step: int = 0
    history: list[dict[str, str]] = field(default_factory=list)
    results: list[Any] = field(default_factory=list)
    state: AgentState = AgentState.IDLE
    metrics: AgentMetrics = field(default_factory=AgentMetrics)
    metadata: dict[str, Any] = field(default_factory=dict)


class Agent(Protocol):
    """Protocol that all agents must satisfy."""

    @property
    def ctx(self) -> AgentContext: ...

    async def plan(self) -> list[str]: ...
    async def execute_step(self, step: str) -> ParsedResponse: ...
    async def reflect(self, result: ParsedResponse) -> str: ...
    async def store(self, reflection: str) -> None: ...


class BaseAgent:
    """Default agent implementation using an LLM backend for all phases."""

    def __init__(
        self,
        backend: InferenceBackend,
        memory: MemoryStore,
        tools: ToolRegistry,
        *,
        system_prompt: str = "",
        role: str = "general",
        agent_id: str | None = None,
        max_tool_rounds: int = 5,
    ) -> None:
        self.backend = backend
        self.memory = memory
        self.tools = tools
        self._max_tool_rounds = max_tool_rounds
        self.system_prompt = system_prompt or self._default_system_prompt()
        self.ctx = AgentContext(
            agent_id=agent_id or uuid.uuid4().hex[:8],
            role=role,
        )

        # Build context manager from model info
        info = backend.model_info()
        n_ctx = info.get("n_ctx", 2048)
        max_tokens = info.get("max_tokens", 512)

        # Detect model family for tool format selection
        model_path = info.get("model_path", info.get("model", ""))
        self._model_family = detect_model_family(model_path)

        # Thinking models need extra generation headroom for <think> blocks to avoid context overflow
        is_thinking = "thinking" in model_path.lower() or "think" in model_path.lower()
        if is_thinking:
            max_tokens = max_tokens + 512
        # Scale task reserve with context size: larger contexts get more room for task content
        if is_thinking:
            task_reserve = min(512, max(256, n_ctx // 16))
        else:
            task_reserve = min(768, max(384, n_ctx // 8))
        budget = ContextBudget.from_total(n_ctx, max_tokens, task_reserve=task_reserve)
        self._context_builder = ContextBuilder(budget, model_family=self._model_family)

    async def plan(self) -> list[str]:
        """Decompose goal into steps using the LLM."""
        messages = await self._build_messages(
            "Break this goal into 2-5 concrete steps. "
            "Output a JSON object with a 'steps' array of short action strings "
            "(one phrase per step, e.g. 'Search for X', 'Summarize findings'). "
            "No conversational text.\n\nGoal: " + self.ctx.goal
        )
        result = await self.backend.complete_json(messages)
        if isinstance(result, list):
            raw_steps = result
        else:
            raw_steps = result.get("steps", [result.get("raw", "Execute the goal directly")])
        if isinstance(raw_steps, str):
            raw_steps = [raw_steps]
        steps = [_normalize_plan_step(s) for s in raw_steps]
        if not steps:
            steps = ["Proceed with the task"]
        self.ctx.plan = steps
        return steps

    async def execute_step(self, step: str) -> ParsedResponse:
        """Execute a single plan step, handling tool calls with multi-round loop."""
        # Build goal context: skip when step IS the goal (skip_plan mode avoids duplication),
        # otherwise include a truncated excerpt of the goal to give the plan step content to work with.
        goal_part = ""
        if self.ctx.goal and step != self.ctx.goal:
            # Truncate goal to fit ~half the task token budget (leave room for step + instructions)
            task_chars = self._context_builder._budget.task * self._context_builder._chars_per_token
            max_context_chars = max(200, task_chars - len(step) - 150)
            goal_text = self.ctx.goal[:max_context_chars]
            if len(self.ctx.goal) > max_context_chars:
                goal_text += "..."
            goal_part = f"\n\nContext:\n{goal_text}\n\n"

        has_tools = bool(self.tools.list_names())
        if has_tools:
            task = (
                f"Execute this step: {step}{goal_part}"
                f"If you need to use a tool, respond with: <tool_call>{{\"name\": \"...\", \"arguments\": {{...}}}}</tool_call> "
                f"Otherwise, respond with your result directly."
            )
        else:
            task = (
                f"Execute this step: {step}{goal_part}"
                f"Respond with your result directly. No tools are available for this step."
            )
        self.ctx.history.append({"role": "user", "content": f"Step: {step}"})

        for _ in range(self._max_tool_rounds):
            guidance = self.ctx.metadata.pop("pending_guidance", None)
            messages = await self._build_messages(task, guidance=guidance)

            result = await self.backend.complete(messages)
            parsed = parse_response(result.text)
            self.ctx.metrics.total_tokens += result.tokens_used

            if not parsed.tool_calls:
                self.ctx.history.append({"role": "assistant", "content": parsed.text})
                return parsed

            # Execute tool calls and build results message
            tool_results: list[str] = []
            for tc in parsed.tool_calls:
                self.ctx.metrics.tool_calls_made += 1
                tool_result = await self.tools.invoke(tc.name, tc.arguments)

                if isinstance(tool_result, dict) and "error" in tool_result:
                    self.ctx.metrics.errors.append(f"Tool {tc.name}: {tool_result['error']}")
                else:
                    self.ctx.metrics.tool_calls_succeeded += 1

                tool_results.append(
                    f"[Tool {tc.name} returned: {_truncate(str(tool_result), 1500)}]"
                )

            # Append assistant (with tool calls) and user (tool results) to history
            self.ctx.history.append({"role": "assistant", "content": result.text})
            self.ctx.history.append(
                {"role": "user", "content": "\n\n".join(tool_results)}
            )

        # Max rounds reached with tool calls; return last response
        self.ctx.history.append({"role": "assistant", "content": parsed.text})
        return parsed

    async def reflect(self, result: ParsedResponse) -> str:
        """Evaluate the result and produce a summary reflection."""
        messages = await self._build_messages(
            f"Briefly evaluate this result (2-3 sentences). "
            f"Note what was accomplished and any issues.\n\n"
            f"Result:\n{_truncate(result.text, 800)}"
        )
        response = await self.backend.complete(messages, max_tokens=256)
        self.ctx.metrics.total_tokens += response.tokens_used
        return parse_response(response.text).text

    async def store(self, reflection: str) -> None:
        """Persist the reflection to the Zettelkasten memory."""
        if _is_meta_reflection(reflection):
            reflection = "Step completed; see artifact for details."
        step_info = ""
        if self.ctx.current_step < len(self.ctx.plan):
            step_info = self.ctx.plan[self.ctx.current_step]

        await self.memory.create_note(
            title=f"Step {self.ctx.current_step + 1}: {_sanitize_step_title(step_info, 50)}",
            content=reflection,
            tags=["reflection", "auto", self.ctx.role],
            agent_id=self.ctx.agent_id,
            source=f"goal:{_truncate(self.ctx.goal, 80)}",
        )

    async def _build_messages(
        self,
        task: str,
        guidance: str | None = None,
    ) -> list[dict[str, str]]:
        """Build context-managed messages list."""
        memory_notes: list[str] = []
        cfg = self.memory.config
        strategy = self.ctx.metadata.get("memory_strategy") or cfg.memory_strategy
        limit = self.ctx.metadata.get("memory_search_limit")
        if limit is None:
            limit = cfg.memory_search_limit
        memory_tags = self.ctx.metadata.get("memory_tags")

        if strategy == "search":
            notes = await self.memory.search(
                self.ctx.goal or task, limit=limit, tags=memory_tags
            )
            memory_notes = [f"[{n.title}]\n{n.content}" for n in notes]
        elif strategy == "recent":
            notes = await self.memory.get_recent(limit=limit)
            if memory_tags:
                notes = [n for n in notes if any(t in n.tags for t in memory_tags)]
            memory_notes = [f"[{n.title}]\n{n.content}" for n in notes]
        elif strategy == "agent_notes":
            notes = await self.memory.get_agent_notes(self.ctx.agent_id, limit=limit)
            if memory_tags:
                notes = [n for n in notes if any(t in n.tags for t in memory_tags)]
            memory_notes = [f"[{n.title}]\n{n.content}" for n in notes]
        elif strategy == "hybrid":
            search_notes = await self.memory.search(
                self.ctx.goal or task, limit=limit // 2, tags=memory_tags
            )
            recent_notes = await self.memory.get_recent(limit=limit // 2)
            if memory_tags:
                tag_ids = self.memory._get_note_ids_with_any_tag(memory_tags)
                recent_notes = [n for n in recent_notes if n.id in tag_ids]
            seen: set[str] = set()
            for n in search_notes + recent_notes:
                if n.id not in seen:
                    seen.add(n.id)
                    memory_notes.append(f"[{n.title}]\n{n.content}")
            memory_notes = memory_notes[:limit]

        tool_defs = self.tools.to_prompt_format()
        image_urls = self.ctx.metadata.pop("image_urls", None)

        return self._context_builder.build(
            system_prompt=self.system_prompt,
            task=task,
            tools=tool_defs if tool_defs else None,
            memory_notes=memory_notes if memory_notes else None,
            history=self.ctx.history,
            guidance=guidance,
            image_urls=image_urls,
        )

    @staticmethod
    def _default_system_prompt() -> str:
        return (
            "You are a focused, efficient agent. Complete tasks step by step. "
            "Use tools when they help accomplish the task. "
            "Be concise in your responses. "
            "When using a tool, respond with a tool_call in the correct format."
        )


def _is_meta_reflection(text: str) -> bool:
    """True if the reflection looks like meta-commentary (e.g. 'Okay, the user wants...') rather than a real summary."""
    if not text or len(text.strip()) < 20:
        return False
    lower = text.strip().lower()
    meta_starts = ("okay,", "let me", "i need to", "the user wants", "so,", "first,", "alright,", "i will", "i should")
    if any(lower.startswith(s) for s in meta_starts):
        return True
    if "the user wants me" in lower or "the user might expect" in lower:
        return True
    return False


def _truncate(text: str, max_len: int) -> str:
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def _sanitize_step_title(step_info: str, max_len: int = 50) -> str:
    """One line, strip conversational prefix, truncate for use as note title."""
    line = step_info.strip().split("\n")[0].strip()
    prefixes = ("Okay, ", "So, ", "Let me ", "I'll ", "Alright, ", "Let's ", "Well, ")
    for p in prefixes:
        if line.startswith(p):
            line = line[len(p) :].strip()
            break
    if len(line) <= max_len:
        return line
    return line[: max_len - 3] + "..."


def _normalize_plan_step(raw: Any) -> str:
    """Turn a raw plan step (dict or string) into a short action string for Execute this step."""
    if isinstance(raw, dict):
        step = raw.get("description") or raw.get("step") or raw.get("name")
        if step is not None:
            return str(step).strip() or "Proceed with the task"
        return "Proceed with the task"
    s = str(raw).strip()
    if not s:
        return "Proceed with the task"
    filler = ("Okay, ", "So, ", "Let me ", "I'll ", "Alright, ", "Let's ", "Well, ")
    for p in filler:
        if s.startswith(p):
            s = s[len(p) :].strip()
            break
    if not s or len(s) < 10:
        return "Proceed with the task"
    if len(s) > 120:
        return s[:117] + "..."
    return s
