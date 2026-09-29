"""Goal decomposition and dependency-aware planning."""

from __future__ import annotations

from dataclasses import dataclass, field

from aof.inference.backend import InferenceBackend
from aof.inference.parsing import extract_json


@dataclass
class Goal:
    """A goal or sub-goal in the planning hierarchy."""

    description: str
    priority: int = 0
    parent_id: str | None = None
    status: str = "pending"  # pending | active | done | failed


@dataclass
class Plan:
    """A decomposed plan with steps and optional dependencies."""

    goals: list[Goal] = field(default_factory=list)
    steps: list[str] = field(default_factory=list)
    dependencies: dict[int, list[int]] = field(default_factory=dict)

    @property
    def ready_steps(self) -> list[tuple[int, str]]:
        """Steps whose dependencies are all resolved (done)."""
        done: set[int] = set()  # Track which steps are done externally
        ready = []
        for i, step in enumerate(self.steps):
            deps = self.dependencies.get(i, [])
            if all(d in done for d in deps):
                ready.append((i, step))
        return ready


class Planner:
    """Decomposes high-level goals into actionable plans using the LLM."""

    def __init__(self, backend: InferenceBackend) -> None:
        self.backend = backend

    async def decompose(self, goal: str, context: str = "") -> Plan:
        """Use the LLM to break a goal into sub-goals and ordered steps.

        The LLM is asked to produce JSON with:
          - steps: ordered list of action strings
          - sub_goals: optional list of higher-level objectives
          - dependencies: optional map of step_index -> [prerequisite indices]
        """
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a planning agent. Decompose goals into concrete, "
                    "actionable steps. Output valid JSON with a 'steps' array. "
                    "Optionally include 'dependencies' mapping step indices to "
                    "lists of prerequisite step indices."
                ),
            },
            {
                "role": "user",
                "content": f"Goal: {goal}\n\nContext: {context}" if context else f"Goal: {goal}",
            },
        ]

        result = await self.backend.complete_json(messages)

        # Parse steps (backend may return a list when model outputs a JSON array)
        if isinstance(result, list):
            raw_steps = result
        else:
            raw_steps = result.get("steps", [])
        if isinstance(raw_steps, str):
            raw_steps = [raw_steps]
        steps = [str(s) for s in raw_steps]

        # Parse sub-goals (only when result is a dict)
        goals = []
        for sg in (result.get("sub_goals", []) if isinstance(result, dict) else []):
            if isinstance(sg, str):
                goals.append(Goal(description=sg))
            elif isinstance(sg, dict):
                goals.append(
                    Goal(
                        description=sg.get("description", str(sg)),
                        priority=sg.get("priority", 0),
                    )
                )

        # Parse dependencies (only when result is a dict)
        deps: dict[int, list[int]] = {}
        raw_deps = result.get("dependencies", {}) if isinstance(result, dict) else {}
        if isinstance(raw_deps, dict):
            for k, v in raw_deps.items():
                try:
                    idx = int(k)
                    if isinstance(v, list):
                        deps[idx] = [int(d) for d in v]
                except (ValueError, TypeError):
                    pass

        return Plan(goals=goals, steps=steps, dependencies=deps)

    async def replan(
        self,
        original_goal: str,
        completed_steps: list[str],
        current_issue: str,
    ) -> Plan:
        """Re-plan when the original plan encounters issues."""
        context = (
            f"Original goal: {original_goal}\n"
            f"Completed steps: {completed_steps}\n"
            f"Current issue: {current_issue}\n"
            f"Create a revised plan to accomplish the remaining work."
        )
        return await self.decompose(original_goal, context=context)
