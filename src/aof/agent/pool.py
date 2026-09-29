"""Concurrent agent pool with asyncio orchestration and evaluator integration."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable

from aof.agent.base import AgentContext, BaseAgent
from aof.agent.evaluator import AgentEvaluator
from aof.agent.lifecycle import AgentLifecycle
from aof.config import AppConfig
from aof.inference.backend import InferenceBackend
from aof.memory.git_ops import GitOps
from aof.memory.store import MemoryStore
from aof.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


# Type alias for agent factory functions
AgentFactory = Callable[[], BaseAgent]


class AgentPool:
    """Manages concurrent execution of multiple agents.

    Architecture:
      - Each agent runs as an asyncio.Task
      - Pool-level semaphore limits concurrent agents (8-16+)
      - While agents wait for inference, they yield to other tasks
      - Evaluator runs as a background task, monitoring health

    Multi-model support:
      - Different agent factories can use different backends
      - e.g. micro agents (0.6B) for search, medium agents (4B+) for planning
    """

    def __init__(
        self,
        backend: InferenceBackend,
        memory: MemoryStore,
        tools: ToolRegistry,
        config: AppConfig,
        git_ops: GitOps | None = None,
    ) -> None:
        self.backend = backend
        self.memory = memory
        self.tools = tools
        self.config = config
        self.git_ops = git_ops

        self._max_concurrent = config.pool.max_agents
        self._semaphore = asyncio.Semaphore(self._max_concurrent)
        self._agents: dict[str, BaseAgent] = {}
        self._tasks: dict[str, asyncio.Task] = {}
        self._evaluator = AgentEvaluator(config.evaluator)
        self._eval_task: asyncio.Task | None = None

        # Multi-model backend registry: role -> backend
        self._backends: dict[str, InferenceBackend] = {"default": backend}

    def register_backend(self, role: str, backend: InferenceBackend) -> None:
        """Register a backend for a specific agent role.

        Example roles:
          - "micro": Qwen3-0.6B for simple tasks
          - "small": Qwen3-4B for code generation
          - "medium": Qwen3-8B+ for orchestration
        """
        self._backends[role] = backend

    def get_backend(self, role: str) -> InferenceBackend:
        """Get the backend for a role, falling back to default."""
        return self._backends.get(role, self._backends["default"])

    async def submit(
        self,
        goal: str,
        *,
        role: str = "general",
        system_prompt: str = "",
        tool_names: list[str] | None = None,
        factory: AgentFactory | None = None,
        image_urls: list[str] | None = None,
    ) -> str:
        """Submit a goal for an agent to work on.

        Returns the agent_id. The agent runs concurrently.
        image_urls: optional image paths for VL (vision) models.
        """
        if factory:
            agent = factory()
        else:
            backend = self.get_backend(role)
            tools = self.tools.filtered(tool_names) if tool_names else self.tools
            agent = BaseAgent(
                backend=backend,
                memory=self.memory,
                tools=tools,
                system_prompt=system_prompt,
                role=role,
                max_tool_rounds=self.config.pipeline.max_tool_rounds,
            )
            if image_urls:
                agent.ctx.metadata["image_urls"] = image_urls

        agent_id = agent.ctx.agent_id
        self._agents[agent_id] = agent
        lifecycle = AgentLifecycle(agent, max_steps=self.config.pipeline.default_max_steps)

        async def _run() -> AgentContext:
            async with self._semaphore:
                ctx = await lifecycle.run(goal)
                # Auto-commit memory after agent completes
                if self.git_ops and self.config.memory.auto_commit:
                    await self.git_ops.auto_commit(
                        message=f"Agent completed: {goal[:60]}",
                        agent_id=agent_id,
                    )
                return ctx

        task = asyncio.create_task(_run(), name=f"agent-{agent_id}")
        self._tasks[agent_id] = task
        logger.info("Submitted agent %s (role=%s): %s", agent_id, role, goal[:80])
        return agent_id

    async def submit_batch(
        self,
        goals: list[str],
        *,
        role: str = "general",
        system_prompt: str = "",
    ) -> list[str]:
        """Submit multiple goals concurrently. Returns list of agent IDs."""
        ids = []
        for goal in goals:
            agent_id = await self.submit(
                goal, role=role, system_prompt=system_prompt
            )
            ids.append(agent_id)
        return ids

    async def wait(self, agent_id: str) -> AgentContext | None:
        """Wait for a specific agent to complete."""
        task = self._tasks.get(agent_id)
        if not task:
            return None
        try:
            return await task
        except Exception as e:
            logger.error("Agent %s failed: %s", agent_id, e)
            return None

    async def wait_all(self) -> dict[str, AgentContext | None]:
        """Wait for all submitted agents to complete."""
        results: dict[str, AgentContext | None] = {}
        for agent_id, task in self._tasks.items():
            try:
                results[agent_id] = await task
            except Exception as e:
                logger.error("Agent %s failed: %s", agent_id, e)
                results[agent_id] = None
        return results

    def get_agent(self, agent_id: str) -> BaseAgent | None:
        return self._agents.get(agent_id)

    def get_rankings(self) -> list[tuple[str, float, str]]:
        """Get current agent rankings (worst-first)."""
        active = [a for a in self._agents.values() if a.ctx.metrics.steps_attempted > 0]
        return self._evaluator.rank_agents(active)

    def get_status(self) -> dict[str, Any]:
        """Get pool status summary."""
        active = sum(1 for t in self._tasks.values() if not t.done())
        completed = sum(1 for t in self._tasks.values() if t.done())
        return {
            "total_agents": len(self._tasks),
            "active": active,
            "completed": completed,
            "max_concurrent": self._max_concurrent,
            "backends": list(self._backends.keys()),
        }

    async def start_evaluator(self) -> None:
        """Start the background evaluator that monitors agent health."""

        async def _eval_loop() -> None:
            while True:
                await asyncio.sleep(self.config.evaluator.check_interval_seconds)
                active_agents = [
                    a
                    for a in self._agents.values()
                    if a.ctx.metrics.steps_attempted > 0
                    and a.ctx.state.name not in ("DONE", "ERROR", "IDLE")
                ]
                for agent in active_agents:
                    guidance = self._evaluator.generate_guidance(agent)
                    if guidance:
                        # Store guidance in agent metadata for next prompt injection
                        agent.ctx.metadata["pending_guidance"] = guidance

                    upgrade = self._evaluator.should_suggest_model_change(agent)
                    if upgrade:
                        logger.warning(
                            "Agent %s (role=%s) may benefit from upgrade to '%s'",
                            agent.ctx.agent_id,
                            agent.ctx.role,
                            upgrade,
                        )
                        agent.ctx.metadata["suggested_upgrade"] = upgrade

        self._eval_task = asyncio.create_task(_eval_loop(), name="evaluator")
        logger.info("Evaluator started (interval=%.1fs)", self.config.evaluator.check_interval_seconds)

    async def stop_evaluator(self) -> None:
        if self._eval_task and not self._eval_task.done():
            self._eval_task.cancel()
            try:
                await self._eval_task
            except asyncio.CancelledError:
                pass
