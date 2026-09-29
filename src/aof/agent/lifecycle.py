"""Agent lifecycle orchestration: init -> plan -> execute -> reflect -> store."""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

from aof.agent.base import AgentContext, AgentState, BaseAgent

if TYPE_CHECKING:
    from aof.agent.planner import Planner

logger = logging.getLogger(__name__)


class AgentLifecycle:
    """Drives a single agent through its full lifecycle.

    Phases:
      1. Plan: decompose goal into steps
      2. Execute: run each step (with tool calls)
      3. Reflect: evaluate the result
      4. Store: persist learnings to memory
    """

    def __init__(
        self,
        agent: BaseAgent,
        max_steps: int = 10,
        *,
        planner: Planner | None = None,
        step_retries: int = 0,
        skip_plan: bool = False,
    ) -> None:
        self.agent = agent
        self.max_steps = max_steps
        self._planner = planner
        self._step_retries = step_retries
        self._skip_plan = skip_plan

    async def run(self, goal: str) -> AgentContext:
        """Run the agent through its full lifecycle for a given goal."""
        ctx = self.agent.ctx
        ctx.goal = goal
        ctx.state = AgentState.PLANNING

        logger.info("Agent %s starting: %s", ctx.agent_id, goal[:80])

        try:
            # Plan (skip when single-step/direct: no tools and max_steps==1)
            if self._skip_plan:
                plan = [goal]
                ctx.plan = plan
            elif self._planner:
                plan_obj = await self._planner.decompose(goal)
                plan = plan_obj.steps
            else:
                plan = await self.agent.plan()
            logger.info(
                "Agent %s planned %d steps: %s",
                ctx.agent_id,
                len(plan),
                [str(s)[:40] for s in plan],
            )

            # Execute each step
            for i, step in enumerate(plan[: self.max_steps]):
                ctx.current_step = i
                ctx.state = AgentState.EXECUTING
                ctx.metrics.steps_attempted += 1

                t0 = time.monotonic()
                result = None
                last_error: Exception | None = None
                for attempt in range(self._step_retries + 1):
                    try:
                        result = await self.agent.execute_step(step)
                        ctx.results.append(result)
                        ctx.metrics.steps_completed += 1
                        break
                    except Exception as e:
                        last_error = e
                        ctx.metrics.errors.append(f"Step {i} (attempt {attempt + 1}): {e}")
                        logger.warning(
                            "Agent %s step %d failed (attempt %d): %s",
                            ctx.agent_id, i, attempt + 1, e,
                        )
                        if attempt < self._step_retries:
                            continue
                if result is None:
                    if last_error:
                        ctx.metrics.errors.append(f"Step {i}: {last_error}")
                        raise last_error
                    continue
                elapsed = (time.monotonic() - t0) * 1000
                ctx.metrics.latencies_ms.append(elapsed)

                # Reflect
                ctx.state = AgentState.REFLECTING
                try:
                    reflection = await self.agent.reflect(result)
                except Exception as e:
                    reflection = f"Reflection failed: {e}"
                    logger.warning("Agent %s reflection failed: %s", ctx.agent_id, e)

                # Store (skip when reflection is the error string to avoid polluting memory)
                ctx.state = AgentState.STORING
                if reflection.startswith("Reflection failed:"):
                    logger.debug("Agent %s skipping store (reflection failed)", ctx.agent_id)
                else:
                    try:
                        await self.agent.store(reflection)
                    except Exception as e:
                        logger.warning("Agent %s store failed: %s", ctx.agent_id, e)

            ctx.state = AgentState.DONE
            logger.info(
                "Agent %s completed. Steps: %d/%d, Health: %.2f",
                ctx.agent_id,
                ctx.metrics.steps_completed,
                ctx.metrics.steps_attempted,
                ctx.metrics.health_score,
            )

        except Exception as e:
            ctx.state = AgentState.ERROR
            ctx.metrics.errors.append(f"Lifecycle error: {e}")
            logger.error("Agent %s failed: %s", ctx.agent_id, e, exc_info=True)

        return ctx
