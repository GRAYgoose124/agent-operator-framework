"""Route generator: refill research backlog from done/backlog context using web search."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from aof.config import AppConfig

logger = logging.getLogger(__name__)

ROUTE_SYSTEM = (
    "You suggest new research directions. You have web_search and add_to_research_queue. "
    "Use web_search to explore related topics and recent results. "
    "Add 1-3 new, non-duplicate research questions with add_to_research_queue. "
    "Do not duplicate questions already in backlog or recent done. Be concise."
)
MAX_ADD_PER_RUN = 3


async def run_route_generator_loop(
    config: AppConfig,
    queue_path: str,
    control: dict,
    *,
    interval_seconds: int = 300,
    min_backlog_target: int = 2,
    backend_role: str = "micro",
) -> None:
    """Run a loop that refills the research queue when backlog is below target.

    Reads recent done and current backlog, runs one LLM round with web_search
    and add_to_research_queue, then sleeps. Exits when control['quit'] is True.
    """
    from aof.agent.base import BaseAgent
    from aof.research import ResearchQueue
    from aof.researcher.runner import _make_multi_backends, _make_services
    from aof.tools.builtin import discovery_tools

    memory, tools, _ = await _make_services(config)
    queue = ResearchQueue(queue_path)
    discovery_tools.register(tools, queue)

    backends = await _make_multi_backends(config, {backend_role, "general", "default"})
    backend = backends.get(backend_role) or backends.get("general") or backends.get("default")
    if not backend:
        backend = next(iter(backends.values()))

    route_tools = tools.filtered(["web_search", "add_to_research_queue"])
    agent = BaseAgent(
        backend=backend,
        memory=memory,
        tools=route_tools,
        system_prompt=ROUTE_SYSTEM,
        role=backend_role,
        max_tool_rounds=5,
    )

    while not control.get("quit", False):
        grouped = queue.list_all()
        backlog = grouped.get("backlog", [])
        done = grouped.get("done", [])
        if len(backlog) >= min_backlog_target:
            await asyncio.sleep(interval_seconds)
            continue

        recent_done = done[-10:] if len(done) > 10 else done
        backlog_preview = [i.question[:80] for i in backlog[:20]]
        done_preview = [i.question[:80] for i in recent_done]
        goal = (
            "Current backlog (do not duplicate):\n"
            + "\n".join(f"- {q}" for q in backlog_preview)
            + "\n\nRecent done:\n"
            + "\n".join(f"- {q}" for q in done_preview)
            + "\n\nUse web_search to find related or combined directions, then add 1-3 new research questions with add_to_research_queue."
        )
        try:
            agent.ctx.history = []
            await agent.execute_step(goal)
            logger.debug("Route generator ran one round; backlog now %d", len(queue.list_all().get("backlog", [])))
        except Exception as e:
            logger.warning("Route generator round failed: %s", e)
        await asyncio.sleep(interval_seconds)

    for be in set(backends.values()):
        await be.shutdown()
    await memory.close()
