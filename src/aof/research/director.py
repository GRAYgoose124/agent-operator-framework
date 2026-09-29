"""Research director: 1x 8B orchestrator + Nx 0.6B crawling experts."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from aof.agent.pool import AgentPool

from aof.agent.base import BaseAgent
from aof.agent.lifecycle import AgentLifecycle
from aof.config import AppConfig
from aof.inference.backend import InferenceBackend
from aof.memory.store import MemoryStore
from aof.research.context import current_research_item_id
from aof.research.models import ResearchItem
from aof.research.queue import ResearchQueue
from aof.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)

EXPERT_TOOL_NAMES = [
    "web_search", "web_news", "web_scrape", "add_citation",
    "crawl_site", "crawl_sitemap", "scrape_with_playwright", "follow_links", "extract_structured",
    "extract_links", "extract_meta", "parse_html_fragment",
    "search_memory", "search_citations", "refine_memory",
    "create_crawler",
]
DIRECTOR_TOOL_NAMES = ["search_memory", "search_citations", "refine_memory"]
DIRECTOR_SYSTEM = (
    "You are a research director. You decompose research questions into 2-4 concrete crawl/gather tasks. "
    "Each task should be a specific instruction for an expert (e.g. 'Search for X and scrape top 3 results', "
    "'Crawl https://example.com/docs for Y'). Output a JSON object with a 'steps' array of strings."
)
DIRECTOR_SYNTHESIZE_SYSTEM = (
    "You synthesize research findings from multiple experts into a clear, cited report. "
    "Use search_memory for context. Store key insights with refine_memory."
)


class ResearchDirector:
    """Orchestrates research: director (8B) plans and synthesizes; experts (0.6B) crawl."""

    def __init__(
        self,
        queue: ResearchQueue,
        pool: AgentPool,
        director_backend: InferenceBackend,
        memory: MemoryStore,
        tools: ToolRegistry,
        config: AppConfig,
        *,
        interrupt_path: Path | None = None,
        on_item_done: Callable[[str, str, list], None] | None = None,
    ) -> None:
        self.queue = queue
        self.pool = pool
        self.director_backend = director_backend
        self.memory = memory
        self.tools = tools
        self.config = config
        self._interrupt_path = Path(
            interrupt_path or config.research.interrupt_flag_path
        )
        self._on_item_done = on_item_done

    def _check_interrupt(self) -> bool:
        """Return True if interrupt requested."""
        return self._interrupt_path.exists()

    def _clear_interrupt(self) -> None:
        """Remove interrupt flag."""
        if self._interrupt_path.exists():
            self._interrupt_path.unlink()

    async def _director_plan(self, question: str) -> list[str]:
        """Use director agent to decompose question into expert tasks."""
        director_tools = self.tools.filtered(DIRECTOR_TOOL_NAMES)
        agent = BaseAgent(
            backend=self.director_backend,
            memory=self.memory,
            tools=director_tools,
            system_prompt=DIRECTOR_SYSTEM,
            max_tool_rounds=2,
        )
        lifecycle = AgentLifecycle(agent, max_steps=1)
        ctx = await lifecycle.run(
            f"Decompose this research question into 2-4 crawl/gather tasks: {question}"
        )
        if not ctx.results:
            return [f"Research: {question}"]
        last = ctx.results[-1]
        text = last.text if hasattr(last, "text") else str(last)
        # Parse JSON steps from response
        import json
        import re
        try:
            match = re.search(r'\{[^{}]*"steps"[^{}]*\}', text)
            if match:
                obj = json.loads(match.group())
                steps = obj.get("steps", [])
                if isinstance(steps, list) and steps:
                    return [str(s) for s in steps]
        except json.JSONDecodeError:
            pass
        # Fallback: split by newlines or numbered items
        lines = [l.strip() for l in text.split("\n") if l.strip()]
        steps = []
        for line in lines:
            m = re.match(r"^\d+[\.\)]\s*(.+)", line)
            steps.append(m.group(1) if m else line)
        return steps[: self.config.research.expert_count] if steps else [f"Research: {question}"]

    async def _director_synthesize(self, question: str, expert_results: list[str]) -> str:
        """Use director agent to synthesize expert findings."""
        director_tools = self.tools.filtered(DIRECTOR_TOOL_NAMES)
        agent = BaseAgent(
            backend=self.director_backend,
            memory=self.memory,
            tools=director_tools,
            system_prompt=DIRECTOR_SYNTHESIZE_SYSTEM,
            max_tool_rounds=3,
        )
        combined = "\n\n---\n\n".join(expert_results)
        lifecycle = AgentLifecycle(agent, max_steps=2)
        ctx = await lifecycle.run(
            f"Research question: {question}\n\nExpert findings:\n{combined}\n\nSynthesize into a report."
        )
        if ctx.results:
            last = ctx.results[-1]
            return last.text if hasattr(last, "text") else str(last)
        return combined

    def _build_checkpoint(self, item: ResearchItem, expert_outputs: list[str]) -> None:
        """Save checkpoint to item for resume on interrupt."""
        sources = [c.to_dict() for c in item.citation_entries]
        partial = "\n\n---\n\n".join(expert_outputs) if expert_outputs else ""
        self.queue.save_checkpoint(
            item.id,
            sources_gathered=sources,
            partial_findings=partial,
            step_index=1,
        )

    async def process_item(self, item: ResearchItem) -> bool:
        """Process one research item. Returns True if done, False if interrupted."""
        if self._check_interrupt():
            self.queue.interrupt(item.id)
            logger.info("Interrupt requested, pushed item %s to backlog", item.id)
            return False

        is_resume = item.checkpoint is not None
        if is_resume:
            cp = item.checkpoint
            goals = [
                f"Continue research from previous run. "
                f"Sources gathered so far: {len(cp.get('sources_gathered', []))} items. "
                f"Partial findings:\n{cp.get('partial_findings', '')[:500]}...\n\n"
                f"Add more sources and findings as needed."
            ]
            expert_count = 1
        else:
            plan = await self._director_plan(item.question)
            expert_count = min(len(plan), self.config.research.expert_count)
            goals = plan[:expert_count]

        expert_tools = self.tools.filtered(EXPERT_TOOL_NAMES)
        base_prompt = (
            "You are a research expert. Use web_search, web_scrape, add_citation, crawl_site, or other tools to gather information. "
            "When you find relevant sources, call add_citation(url, title, snippet) to record them. "
            "Be thorough. Return your findings as structured text."
        )
        if item.human_notes:
            notes = "\n".join(f"- {n}" for n in item.human_notes)
            base_prompt += f"\n\nHuman guidance for this task:\n{notes}"
        if is_resume:
            expert_prompt = (
                base_prompt
                + f"\n\nResume from previous run:\nSources: {item.checkpoint.get('sources_gathered', [])}\n"
                f"Partial findings: {item.checkpoint.get('partial_findings', '')[:800]}"
            )
        else:
            expert_prompt = base_prompt

        token = current_research_item_id.set(item.id)
        try:
            agent_ids = []
            for goal in goals:
                agent_id = await self.pool.submit(
                    goal,
                    role=self.config.research.expert_role,
                    system_prompt=expert_prompt,
                    tool_names=EXPERT_TOOL_NAMES,
                )
                agent_ids.append(agent_id)

            results = await self.pool.wait_all()
        finally:
            current_research_item_id.reset(token)
        expert_outputs = []
        for agent_id in agent_ids:
            ctx = results.get(agent_id)
            if ctx and ctx.results:
                last = ctx.results[-1]
                text = last.text if hasattr(last, "text") else str(last)
                expert_outputs.append(text)
            else:
                expert_outputs.append("(No output)")

        if self._check_interrupt():
            self._build_checkpoint(item, expert_outputs)
            self.queue.interrupt(item.id)
            logger.info("Interrupt requested, saved checkpoint for item %s", item.id)
            return False

        report = await self._director_synthesize(item.question, expert_outputs)
        content_with_sources = report
        if item.citation_entries:
            content_with_sources += "\n\n## Sources\n" + "\n".join(
                f"- [{c.title}]({c.url}): {c.snippet}" for c in item.citation_entries
            )
        note = await self.memory.create_note(
            title=f"Research: {item.question[:50]}...",
            content=content_with_sources,
            tags=["research", "synthesis", item.id],
            source=f"research:{item.id}",
            agent_id="director",
        )
        if item.citation_entries:
            await self.memory.add_citations_to_vector_store(
                item.citation_entries, source_id=note.id
            )
        self.queue.mark_done(item.id)
        logger.info("Research item %s done", item.id)
        if self._on_item_done:
            sources = [c.to_dict() for c in item.citation_entries]
            self._on_item_done(item.id, report, sources, question=item.question)
        return True

    async def run(self, limit: int | None = None) -> int:
        """Process queue until empty or limit. Returns number processed."""
        processed = 0
        while True:
            item = self.queue.pop_next()
            if item is None:
                break
            if limit is not None and processed >= limit:
                self.queue.interrupt(item.id)
                break
            done = await self.process_item(item)
            if done:
                processed += 1
            else:
                break
        return processed


