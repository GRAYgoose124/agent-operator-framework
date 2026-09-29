"""Citation tracking tool for research pipeline."""

from __future__ import annotations

from typing import TYPE_CHECKING

from aof.research.context import current_research_item_id
from aof.tools.registry import ToolRegistry

if TYPE_CHECKING:
    from aof.research.queue import ResearchQueue


def register(registry: ToolRegistry, queue: ResearchQueue) -> None:
    """Register add_citation tool. Requires ResearchQueue instance."""

    @registry.register_function(
        name="add_citation",
        description="Add a citation for a source found during research. Store URL, title, and snippet. Use when you find relevant web pages, articles, or documents.",
        parameters={
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The URL of the source",
                },
                "title": {
                    "type": "string",
                    "description": "Title of the source (page, article, etc.)",
                },
                "snippet": {
                    "type": "string",
                    "description": "Brief excerpt or summary of the relevant content",
                },
            },
            "required": ["url", "title", "snippet"],
        },
    )
    async def add_citation(url: str, title: str, snippet: str) -> dict:
        item_id = current_research_item_id.get()
        if not item_id:
            return {"error": "No active research item. Use add_citation only during research."}
        queue.add_citation_to_item(item_id, url=url, title=title, snippet=snippet)
        return {"success": True, "message": f"Citation added: {title}"}
