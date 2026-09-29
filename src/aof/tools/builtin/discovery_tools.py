"""Discovery tools: add promising topics to research queue."""

from __future__ import annotations

from typing import TYPE_CHECKING

from aof.tools.registry import ToolRegistry

if TYPE_CHECKING:
    from aof.research.queue import ResearchQueue


def register(registry: ToolRegistry, research_queue: ResearchQueue) -> None:
    """Register add_to_research_queue tool. Requires ResearchQueue instance."""

    @registry.register_function(
        name="add_to_research_queue",
        description="Add a research question or topic to the research backlog. Use when you discover an interesting topic, URL, or question worth deeper research.",
        parameters={
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "Research question or topic to add to the backlog",
                },
            },
            "required": ["question"],
        },
    )
    async def add_to_research_queue(question: str) -> dict:
        if not (question or "").strip():
            return {"error": "Question cannot be empty."}
        item_id = research_queue.add(question.strip())
        return {"success": True, "item_id": item_id, "message": f"Added to research backlog: {question[:60]}..."}
