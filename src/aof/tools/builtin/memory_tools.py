"""Memory refinement tools for consolidating and improving Zettelkasten notes."""

from __future__ import annotations

from typing import TYPE_CHECKING

from aof.tools.registry import ToolRegistry

if TYPE_CHECKING:
    from aof.memory.store import MemoryStore


def register(registry: ToolRegistry, memory: MemoryStore) -> None:
    """Register memory refinement tools. Requires MemoryStore instance."""

    @registry.register_function(
        name="refine_memory",
        description="Merge or consolidate related memory notes. Create a new refined note from multiple source notes, or update tags on existing notes.",
        parameters={
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "description": "Action: 'merge' to create a new note from sources, or 'update_tags' to update tags on a note",
                    "enum": ["merge", "update_tags"],
                },
                "source_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "For merge: list of note IDs to consolidate",
                },
                "note_id": {
                    "type": "string",
                    "description": "For update_tags: the note ID to update",
                },
                "title": {
                    "type": "string",
                    "description": "For merge: title for the new consolidated note",
                },
                "content": {
                    "type": "string",
                    "description": "For merge: synthesized content for the new note",
                },
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Tags for the new note (merge) or updated tags (update_tags)",
                },
            },
            "required": ["action"],
        },
    )
    async def refine_memory(
        action: str,
        source_ids: list[str] | None = None,
        note_id: str | None = None,
        title: str | None = None,
        content: str | None = None,
        tags: list[str] | None = None,
    ) -> dict:
        try:
            if action == "merge":
                if not source_ids or not title or not content:
                    return {"error": "merge requires source_ids, title, and content"}
                refs = ", ".join(f"[[{sid}]]" for sid in source_ids)
                note = await memory.create_note(
                    title=title,
                    content=content + f"\n\n## Source notes\n{refs}",
                    tags=tags or ["refined", "merged"],
                    source=f"merged:{','.join(source_ids)}",
                    agent_id="refine_memory",
                )
                return {"success": True, "note_id": note.id, "title": note.title}
            elif action == "update_tags":
                if not note_id or tags is None:
                    return {"error": "update_tags requires note_id and tags"}
                note = await memory.get_note(note_id)
                if not note:
                    return {"error": f"Note not found: {note_id}"}
                note.tags = tags
                await memory.update_note(note)
                return {"success": True, "note_id": note_id, "tags": tags}
            else:
                return {"error": f"Unknown action: {action}"}
        except Exception as e:
            return {"error": str(e)}

    @registry.register_function(
        name="search_memory",
        description="Search the Zettelkasten memory by query. Returns relevant notes for context.",
        parameters={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query (keywords or natural language)",
                },
                "limit": {
                    "type": "integer",
                    "description": "Max results (default 5)",
                },
            },
            "required": ["query"],
        },
    )
    async def search_memory(query: str, limit: int = 5) -> list[dict]:
        notes = await memory.search(query, limit=limit)
        return [
            {"id": n.id, "title": n.title, "content": n.content[:500], "tags": n.tags}
            for n in notes
        ]

    @registry.register_function(
        name="search_citations",
        description="Search research citations by semantic similarity. Returns sources (url, title, snippet) relevant to the query. Use when you need to find sources about a topic.",
        parameters={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query (topic or natural language)",
                },
                "limit": {
                    "type": "integer",
                    "description": "Max results (default 5)",
                },
            },
            "required": ["query"],
        },
    )
    async def search_citations(query: str, limit: int = 5) -> list[dict]:
        return await memory.search_citations(query, limit=limit)
