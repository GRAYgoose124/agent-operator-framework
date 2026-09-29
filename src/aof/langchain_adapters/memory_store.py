"""LangGraph BaseStore wrapper for AOF MemoryStore (Zettelkasten)."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any, TYPE_CHECKING

from langgraph.store.base import BaseStore, GetOp, Item, PutOp, Result, SearchOp, SearchItem

if TYPE_CHECKING:
    from aof.memory.store import MemoryStore
    from aof.memory.zettel import ZettelNote

DEFAULT_NAMESPACE = ("zettel",)
FILESYSTEM_NAMESPACE = ("filesystem",)


def _note_to_value(note: ZettelNote) -> dict[str, Any]:
    """Serialize ZettelNote to store value dict."""
    return {
        "id": note.id,
        "title": note.title,
        "content": note.content,
        "tags": note.tags,
        "links": note.links,
        "created_at": note.created_at.isoformat(),
        "updated_at": note.updated_at.isoformat(),
        "source": note.source,
        "agent_id": note.agent_id,
    }


def _value_to_note(value: dict[str, Any]) -> ZettelNote:
    """Deserialize store value dict to ZettelNote."""
    from aof.memory.zettel import ZettelNote

    return ZettelNote(
        id=value.get("id", ""),
        title=value.get("title", "Untitled"),
        content=value.get("content", ""),
        tags=value.get("tags", []),
        links=value.get("links", []),
        created_at=datetime.fromisoformat(value["created_at"]) if value.get("created_at") else datetime.now(timezone.utc),
        updated_at=datetime.fromisoformat(value["updated_at"]) if value.get("updated_at") else datetime.now(timezone.utc),
        source=value.get("source", ""),
        agent_id=value.get("agent_id", ""),
    )


class ZettelStoreAdapter(BaseStore):
    """LangGraph BaseStore that wraps AOF MemoryStore for Zettelkasten notes."""

    def __init__(self, memory: MemoryStore) -> None:
        self._memory = memory

    def batch(self, ops: list) -> list[Result]:
        """Execute operations synchronously."""
        return asyncio.run(self.abatch(ops))

    async def abatch(self, ops: list) -> list[Result]:
        """Execute operations asynchronously."""
        results: list[Result] = []
        for op in ops:
            if isinstance(op, GetOp):
                item = await self._handle_get(op)
                results.append(item)
            elif isinstance(op, PutOp):
                await self._handle_put(op)
                results.append(None)
            elif isinstance(op, SearchOp):
                items = await self._handle_search(op)
                results.append(items)
            else:
                results.append(None)
        return results

    def _path_to_note_id(self, path: str) -> str:
        """Convert filesystem path to note id (safe for filename)."""
        import hashlib
        safe = path.replace("/", "_").replace("\\", "_").strip("_")
        if len(safe) > 60:
            h = hashlib.sha256(path.encode()).hexdigest()[:8]
            safe = safe[:52] + "_" + h
        return f"mem_{safe}" if safe else "mem_unknown"

    async def _handle_get(self, op: GetOp) -> Item | None:
        namespace = tuple(op.namespace) if hasattr(op, "namespace") else op[0]
        key = op.key if hasattr(op, "key") else op[1]
        ns = tuple(namespace)
        if ns == DEFAULT_NAMESPACE:
            note = await self._memory.get_note(key)
            if note is None:
                return None
            value = _note_to_value(note)
            return Item(
                namespace=list(namespace),
                key=key,
                value=value,
                created_at=note.created_at.isoformat(),
                updated_at=note.updated_at.isoformat(),
            )
        if ns == FILESYSTEM_NAMESPACE:
            note_id = self._path_to_note_id(key)
            note = await self._memory.get_note(note_id)
            if note is None:
                return None
            content = note.content
            return Item(
                namespace=list(namespace),
                key=key,
                value={"content": content} if isinstance(content, str) else content,
                created_at=note.created_at.isoformat(),
                updated_at=note.updated_at.isoformat(),
            )
        return None

    async def _handle_put(self, op: PutOp) -> None:
        namespace = tuple(op.namespace) if hasattr(op, "namespace") else op[0]
        key = op.key if hasattr(op, "key") else op[1]
        value = op.value if hasattr(op, "value") else op[2]
        if value is None:
            return
        ns = tuple(namespace)
        if ns == DEFAULT_NAMESPACE:
            note = _value_to_note(value)
            note.id = key
            existing = await self._memory.get_note(key)
            if existing:
                await self._memory.update_note(note)
            else:
                await self._memory.create_note(
                    title=note.title,
                    content=note.content,
                    tags=note.tags,
                    links=note.links,
                    source=note.source,
                    agent_id=note.agent_id,
                    note_id=key,
                )
        elif ns == FILESYSTEM_NAMESPACE:
            content = value.get("content", str(value)) if isinstance(value, dict) else str(value)
            note_id = self._path_to_note_id(key)
            from aof.memory.zettel import ZettelNote

            note = ZettelNote(
                id=note_id,
                title=key,
                content=content,
                tags=["memory", "filesystem"],
                source=f"path:{key}",
                agent_id="deep_agent",
            )
            existing = await self._memory.get_note(note_id)
            if existing:
                note.created_at = existing.created_at
                await self._memory.update_note(note)
            else:
                await self._memory.create_note(
                    title=note.title,
                    content=note.content,
                    tags=note.tags,
                    source=note.source,
                    agent_id=note.agent_id,
                    note_id=note_id,
                )

    async def _handle_search(self, op: SearchOp) -> list[SearchItem]:
        namespace = tuple(op.namespace_prefix) if hasattr(op, "namespace_prefix") else op[0]
        query = op.query if hasattr(op, "query") else op[4] if len(op) > 4 else None
        limit = op.limit if hasattr(op, "limit") else op[2] if len(op) > 2 else 10
        ns = tuple(namespace)
        if ns != DEFAULT_NAMESPACE and ns != FILESYSTEM_NAMESPACE:
            return []
        if query:
            notes = await self._memory.search(query, limit=limit)
        else:
            notes = await self._memory.get_recent(limit=limit)
        result_ns = list(ns) if ns else list(DEFAULT_NAMESPACE)
        return [
            SearchItem(
                namespace=result_ns,
                key=note.id,
                value=_note_to_value(note),
                created_at=note.created_at.isoformat(),
                updated_at=note.updated_at.isoformat(),
                score=None,
            )
            for note in notes
        ]
