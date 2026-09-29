"""Dual-storage memory: SQLite FTS5 index + markdown files."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from aof.config import MemoryConfig
from aof.memory.zettel import ZettelNote

if TYPE_CHECKING:
    from aof.research.models import Citation

logger = logging.getLogger(__name__)


def _vector_db_path(config: MemoryConfig) -> Path:
    """Resolve vector DB path; default is {notes_dir}/chroma."""
    if config.vector_db_path:
        return Path(config.vector_db_path)
    return Path(config.notes_dir) / "chroma"


class MemoryStore:
    """Zettelkasten memory with SQLite for search and markdown for persistence.

    SQLite provides fast full-text search via FTS5.
    Markdown files are human-readable and git-trackable.
    """

    def __init__(self, config: MemoryConfig) -> None:
        self._config = config
        self._db_path = Path(config.db_path)
        self._notes_dir = Path(config.notes_dir)
        self._db: sqlite3.Connection | None = None
        self._vector_store = None
        self._citation_store = None
        self._agentic_adapter = None
        if config.memory_strategy == "hybrid":
            try:
                from aof.memory.vector_store import CitationVectorStore, VectorStore

                vpath = _vector_db_path(config)
                self._vector_store = VectorStore(
                    persist_directory=vpath,
                    embedding_model=config.embedding_model,
                )
                self._citation_store = CitationVectorStore(
                    persist_directory=vpath,
                    embedding_model=config.embedding_model,
                )
            except ImportError:
                logger.warning(
                    "Hybrid memory requires chromadb and sentence-transformers; "
                    "falling back to FTS-only search"
                )

    @property
    def config(self) -> MemoryConfig:
        return self._config

    async def initialize(self) -> None:
        """Create tables and ensure directories exist."""
        self._notes_dir.mkdir(parents=True, exist_ok=True)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)

        self._db = sqlite3.connect(str(self._db_path))
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute(
            """
            CREATE TABLE IF NOT EXISTS notes (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                tags TEXT,
                links TEXT,
                created_at TEXT,
                updated_at TEXT,
                source TEXT,
                agent_id TEXT,
                content_preview TEXT
            )
            """
        )
        # FTS5 for full-text search
        self._db.execute(
            """
            CREATE VIRTUAL TABLE IF NOT EXISTS notes_fts
            USING fts5(id, title, content_preview, tags)
            """
        )
        self._db.commit()
        if getattr(self._config, "agentic_enabled", False):
            try:
                from aof.memory.agentic_adapter import AgenticAdapter
                self._agentic_adapter = AgenticAdapter(self._config)
                if self._agentic_adapter.enabled:
                    logger.info("Agentic memory adapter enabled")
            except Exception as e:
                logger.warning("Agentic adapter not available: %s", e)
        logger.info("Memory store initialized at %s", self._db_path)

    async def close(self) -> None:
        if self._db:
            self._db.close()
            self._db = None

    async def create_note(
        self,
        title: str,
        content: str,
        tags: list[str],
        links: list[str] | None = None,
        source: str = "",
        agent_id: str = "",
        note_id: str | None = None,
        *,
        kind: str = "note",
        status: str = "raw",
        sources: list[str] | None = None,
        supersedes: list[str] | None = None,
        confidence: float | None = None,
    ) -> ZettelNote:
        """Create a new note, persist to markdown + index in SQLite."""
        note = ZettelNote(
            id=note_id or ZettelNote.new_id(agent_id, content),
            title=title,
            content=content,
            tags=tags,
            links=links or [],
            source=source,
            agent_id=agent_id,
            kind=kind,
            status=status,
            sources=sources or [],
            supersedes=supersedes or [],
            confidence=confidence,
        )

        # Write markdown file
        file_path = self._notes_dir / f"{note.id}.md"
        file_path.write_text(note.to_markdown(), encoding="utf-8")

        # Index in SQLite
        self._index_note(note)

        # Index in vector store when hybrid
        if self._vector_store:
            await self._add_to_vector_store(note)

        # Mirror into A-MEM when agentic memory is enabled
        if self._agentic_adapter and self._agentic_adapter.enabled:
            await self._agentic_adapter.add_note(
                note.content,
                title=note.title,
                tags=note.tags,
            )

        logger.debug("Created note %s: %s", note.id, title)
        return note

    async def update_note(self, note: ZettelNote) -> None:
        """Update an existing note."""
        note.updated_at = datetime.now(timezone.utc)
        file_path = self._notes_dir / f"{note.id}.md"
        file_path.write_text(note.to_markdown(), encoding="utf-8")
        self._index_note(note)
        if self._vector_store:
            await self._add_to_vector_store(note)

    async def get_note(self, note_id: str) -> ZettelNote | None:
        """Read a note from its markdown file."""
        file_path = self._notes_dir / f"{note_id}.md"
        if not file_path.exists():
            return None
        text = file_path.read_text(encoding="utf-8")
        return ZettelNote.from_markdown(text)

    async def _get_notes_batch(self, ids: list[str]) -> list[ZettelNote]:
        """Load multiple notes by id in parallel. Preserves order; skips missing."""
        if not ids:
            return []
        notes = await asyncio.gather(*[self.get_note(nid) for nid in ids])
        return [n for n in notes if n is not None]

    async def search(
        self,
        query: str,
        limit: int = 5,
        tags: list[str] | None = None,
    ) -> list[ZettelNote]:
        """Full-text search via FTS5; when hybrid strategy, merge with vector similarity.
        When agentic memory is enabled, search A-MEM first and fall back to FTS if needed.

        When tags is provided, filter results to notes that have any of the given tags.
        """
        assert self._db is not None

        # Agentic path: search A-MEM first when enabled
        if self._agentic_adapter and self._agentic_adapter.enabled:
            try:
                loop = asyncio.get_running_loop()
                raw = await loop.run_in_executor(
                    None,
                    lambda: self._agentic_adapter.search(query, k=limit * 2 if tags else limit),
                )
                notes: list[ZettelNote] = []
                for r in raw:
                    note_tags = r.get("tags") or []
                    if tags and not any(t in note_tags for t in tags):
                        continue
                    notes.append(
                        ZettelNote(
                            id=r.get("id", ""),
                            title=r.get("title", "Untitled"),
                            content=(r.get("content") or "")[: self._config.fts_preview_chars * 2],
                            tags=note_tags,
                            links=[],
                            source="agentic",
                            agent_id="a-mem",
                        )
                    )
                if notes:
                    return notes[:limit]
            except Exception as e:
                logger.debug("Agentic search fallback to FTS: %s", e)

        if self._vector_store and self._config.memory_strategy == "hybrid":
            await self._ensure_vector_backfill()
            notes = await self._hybrid_search(query, limit * 2 if tags else limit)
        else:
            fts_query = " OR ".join(f'"{word}"*' for word in query.split() if word)
            if tags:
                tag_ids = self._get_note_ids_with_any_tag(tags)
                if not tag_ids:
                    return []
                id_list = list(tag_ids)[:500]
                placeholders = ",".join("?" * len(id_list))
                try:
                    rows = self._db.execute(
                        f"SELECT id FROM notes_fts WHERE notes_fts MATCH ? AND id IN ({placeholders}) LIMIT ?",
                        [fts_query] + id_list + [limit],
                    ).fetchall()
                except sqlite3.OperationalError:
                    rows = self._db.execute(
                        "SELECT id FROM notes WHERE (title LIKE ? OR content_preview LIKE ?) AND id IN ({}) LIMIT ?".format(placeholders),
                        [f"%{query}%", f"%{query}%"] + id_list + [limit],
                    ).fetchall()
            else:
                try:
                    rows = self._db.execute(
                        "SELECT id FROM notes_fts WHERE notes_fts MATCH ? LIMIT ?",
                        (fts_query, limit),
                    ).fetchall()
                except sqlite3.OperationalError:
                    rows = self._db.execute(
                        "SELECT id FROM notes WHERE title LIKE ? OR content_preview LIKE ? LIMIT ?",
                        (f"%{query}%", f"%{query}%", limit),
                    ).fetchall()
            note_ids = [row[0] for row in rows]
            note_ids = self._order_ids_by_created_at(note_ids)
            notes = await self._get_notes_batch(note_ids)
            if tags:
                filtered_ids = self._get_note_ids_with_any_tag(tags)
                notes = [n for n in notes if n.id in filtered_ids]
        return notes[:limit]

    async def search_semantic(self, query: str, limit: int = 5) -> list[ZettelNote]:
        """Pure vector similarity search. Requires hybrid strategy and vector store."""
        if not self._vector_store:
            return []
        loop = asyncio.get_running_loop()
        pairs = await loop.run_in_executor(
            None, lambda: self._vector_store.search(query, limit=limit)
        )
        note_ids = [note_id for note_id, _ in pairs]
        return await self._get_notes_batch(note_ids)

    async def search_by_tag(self, tag: str, limit: int = 10) -> list[ZettelNote]:
        """Find notes with a specific tag."""
        assert self._db is not None
        rows = self._db.execute(
            "SELECT id FROM notes WHERE tags LIKE ? LIMIT ?",
            (f"%{tag}%", limit),
        ).fetchall()
        note_ids = [row[0] for row in rows]
        return await self._get_notes_batch(note_ids)

    async def get_notes_by_tags(
        self, tags: list[str], limit: int = 50, exclude_tags: list[str] | None = None,
    ) -> list[ZettelNote]:
        """Get notes that have any of the given tags, optionally excluding notes with certain tags."""
        assert self._db is not None
        if not tags:
            return []
        conditions = " OR ".join("tags LIKE ?" for _ in tags)
        params: list = [f'%"{t}"%' for t in tags]
        if exclude_tags:
            for et in exclude_tags:
                conditions += " AND tags NOT LIKE ?"
                params.append(f'%"{et}"%')
        rows = self._db.execute(
            f"SELECT id FROM notes WHERE ({conditions}) ORDER BY created_at DESC LIMIT ?",
            params + [limit],
        ).fetchall()
        note_ids = [row[0] for row in rows]
        return await self._get_notes_batch(note_ids)

    async def get_linked(self, note_id: str) -> list[ZettelNote]:
        """Get all notes linked from the given note."""
        note = await self.get_note(note_id)
        if not note:
            return []
        results = []
        for lid in note.links:
            linked = await self.get_note(lid)
            if linked:
                results.append(linked)
        return results

    async def get_recent(self, limit: int = 10) -> list[ZettelNote]:
        """Get the most recently created notes."""
        assert self._db is not None
        rows = self._db.execute(
            "SELECT id FROM notes ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        note_ids = [row[0] for row in rows]
        return await self._get_notes_batch(note_ids)

    async def get_agent_notes(self, agent_id: str, limit: int = 20) -> list[ZettelNote]:
        """Get notes created by a specific agent."""
        assert self._db is not None
        rows = self._db.execute(
            "SELECT id FROM notes WHERE agent_id = ? ORDER BY created_at DESC LIMIT ?",
            (agent_id, limit),
        ).fetchall()
        note_ids = [row[0] for row in rows]
        return await self._get_notes_batch(note_ids)

    async def count(self) -> int:
        assert self._db is not None
        row = self._db.execute("SELECT COUNT(*) FROM notes").fetchone()
        return row[0] if row else 0

    async def add_citations_to_vector_store(
        self,
        citations: list[Citation],
        source_id: str = "",
    ) -> None:
        """Add citations to the citation vector store for semantic retrieval.

        Requires hybrid memory strategy. Call when storing research with citations.
        """
        if not self._citation_store:
            return
        from aof.research.models import Citation as CitationCls

        def _add_one(c: CitationCls, cid: str, sid: str) -> None:
            self._citation_store.add(cid, c.url, c.title, c.snippet, sid)

        loop = asyncio.get_running_loop()
        for i, c in enumerate(citations):
            if not isinstance(c, CitationCls):
                continue
            cid = hashlib.md5(f"{c.url}{c.title}{source_id}{i}".encode()).hexdigest()[:16]
            await loop.run_in_executor(None, _add_one, c, cid, source_id)

    async def search_citations(self, query: str, limit: int = 5) -> list[dict]:
        """Search citations by semantic similarity.

        Returns list of dicts with url, title, snippet, source_id.
        Requires hybrid memory strategy.
        """
        if not self._citation_store:
            return []
        loop = asyncio.get_running_loop()
        pairs = await loop.run_in_executor(
            None, lambda: self._citation_store.search(query, limit=limit)
        )
        return [meta for meta, _ in pairs if meta]

    # -- private ------------------------------------------------------------

    async def _ensure_vector_backfill(self) -> None:
        """On first hybrid use, backfill existing markdown notes into vector store."""
        if not self._vector_store or not self._db:
            return
        loop = asyncio.get_running_loop()
        db_count = self._db.execute("SELECT COUNT(*) FROM notes").fetchone()[0]
        vs_count = await loop.run_in_executor(None, self._vector_store.count)
        if vs_count >= db_count:
            return
        # Backfill: load all notes from SQLite, add to vector store
        rows = self._db.execute("SELECT id FROM notes").fetchall()
        for (note_id,) in rows:
            note = await self.get_note(note_id)
            if note:
                await self._add_to_vector_store(note)
        logger.info("Vector store backfilled %d notes", len(rows))

    async def _add_to_vector_store(self, note: ZettelNote) -> None:
        """Add note to vector store (runs in executor to avoid blocking)."""
        text = f"{note.title}\n{note.content}"
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            None,
            lambda: self._vector_store.add(
                note.id,
                text,
                metadata={"title": note.title},
            ),
        )

    def _get_note_ids_with_any_tag(self, tags: list[str]) -> set[str]:
        """Return note ids that have any of the given tags."""
        assert self._db is not None
        if not tags:
            return set()
        conditions = " OR ".join("tags LIKE ?" for _ in tags)
        params = [f'%"{t}"%' for t in tags]
        rows = self._db.execute(
            f"SELECT id FROM notes WHERE {conditions}",
            params,
        ).fetchall()
        return {row[0] for row in rows}

    def _order_ids_by_created_at(self, ids: list[str]) -> list[str]:
        """Return the given note ids sorted by created_at DESC (newer first)."""
        assert self._db is not None
        if not ids:
            return []
        placeholders = ",".join("?" * len(ids))
        rows = self._db.execute(
            f"SELECT id FROM notes WHERE id IN ({placeholders}) ORDER BY created_at DESC",
            ids,
        ).fetchall()
        return [row[0] for row in rows]

    async def _hybrid_search(self, query: str, limit: int) -> list[ZettelNote]:
        """Run FTS and vector search in parallel; merge with FTS-first ordering."""
        fts_query = " OR ".join(f'"{word}"*' for word in query.split() if word)
        fts_rows = []
        try:
            fts_rows = self._db.execute(
                "SELECT id FROM notes_fts WHERE notes_fts MATCH ? LIMIT ?",
                (fts_query, limit),
            ).fetchall()
        except sqlite3.OperationalError:
            fts_rows = self._db.execute(
                "SELECT id FROM notes WHERE title LIKE ? OR content_preview LIKE ? LIMIT ?",
                (f"%{query}%", f"%{query}%", limit),
            ).fetchall()

        loop = asyncio.get_running_loop()
        vector_pairs = await loop.run_in_executor(
            None, lambda: self._vector_store.search(query, limit=limit)
        )

        seen: set[str] = set()
        ordered_ids: list[str] = []
        for (note_id,) in fts_rows:
            if note_id not in seen:
                seen.add(note_id)
                ordered_ids.append(note_id)
        for note_id, _ in vector_pairs:
            if note_id not in seen and len(ordered_ids) < limit:
                seen.add(note_id)
                ordered_ids.append(note_id)
        ordered_ids = self._order_ids_by_created_at(ordered_ids)
        notes = await self._get_notes_batch(ordered_ids)
        return notes[:limit]

    def _index_note(self, note: ZettelNote) -> None:
        """Insert or update the SQLite index + FTS table."""
        assert self._db is not None
        max_preview = getattr(self._config, "fts_preview_chars", 500)
        preview = note.content[:max_preview]
        tags_json = json.dumps(note.tags)
        links_json = json.dumps(note.links)

        # Upsert main table
        self._db.execute(
            """
            INSERT INTO notes (id, title, tags, links, created_at, updated_at, source, agent_id, content_preview)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                title=excluded.title, tags=excluded.tags, links=excluded.links,
                updated_at=excluded.updated_at, content_preview=excluded.content_preview
            """,
            (
                note.id, note.title, tags_json, links_json,
                note.created_at.isoformat(), note.updated_at.isoformat(),
                note.source, note.agent_id, preview,
            ),
        )

        # Upsert FTS table (delete + insert since FTS5 doesn't support ON CONFLICT)
        self._db.execute("DELETE FROM notes_fts WHERE id = ?", (note.id,))
        self._db.execute(
            "INSERT INTO notes_fts (id, title, content_preview, tags) VALUES (?, ?, ?, ?)",
            (note.id, note.title, preview, tags_json),
        )
        self._db.commit()
