"""Research queue with Kanban persistence."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from aof.research.models import Citation, ResearchItem, ResearchItemStatus

logger = logging.getLogger(__name__)


class ResearchQueue:
    """Kanban-style research queue with JSON persistence."""

    def __init__(self, queue_path: str | Path) -> None:
        self._path = Path(queue_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._items: dict[str, ResearchItem] = {}
        self._load()

    def _load(self) -> None:
        """Load queue from disk."""
        if not self._path.exists():
            self._items = {}
            return
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
            self._items = {
                item["id"]: ResearchItem.from_dict(item)
                for item in data.get("items", [])
            }
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning("Failed to load research queue: %s", e)
            self._items = {}

    def reload(self) -> None:
        """Reload queue from disk (e.g. after REPL or route generator updates)."""
        self._load()

    def _save(self) -> None:
        """Persist queue to disk."""
        data = {
            "items": [item.to_dict() for item in self._items.values()],
        }
        self._path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def add(self, question: str) -> str:
        """Add a research question to backlog. Returns item ID."""
        item = ResearchItem(
            id=ResearchItem.new_id(),
            question=question.strip(),
            status=ResearchItemStatus.BACKLOG,
        )
        self._items[item.id] = item
        self._save()
        return item.id

    def pop_next(self) -> ResearchItem | None:
        """Get next backlog item and mark it in_progress. Returns None if empty."""
        backlog = [i for i in self._items.values() if i.status == ResearchItemStatus.BACKLOG]
        if not backlog:
            return None
        # Use created_at for ordering (FIFO)
        item = min(backlog, key=lambda i: i.created_at)
        item.status = ResearchItemStatus.IN_PROGRESS
        item.updated_at = datetime.now(timezone.utc)
        self._save()
        return item

    def mark_done(self, item_id: str) -> None:
        """Mark item as done. Clears checkpoint."""
        item = self._items.get(item_id)
        if item:
            item.status = ResearchItemStatus.DONE
            item.checkpoint = None
            item.updated_at = datetime.now(timezone.utc)
            self._save()

    def save_checkpoint(
        self,
        item_id: str,
        sources_gathered: list[dict],
        partial_findings: str,
        step_index: int = 0,
    ) -> None:
        """Save checkpoint to item for resume. Item must exist and be in_progress."""
        item = self._items.get(item_id)
        if item:
            item.checkpoint = {
                "item_id": item_id,
                "question": item.question,
                "sources_gathered": sources_gathered,
                "partial_findings": partial_findings,
                "step_index": step_index,
            }
            item.updated_at = datetime.now(timezone.utc)
            self._save()

    def mark_blocked(self, item_id: str) -> None:
        """Mark item as blocked."""
        item = self._items.get(item_id)
        if item:
            item.status = ResearchItemStatus.BLOCKED
            item.updated_at = datetime.now(timezone.utc)
            self._save()

    def interrupt(self, item_id: str) -> None:
        """Push in_progress item back to backlog."""
        item = self._items.get(item_id)
        if item and item.status == ResearchItemStatus.IN_PROGRESS:
            item.status = ResearchItemStatus.BACKLOG
            item.updated_at = datetime.now(timezone.utc)
            self._save()

    def promote(self, item_id: str) -> None:
        """Move blocked item to backlog."""
        item = self._items.get(item_id)
        if item and item.status == ResearchItemStatus.BLOCKED:
            item.status = ResearchItemStatus.BACKLOG
            item.updated_at = datetime.now(timezone.utc)
            self._save()

    def restart_done(
        self,
        item_id: str,
        refinement_note: str | None = None,
        *,
        clear_citations: bool = False,
    ) -> bool:
        """Move a done item back to backlog for re-run. Optionally add a refinement note
        and/or clear citations so the next run gathers fresh sources. Returns True if restarted.
        """
        item = self._items.get(item_id)
        if not item or item.status != ResearchItemStatus.DONE:
            return False
        item.status = ResearchItemStatus.BACKLOG
        item.checkpoint = None
        if clear_citations:
            item.citation_entries = []
            item.citations = []
            item.sources = []
        if refinement_note:
            item.human_notes.append(refinement_note.strip())
        item.updated_at = datetime.now(timezone.utc)
        self._save()
        return True

    def restart_all_done(
        self,
        refinement_note: str | None = None,
        *,
        clear_citations: bool = False,
    ) -> int:
        """Move all done items back to backlog. Optionally add a note and/or clear citations.
        Returns the number of items restarted.
        """
        done = [i for i in self._items.values() if i.status == ResearchItemStatus.DONE]
        for item in done:
            item.status = ResearchItemStatus.BACKLOG
            item.checkpoint = None
            if clear_citations:
                item.citation_entries = []
                item.citations = []
                item.sources = []
            if refinement_note:
                item.human_notes.append(refinement_note.strip())
            item.updated_at = datetime.now(timezone.utc)
        if done:
            self._save()
        return len(done)

    def add_human_note(self, item_id: str, note: str) -> bool:
        """Append a human note to an item (in_progress or backlog). Returns True if added."""
        item = self._items.get(item_id)
        if item:
            item.human_notes.append(note.strip())
            item.updated_at = datetime.now(timezone.utc)
            self._save()
            return True
        return False

    def get_in_progress_item(self) -> ResearchItem | None:
        """Return the current in-progress item, if any."""
        in_progress = [i for i in self._items.values() if i.status == ResearchItemStatus.IN_PROGRESS]
        return in_progress[0] if in_progress else None

    def add_citation_to_item(
        self, item_id: str, url: str, title: str, snippet: str
    ) -> None:
        """Append a citation to a research item and persist."""
        item = self._items.get(item_id)
        if item:
            item.citation_entries.append(
                Citation(url=url, title=title, snippet=snippet)
            )
            item.updated_at = datetime.now(timezone.utc)
            self._save()

    def list_all(self) -> dict[str, list[ResearchItem]]:
        """Group items by status for Kanban view."""
        result: dict[str, list[ResearchItem]] = {
            "backlog": [],
            "in_progress": [],
            "blocked": [],
            "done": [],
        }
        for item in self._items.values():
            key = item.status.value
            if key in result:
                result[key].append(item)
        for key in result:
            result[key].sort(key=lambda i: i.created_at)
        return result
