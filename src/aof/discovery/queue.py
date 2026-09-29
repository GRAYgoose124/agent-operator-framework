"""Discovery queue with JSON persistence."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from aof.discovery.models import DiscoveryItem, DiscoveryItemStatus

logger = logging.getLogger(__name__)


class DiscoveryQueue:
    """Breadth-first discovery queue. Persists to JSON."""

    def __init__(self, queue_path: str | Path) -> None:
        self._path = Path(queue_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._items: dict[str, DiscoveryItem] = {}
        self._load()

    def _load(self) -> None:
        """Load queue from disk."""
        if not self._path.exists():
            self._items = {}
            return
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
            self._items = {
                item["id"]: DiscoveryItem.from_dict(item)
                for item in data.get("items", [])
            }
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning("Failed to load discovery queue: %s", e)
            self._items = {}

    def _save(self) -> None:
        """Persist queue to disk."""
        data = {
            "items": [item.to_dict() for item in self._items.values()],
        }
        self._path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def add(self, seed: str) -> str:
        """Add a seed (URL or topic) to backlog. Returns item ID."""
        item = DiscoveryItem(
            id=DiscoveryItem.new_id(),
            seed=seed.strip(),
            status=DiscoveryItemStatus.BACKLOG,
        )
        self._items[item.id] = item
        self._save()
        return item.id

    def pop_next(self) -> DiscoveryItem | None:
        """Get next backlog item and mark it in_progress. Returns None if empty."""
        backlog = [i for i in self._items.values() if i.status == DiscoveryItemStatus.BACKLOG]
        if not backlog:
            return None
        item = min(backlog, key=lambda i: i.created_at)
        item.status = DiscoveryItemStatus.IN_PROGRESS
        item.updated_at = datetime.now(timezone.utc)
        self._save()
        return item

    def mark_done(self, item_id: str) -> None:
        """Mark item as done."""
        item = self._items.get(item_id)
        if item:
            item.status = DiscoveryItemStatus.DONE
            item.updated_at = datetime.now(timezone.utc)
            self._save()

    def list_all(self) -> dict[str, list[DiscoveryItem]]:
        """Group items by status."""
        result: dict[str, list[DiscoveryItem]] = {
            "backlog": [],
            "in_progress": [],
            "done": [],
        }
        for item in self._items.values():
            key = item.status.value
            if key in result:
                result[key].append(item)
        for key in result:
            result[key].sort(key=lambda i: i.created_at)
        return result
