"""Discovery queue data models."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum


class DiscoveryItemStatus(str, Enum):
    """Status for discovery items."""

    BACKLOG = "backlog"
    IN_PROGRESS = "in_progress"
    DONE = "done"


@dataclass
class DiscoveryItem:
    """A seed (URL or topic) to explore for breadth-first discovery."""

    id: str
    seed: str  # URL or topic to explore
    status: DiscoveryItemStatus = DiscoveryItemStatus.BACKLOG
    discovered_urls: list[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @staticmethod
    def new_id() -> str:
        return uuid.uuid4().hex[:8]

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "seed": self.seed,
            "status": self.status.value,
            "discovered_urls": self.discovered_urls,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> DiscoveryItem:
        created = datetime.fromisoformat(data.get("created_at", ""))
        updated = datetime.fromisoformat(data.get("updated_at", ""))
        return cls(
            id=data["id"],
            seed=data["seed"],
            status=DiscoveryItemStatus(data.get("status", "backlog")),
            discovered_urls=data.get("discovered_urls", []),
            created_at=created,
            updated_at=updated,
        )
