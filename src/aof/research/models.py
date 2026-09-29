"""Research queue data models."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum


@dataclass
class Citation:
    """Structured citation for a research source."""

    url: str
    title: str
    snippet: str

    def to_dict(self) -> dict:
        return {"url": self.url, "title": self.title, "snippet": self.snippet}

    @classmethod
    def from_dict(cls, data: dict) -> Citation:
        return cls(
            url=data.get("url", ""),
            title=data.get("title", ""),
            snippet=data.get("snippet", ""),
        )


class ResearchItemStatus(str, Enum):
    """Kanban column states for research items."""

    BACKLOG = "backlog"
    IN_PROGRESS = "in_progress"
    BLOCKED = "blocked"
    DONE = "done"


@dataclass
class ResearchItem:
    """A single research question in the queue."""

    id: str
    question: str
    status: ResearchItemStatus = ResearchItemStatus.BACKLOG
    sources: list[str] = field(default_factory=list)
    citations: list[str] = field(default_factory=list)  # Legacy: formatted strings
    citation_entries: list[Citation] = field(default_factory=list)  # Structured
    human_notes: list[str] = field(default_factory=list)  # REPL-injected guidance
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    checkpoint: dict | None = None  # For future Phase 2 resume

    @staticmethod
    def new_id() -> str:
        return uuid.uuid4().hex[:8]

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "question": self.question,
            "status": self.status.value,
            "sources": self.sources,
            "citations": self.citations,
            "citation_entries": [c.to_dict() for c in self.citation_entries],
            "human_notes": self.human_notes,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "checkpoint": self.checkpoint,
        }

    @classmethod
    def from_dict(cls, data: dict) -> ResearchItem:
        created = datetime.fromisoformat(data.get("created_at", ""))
        updated = datetime.fromisoformat(data.get("updated_at", ""))
        citation_entries = [
            Citation.from_dict(ce) for ce in data.get("citation_entries", [])
        ]
        return cls(
            id=data["id"],
            question=data["question"],
            status=ResearchItemStatus(data.get("status", "backlog")),
            sources=data.get("sources", []),
            citations=data.get("citations", []),
            citation_entries=citation_entries,
            human_notes=data.get("human_notes", []),
            created_at=created,
            updated_at=updated,
            checkpoint=data.get("checkpoint"),
        )
