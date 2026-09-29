"""Zettelkasten note data model with markdown serialization."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone


def _make_id(agent_id: str, content: str) -> str:
    """Generate a short deterministic note ID: YYYYMMDD + agent prefix + content hash."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%d")
    h = hashlib.sha256(content.encode()).hexdigest()[:6]
    return f"{ts}-{agent_id[:4]}-{h}"


@dataclass
class ZettelNote:
    """A single Zettelkasten note (atomic knowledge unit)."""

    id: str
    title: str
    content: str
    tags: list[str] = field(default_factory=list)
    links: list[str] = field(default_factory=list)  # IDs of linked notes
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    source: str = ""
    agent_id: str = ""

    @staticmethod
    def new_id(agent_id: str, content: str) -> str:
        return _make_id(agent_id, content)

    def to_markdown(self) -> str:
        """Serialize to markdown with YAML frontmatter."""
        tags_str = ", ".join(self.tags)
        links_lines = "\n".join(f"- [[{lid}]]" for lid in self.links)
        return (
            f"---\n"
            f"id: {self.id}\n"
            f"title: {self.title}\n"
            f"tags: [{tags_str}]\n"
            f"created: {self.created_at.isoformat()}\n"
            f"updated: {self.updated_at.isoformat()}\n"
            f"source: {self.source}\n"
            f"agent: {self.agent_id}\n"
            f"---\n\n"
            f"{self.content}\n\n"
            f"## Links\n{links_lines}\n"
        )

    @classmethod
    def from_markdown(cls, text: str) -> ZettelNote:
        """Parse a markdown file back into a ZettelNote."""
        # Split frontmatter and body
        fm_match = re.match(r"^---\n(.*?)\n---\n(.*)$", text, re.DOTALL)
        if not fm_match:
            raise ValueError("Invalid note format: no YAML frontmatter found")

        fm_text = fm_match.group(1)
        body = fm_match.group(2).strip()

        # Parse frontmatter key: value pairs
        fm: dict[str, str] = {}
        for line in fm_text.strip().splitlines():
            if ":" in line:
                key, _, val = line.partition(":")
                fm[key.strip()] = val.strip()

        # Parse tags from [tag1, tag2] format
        tags_raw = fm.get("tags", "[]")
        tags_inner = tags_raw.strip("[]")
        tags = [t.strip() for t in tags_inner.split(",") if t.strip()]

        # Parse links from body: - [[id]]
        links: list[str] = re.findall(r"\[\[([^\]]+)\]\]", body)

        # Strip the ## Links section from content
        content = re.sub(r"\n## Links\n.*$", "", body, flags=re.DOTALL).strip()

        # Parse timestamps
        created = _parse_dt(fm.get("created", ""))
        updated = _parse_dt(fm.get("updated", ""))

        return cls(
            id=fm.get("id", "unknown"),
            title=fm.get("title", "Untitled"),
            content=content,
            tags=tags,
            links=links,
            created_at=created,
            updated_at=updated,
            source=fm.get("source", ""),
            agent_id=fm.get("agent", ""),
        )


def _parse_dt(s: str) -> datetime:
    if not s:
        return datetime.now(timezone.utc)
    try:
        return datetime.fromisoformat(s)
    except ValueError:
        return datetime.now(timezone.utc)
