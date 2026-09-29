"""Zettelkasten note data model with markdown serialization."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone

NOTE_KINDS = ("note", "claim", "hub", "report")
NOTE_STATUSES = ("raw", "refined", "canonical", "archived")


def _make_id(agent_id: str, content: str) -> str:
    """Generate a short deterministic note ID: YYYYMMDD + agent prefix + content hash."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%d")
    h = hashlib.sha256(content.encode()).hexdigest()[:6]
    return f"{ts}-{agent_id[:4]}-{h}"


@dataclass
class ZettelNote:
    """A single Zettelkasten note (atomic knowledge unit).

    The refinement fields (`kind`, `status`, `sources`, `supersedes`, `superseded_by`, `confidence`) default
    to values that leave pre-existing notes valid. Refinement never deletes: merged originals are kept with
    status `archived`, `superseded_by` pointing at the canonical note, and listed in its `supersedes`.
    """

    id: str
    title: str
    content: str
    tags: list[str] = field(default_factory=list)
    links: list[str] = field(default_factory=list)  # IDs of linked notes
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    source: str = ""
    agent_id: str = ""
    kind: str = "note"  # one of NOTE_KINDS
    status: str = "raw"  # one of NOTE_STATUSES
    sources: list[str] = field(default_factory=list)  # URLs / citation ids the note's claims trace to
    supersedes: list[str] = field(default_factory=list)  # ids of notes merged into this one
    superseded_by: str = ""
    confidence: float | None = None

    @staticmethod
    def new_id(agent_id: str, content: str) -> str:
        return _make_id(agent_id, content)

    def to_markdown(self) -> str:
        """Serialize to markdown with YAML frontmatter."""
        tags_str = ", ".join(self.tags)
        links_lines = "\n".join(f"- [[{lid}]]" for lid in self.links)
        extra = ""
        if self.kind != "note":
            extra += f"kind: {self.kind}\n"
        if self.status != "raw":
            extra += f"status: {self.status}\n"
        if self.confidence is not None:
            extra += f"confidence: {self.confidence}\n"
        if self.superseded_by:
            extra += f"superseded_by: {self.superseded_by}\n"
        sections = ""
        if self.sources:
            sections += "## Provenance\n" + "\n".join(f"- {u}" for u in self.sources) + "\n\n"
        if self.supersedes:
            sections += "## Merged from\n" + "\n".join(f"- {i}" for i in self.supersedes) + "\n\n"
        return (
            f"---\n"
            f"id: {self.id}\n"
            f"title: {self.title}\n"
            f"tags: [{tags_str}]\n"
            f"created: {self.created_at.isoformat()}\n"
            f"updated: {self.updated_at.isoformat()}\n"
            f"source: {self.source}\n"
            f"agent: {self.agent_id}\n"
            f"{extra}"
            f"---\n\n"
            f"{self.content}\n\n"
            f"{sections}"
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

        sources = _section_items(body, "Provenance")
        supersedes = _section_items(body, "Merged from")

        # Strip the trailing sections (Links, then Sources/Supersedes) from content
        content = re.sub(r"\n## Links(?:\n.*)?$", "", body, flags=re.DOTALL)  # heading survives .strip() when empty
        content = re.sub(r"\n+## (?:Provenance|Merged from)\n(?:- .*(?:\n|$))+", "", content).strip()

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
            kind=fm.get("kind", "note"),
            status=fm.get("status", "raw"),
            sources=sources,
            supersedes=supersedes,
            superseded_by=fm.get("superseded_by", ""),
            confidence=float(fm["confidence"]) if fm.get("confidence") else None,
        )


def _section_items(body: str, name: str) -> list[str]:
    """Items of a `## <name>` bullet section, or [] if absent."""
    m = re.search(rf"^## {name}\n((?:- .*(?:\n|$))+)", body, flags=re.MULTILINE)
    if not m:
        return []
    return [line[2:].strip() for line in m.group(1).splitlines() if line.startswith("- ")]


def _parse_dt(s: str) -> datetime:
    if not s:
        return datetime.now(timezone.utc)
    try:
        return datetime.fromisoformat(s)
    except ValueError:
        return datetime.now(timezone.utc)
