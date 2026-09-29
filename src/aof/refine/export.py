"""Export a vault as plain markdown that Obsidian (or any wikilink-aware tool) opens as-is.

Layout:  Index.md, hubs/, claims/, Sources.md, and (optionally) archive/ with every merged or curated-out
claim, so an export is lossless. Wikilinks are `[[id|readable title]]`.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Sequence

from aof.memory.store import MemoryStore
from aof.memory.zettel import ZettelNote


def _yaml(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _frontmatter(n: ZettelNote) -> str:
    lines = ["---", f"id: {n.id}", f"title: {_yaml(n.title)}", f"kind: {n.kind}", f"status: {n.status}"]
    if n.confidence is not None:
        lines.append(f"confidence: {n.confidence}")
    lines.append("tags: [" + ", ".join(_yaml(t) for t in n.tags) + "]")
    if n.sources:
        lines.append("sources:")
        lines += [f"  - {_yaml(u)}" for u in n.sources]
    if n.superseded_by:
        lines.append(f"superseded_by: {_yaml(n.superseded_by)}")
    lines.append("---")
    return "\n".join(lines)


def _wikilinks(text: str, titles: dict[str, str]) -> str:
    return re.sub(r"\[\[([^\]|]+)\]\]", lambda m: f"[[{m.group(1)}|{titles.get(m.group(1), m.group(1))[:80]}]]", text)


def _body(n: ZettelNote, titles: dict[str, str]) -> str:
    body = _wikilinks(n.content, titles)
    related = [i for i in n.links if i in titles and f"[[{i}" not in body]
    if related:
        body += "\n\n## Related\n" + "\n".join(f"- [[{i}|{titles[i][:80]}]]" for i in related)
    if n.supersedes:
        body += "\n\n## Merged from\n" + "\n".join(f"- [[{i}]]" for i in n.supersedes)
    return body


async def export_vault(
    store: MemoryStore,
    out_dir: str | Path,
    *,
    title: str = "Knowledge vault",
    include_archive: bool = True,
) -> dict[str, int]:
    """Write the vault under `out_dir`. Returns counts of files written per folder."""
    out = Path(out_dir)
    claims_all = await store.get_notes_by_tags(["claim"], limit=100000)
    hubs = await store.get_notes_by_tags(["hub"], limit=100000)
    live = [n for n in claims_all if n.status != "archived"]
    archived = [n for n in claims_all if n.status == "archived"]
    titles = {n.id: n.title for n in [*claims_all, *hubs]}

    def write(folder: str, notes: Sequence[ZettelNote]) -> int:
        (out / folder).mkdir(parents=True, exist_ok=True)
        for n in notes:
            (out / folder / f"{n.id}.md").write_text(
                f"{_frontmatter(n)}\n\n# {n.title}\n\n{_body(n, titles)}\n", encoding="utf-8"
            )
        return len(notes)

    counts = {"hubs": write("hubs", hubs), "claims": write("claims", live)}
    if include_archive:
        counts["archive"] = write("archive", archived)

    sources: dict[str, list[str]] = {}
    for n in live:
        for u in n.sources:
            sources.setdefault(u, []).append(n.id)
    lines = [f"# {title}", "", f"{len(live)} claims, {len(hubs)} hubs, {len(sources)} sources.", "", "## Topics", ""]
    lines += [f"- [[{h.id}|{h.title}]] ({len(h.links)})" for h in sorted(hubs, key=lambda h: -len(h.links))]
    hub_members = {i for h in hubs for i in h.links}
    loose = [n for n in live if n.id not in hub_members]
    if loose:
        lines += ["", "## Unsorted claims", ""]
        lines += [f"- [[{n.id}|{n.title[:80]}]]" for n in loose]
    (out / "Index.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    src_lines = ["# Sources", ""]
    for url, ids in sorted(sources.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        src_lines.append(f"- {url} ({len(ids)} claims)")
    (out / "Sources.md").write_text("\n".join(src_lines) + "\n", encoding="utf-8")
    counts["sources"] = len(sources)
    return counts
