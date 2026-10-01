"""Export a vault as plain markdown that Obsidian (or any wikilink-aware tool) opens as-is.

Layout: `Index.md`, `reports/`, `hubs/`, `concepts/`, `sources/`, `claims/`, `Sources.md`, and (optionally) `archive/`
with every merged or curated-out claim, so an export is lossless.

File names are readable titles (Obsidian labels graph nodes by file name, so `claim-000c1f62` everywhere makes the
graph unreadable); the note id stays in the frontmatter. Wikilinks are `[[file name|readable title]]`. Every export
first removes the markdown it wrote last time, so notes that no longer exist do not linger as orphans. Archived claims
link to the note they were merged into (or to their source), so the archive is not a cloud of isolated nodes.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Sequence

from aof.memory.store import MemoryStore
from aof.memory.zettel import ZettelNote
from aof.refine.verify import url_work_key

logger = logging.getLogger(__name__)

FOLDERS = {"report": "reports", "hub": "hubs", "concept": "concepts", "source": "sources", "claim": "claims"}
ARCHIVE = "archive"
_BAD_CHARS = re.compile(r'[\\/:*?"<>|#^\[\]\n\r\t]+')

# Obsidian graph defaults: colour by layer, hide the archive (still in the vault, just not in the graph view).
_GRAPH_JSON = {
    "search": f"-path:{ARCHIVE}",
    "showTags": False,
    "showAttachments": False,
    "hideUnresolved": True,
    "showOrphans": True,
    "showArrow": False,
    "colorGroups": [
        {"query": "path:reports", "color": {"a": 1, "rgb": 15158332}},
        {"query": "path:hubs", "color": {"a": 1, "rgb": 3447003}},
        {"query": "path:concepts", "color": {"a": 1, "rgb": 15844367}},
        {"query": "path:sources", "color": {"a": 1, "rgb": 10181046}},
    ],
    "nodeSizeMultiplier": 1.2,
    "lineSizeMultiplier": 0.6,
    "textFadeMultiplier": -1,
}


def _yaml(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _link(name: str, shown: str) -> str:
    return f"[[{name}]]" if name == shown else f"[[{name}|{shown}]]"


def _short(title: str, limit: int = 60) -> str:
    """Title cut at a word boundary (no mid-word truncation)."""
    return title if len(title) <= limit else title[:limit].rsplit(" ", 1)[0] + "…"


def file_names(notes: Sequence[ZettelNote]) -> dict[str, str]:
    """{note id: unique, filesystem-safe, readable file stem}."""
    used: set[str] = set()
    out: dict[str, str] = {}
    for n in notes:
        limit = 70 if n.kind == "claim" else 90
        base = _BAD_CHARS.sub(" ", n.title or n.id)
        base = re.sub(r"\s+", " ", base).strip(" .") or n.id
        base = _short(base, limit).rstrip("…").strip(" .")
        if n.kind == "claim" or n.status == "archived":
            base = f"{base} ({n.id[-6:]})"  # sentence titles collide easily; a short id keeps them apart
        stem, k = base, 2
        while stem.lower() in used:
            stem, k = f"{base} {k}", k + 1
        used.add(stem.lower())
        out[n.id] = stem
    return out


def _frontmatter(n: ZettelNote) -> str:
    lines = ["---", f"id: {n.id}", f"title: {_yaml(n.title)}", f"kind: {n.kind}", f"status: {n.status}"]
    lines.append(f"aliases: [{_yaml(n.id)}]")
    if n.confidence is not None:
        lines.append(f"confidence: {n.confidence}")
    lines.append("tags: [" + ", ".join(_yaml(t) for t in n.tags if ":" not in t or t.startswith(("grade:", "entity:"))) + "]")
    if n.sources:
        lines.append("sources:")
        lines += [f"  - {_yaml(u)}" for u in n.sources[:40]]
    if n.superseded_by:
        lines.append(f"superseded_by: {_yaml(n.superseded_by)}")
    lines.append("---")
    return "\n".join(lines)


def _wikilinks(
    text: str, names: dict[str, str], titles: dict[str, str], kinds: dict[str, str], alias: str | None = None,
) -> str:
    """`[[id]]` -> `[[file name|readable title]]`; links to claims get the fixed `alias` when given (for lists that
    already print the claim text next to the link)."""

    def sub(m: re.Match) -> str:
        target, _, given = m.group(1).partition("|")
        if target not in names:
            return given or target  # not exported (e.g. archived): plain text, no dangling node
        shown = given or (alias if alias and kinds.get(target) == "claim" else "") or _short(titles.get(target, target))
        return _link(names[target], shown)

    return re.sub(r"\[\[([^\]]+)\]\]", sub, text)


def _body(n: ZettelNote, names: dict[str, str], titles: dict[str, str], kinds: dict[str, str]) -> str:
    listing = n.kind in ("hub", "concept", "source")
    body = _wikilinks(n.content, names, titles, kinds, alias="→" if listing else None)
    if n.kind in ("claim", "report"):
        shown = set(re.findall(r"\[\[([^\]|]+)\|", body))
        typed: dict[str, list[str]] = {"hub": [], "concept": [], "source": [], "claim": [], "report": []}
        for i in n.links:
            if i in names and names[i] not in shown and kinds.get(i) in typed:
                typed[kinds[i]].append(i)
        heads = {"hub": "Topic", "concept": "Concepts", "source": "Source", "claim": "Related claims", "report": "Reports"}
        extra = []
        for kind in ("hub", "concept", "source", "report"):
            if typed[kind]:
                extra.append(f"**{heads[kind]}:** " + ", ".join(f"[[{names[i]}|{_short(titles[i], 50)}]]" for i in typed[kind]))
        if typed["claim"]:
            extra.append(f"**{heads['claim']}:**\n" + "\n".join(f"- [[{names[i]}|{_short(titles[i], 90)}]]" for i in typed["claim"]))
        if extra:
            body += "\n\n---\n" + "\n\n".join(extra)
    if n.supersedes:
        merged = [i for i in n.supersedes if i in names]
        if merged:
            body += "\n\n**Merged from:** " + ", ".join(f"[[{names[i]}|{i}]]" for i in merged)
    return body


def _claim_ids(hub: ZettelNote, by_id: dict[str, ZettelNote]) -> set[str]:
    """Claims reachable from a hub through its subtopic hubs."""
    out: set[str] = set()
    for i in hub.links:
        if i in by_id:
            out |= _claim_ids(by_id[i], by_id)
        else:
            out.add(i)
    return out


def _clean(out: Path) -> None:
    """Remove markdown this exporter wrote before (managed folders and index files only)."""
    for folder in [*FOLDERS.values(), ARCHIVE]:
        d = out / folder
        if d.is_dir():
            for f in d.glob("*.md"):
                try:
                    f.unlink()
                except PermissionError:  # open in another program (Windows locks): it is overwritten or left stale
                    logger.warning("export: could not remove %s (in use)", f)
    for name in ("Index.md", "Sources.md"):
        (out / name).unlink(missing_ok=True)


async def export_vault(
    store: MemoryStore,
    out_dir: str | Path,
    *,
    title: str = "Knowledge vault",
    include_archive: bool = True,
    obsidian_config: bool = True,
) -> dict[str, int]:
    """Write the vault under `out_dir`. Returns counts of files written per folder."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    _clean(out)
    claims_all = await store.get_notes_by_tags(["claim"], limit=100000)
    derived = await store.get_notes_by_tags(["hub", "concept", "source", "report"], limit=100000, exclude_tags=["archived"])
    derived = [n for n in derived if n.status != "archived" and n.kind in FOLDERS]
    live = [n for n in claims_all if n.status != "archived"]
    archived = [n for n in claims_all if n.status == "archived"] if include_archive else []
    exported = [*derived, *live, *archived]
    names = file_names(exported)
    titles = {n.id: n.title for n in exported}
    kinds = {n.id: n.kind for n in exported}
    source_note = {}
    for n in derived:
        if n.kind == "source" and n.sources:
            source_note[url_work_key(n.sources[0])] = n.id

    counts: dict[str, int] = {}

    def write(folder: str, n: ZettelNote, body: str) -> None:
        (out / folder).mkdir(parents=True, exist_ok=True)
        (out / folder / f"{names[n.id]}.md").write_text(
            f"{_frontmatter(n)}\n\n# {n.title}\n\n{body}\n", encoding="utf-8"
        )
        counts[folder] = counts.get(folder, 0) + 1

    for n in [*derived, *live]:
        write(FOLDERS[n.kind], n, _body(n, names, titles, kinds))
    for n in archived:
        body = _body(n, names, titles, kinds)
        target = n.superseded_by if n.superseded_by in names and kinds.get(n.superseded_by) == "claim" else ""
        if target:
            body += f"\n\n**Merged into:** [[{names[target]}|{_short(titles[target], 80)}]]"
        else:
            src = next((source_note[url_work_key(u)] for u in n.sources if url_work_key(u) in source_note), "")
            reason = "curated out (off-topic for its question)" if "off-topic" in n.tags else "archived"
            body += f"\n\n**Status:** {reason}." + (f" **Source:** [[{names[src]}|{_short(titles[src], 80)}]]" if src else "")
        write(ARCHIVE, n, body)

    hubs = [n for n in derived if n.kind == "hub"]
    concepts = sorted((n for n in derived if n.kind == "concept"), key=lambda n: -len(n.links))
    reports = [n for n in derived if n.kind == "report"]
    sources = [n for n in derived if n.kind == "source"]
    lines = [f"# {title}", "", f"{len(live)} claims, {len(hubs)} topic hubs, {len(concepts)} concepts, "
             f"{len(sources)} sources, {len(reports)} reports.", ""]
    if reports:
        lines += ["## Reports", ""]
        lines += [f"- {_link(names[r.id], r.title)}" for r in sorted(reports, key=lambda r: r.title)]
        lines.append("")
    lines += ["## Topics", ""]
    by_id = {h.id: h for h in hubs}
    size = {h.id: len(_claim_ids(h, by_id)) for h in hubs}
    top = [h for h in hubs if "hub-level:0" in h.tags or not any(t.startswith("hub-level:") for t in h.tags)]

    def tree(h: ZettelNote, depth: int) -> None:
        lines.append("    " * depth + f"- {_link(names[h.id], h.title)} ({size[h.id]} claims)")
        for c in sorted((by_id[i] for i in h.links if i in by_id), key=lambda c: -size[c.id]):
            tree(c, depth + 1)

    for h in sorted(top, key=lambda h: -size[h.id]):
        tree(h, 0)
    if concepts:
        lines += ["", "## Concepts", ""]
        lines += [", ".join(_link(names[c.id], c.title) for c in sorted(concepts[:300], key=lambda c: c.title.lower()))]
    hub_members = {i for h in hubs for i in h.links}
    loose = [n for n in live if n.id not in hub_members]
    if loose:
        lines += ["", "## Unsorted claims", ""]
        lines += [f"- [[{names[n.id]}|{_short(n.title, 80)}]]" for n in loose]
    (out / "Index.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    src_lines = ["# Sources", "", f"{len(sources)} works.", ""]
    for s in sorted(sources, key=lambda s: (-len(s.links), s.title.lower())):
        src_lines.append(f"- [[{names[s.id]}|{_short(s.title, 110)}]] ({len(s.links)} claims)")
    (out / "Sources.md").write_text("\n".join(src_lines) + "\n", encoding="utf-8")
    counts["sources_total"] = len(sources)

    if obsidian_config:
        cfg = out / ".obsidian" / "graph.json"
        if not cfg.exists():  # never overwrite the user's own graph settings
            cfg.parent.mkdir(parents=True, exist_ok=True)
            cfg.write_text(json.dumps(_GRAPH_JSON, indent=2), encoding="utf-8")
    return counts
