"""Repairs for vaults written by earlier versions of the pipeline."""

from __future__ import annotations

import re

from aof.memory.store import MemoryStore
from aof.memory.zettel import ZettelNote
from aof.refine.assess import live_claims
from aof.refine.verify import claim_text, own_quotes, same_statement

_LINE = re.compile(r'^- "(.*)" — (.*), (\S+)$')


def _split_sections(content: str) -> tuple[str, dict[str, str]]:
    """(headline, {section name: body}) for content shaped `claim\\n\\n**Evidence**\\n...\\n\\n**Corroboration**\\n...`."""
    parts = re.split(r"\n\n\*\*([^*]+)\*\*\n", content)
    return parts[0], {parts[i]: parts[i + 1] for i in range(1, len(parts) - 1, 2)}


def _rebuild(headline: str, sections: dict[str, str]) -> str:
    return headline + "".join(f"\n\n**{name}**\n{body}" for name, body in sections.items() if body.strip())


def strip_false_corroboration(note: ZettelNote) -> bool:
    """Drop corroboration lines that merely repeat a quote the note already has (same paper under another URL).

    Sources contributed only by dropped lines are removed too; a note falls back to `refined` if nothing
    independent remains. Returns True if the note changed.
    """
    if "**Corroboration**" not in note.content:
        return False
    headline, sections = _split_sections(note.content)
    have = {q for q in own_quotes(note)}
    keep, dropped_urls, kept_urls = [], [], []
    for line in sections.get("Corroboration", "").splitlines():
        m = _LINE.match(line.strip())
        if not m:
            keep.append(line)
            continue
        quote, _, url = m.groups()
        if any(same_statement(quote, q) for q in have):
            dropped_urls.append(url)
        else:
            keep.append(line)
            kept_urls.append(url)
    if not dropped_urls:
        return False
    evidence_urls = [m.group(3) for ln in sections.get("Evidence", "").splitlines() if (m := _LINE.match(ln.strip()))]
    sections["Corroboration"] = "\n".join(keep) if kept_urls else ""
    note.content = _rebuild(headline, sections)
    # A URL stays only if the evidence block cites it or a surviving corroboration line does.
    note.sources = [u for u in note.sources if u in evidence_urls or u in kept_urls or u not in dropped_urls]
    if not kept_urls and len(set(evidence_urls)) < 2 and note.status == "canonical":
        note.status = "refined"
    return True


def strip_weak_conflict(note: ZettelNote, similarity, min_sim: float = 0.65) -> bool:
    """Remove a conflict flag whose "conflicting evidence" is not actually about the claim.

    `similarity(a, b)` scores two sentences. A conflict survives only if some conflicting quote is at least
    `min_sim` similar to the claim. Returns True if the flag was removed.
    """
    if "conflict" not in note.tags or "**Conflicting evidence**" not in note.content:
        return False
    headline, sections = _split_sections(note.content)
    quotes = [m.group(1) for ln in sections.get("Conflicting evidence", "").splitlines() if (m := _LINE.match(ln.strip()))]
    if any(similarity(headline, q) >= min_sim for q in quotes):
        return False
    sections.pop("Conflicting evidence", None)
    note.content = _rebuild(headline, sections)
    note.tags = [t for t in note.tags if t != "conflict"]
    return True


async def repair_weak_conflicts(store: MemoryStore, similarity) -> tuple[int, int]:
    """Apply `strip_weak_conflict` across the vault. Returns (flagged notes examined, flags removed)."""
    examined = removed = 0
    for note in await live_claims(store):
        if "conflict" not in note.tags:
            continue
        examined += 1
        if strip_weak_conflict(note, similarity):
            await store.update_note(note)
            removed += 1
    return examined, removed


async def repair_false_corroboration(store: MemoryStore) -> tuple[int, int]:
    """Apply `strip_false_corroboration` across the vault. Returns (notes examined, notes repaired)."""
    examined = repaired = 0
    for note in await live_claims(store):
        if "**Corroboration**" not in note.content:
            continue
        examined += 1
        if strip_false_corroboration(note):
            await store.update_note(note)
            repaired += 1
    return examined, repaired
