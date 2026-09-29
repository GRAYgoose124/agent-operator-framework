"""Standalone-ness of claims: can a sentence be understood without the paper it came from?

The opener heuristic in `needs_context` only catches discourse markers ("In turn, ..."). Subjectless statements
("Increases happened in the ventral CA1...") slip through, so a small model grades every remaining claim, and the
verified decontextualiser rewrites those that need it. A claim that cannot be repaired is tagged, not hidden.
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, field
from typing import Sequence

from aof.memory.store import MemoryStore
from aof.memory.zettel import ZettelNote
from aof.refine.claims import Claim
from aof.refine.decontext import rewrite_verified
from aof.refine.verify import claim_text
from aof.specialists import CLASSIFY, SpecialistExhausted, SpecialistRegistry

LABELS = ("standalone", "context_dependent")
_DESCRIPTION = (
    "Can this sentence be understood on its own, by a reader who has not seen the paper it came from? "
    "standalone = it names its subject and says something complete. "
    "context_dependent = it has no explicit subject, or refers to unnamed things (this, these, the model, the "
    "animals, the task, 'increases happened in...') that only the surrounding text explains."
)
_EVIDENCE_LINE = re.compile(r'^- "(.*)" — (.*), (\S+)$')

CHECKED_TAG = "standalone-checked"
UNRESOLVED_TAG = "standalone:unresolved"
REWRITTEN_TAG = "standalone:rewritten"


@dataclass
class StandaloneReport:
    checked: int = 0
    already_fine: int = 0
    rewritten: int = 0
    unresolved: int = 0
    skipped: int = 0
    unresolved_ids: list[str] = field(default_factory=list)

    def summary(self) -> dict[str, int]:
        return {"checked": self.checked, "standalone": self.already_fine, "rewritten": self.rewritten,
                "unresolved": self.unresolved, "skipped": self.skipped}


def first_evidence(note: ZettelNote) -> tuple[str, str, str] | None:
    """(quote, cite, url) of the note's first Evidence line, or None."""
    block = note.content.split("**Evidence**", 1)
    if len(block) < 2:
        return None
    for line in block[1].splitlines():
        m = _EVIDENCE_LINE.match(line.strip())
        if m:
            return m.groups()
    return None


def is_verbatim(note: ZettelNote) -> bool:
    """True if the note's headline is still the source quote (no verified rewrite has replaced it)."""
    ev = first_evidence(note)
    return ev is not None and claim_text(note).strip() == ev[0].strip()


async def is_standalone(text: str, registry: SpecialistRegistry, only: str | None = None) -> bool | None:
    """True/False from the classifier, or None if no provider could answer."""
    try:
        label = (await registry.call(CLASSIFY, text, list(LABELS), description=_DESCRIPTION, only=only)).value
    except SpecialistExhausted:
        return None
    return label == "standalone"


async def make_standalone(
    notes: Sequence[ZettelNote],
    *,
    store: MemoryStore,
    registry: SpecialistRegistry,
    only: str | None = None,
    concurrency: int = 4,
) -> StandaloneReport:
    """Check each note's headline; rewrite context-dependent ones (verified), tag the rest. Idempotent."""
    report = StandaloneReport()
    gate = asyncio.Semaphore(concurrency)

    async def one(note: ZettelNote) -> None:
        ev = first_evidence(note)
        if CHECKED_TAG in note.tags or ev is None or not is_verbatim(note):
            report.skipped += 1
            return
        async with gate:
            verdict = await is_standalone(claim_text(note), registry, only)
            if verdict is None:
                report.skipped += 1
                return
            report.checked += 1
            if verdict:
                report.already_fine += 1
            else:
                quote, cite, url = ev
                claim = Claim(text=quote, url=url, title=re.sub(r"\s*\(\d{4}\)$", "", cite), source="", authority=0)
                fixed = await rewrite_verified(claim, registry)
                if fixed:
                    head, _, rest = note.content.partition("\n\n**Evidence**")
                    note.content = fixed + "\n\n**Evidence**" + rest
                    note.title = fixed if len(fixed) <= 90 else fixed[:89].rsplit(" ", 1)[0] + "…"
                    note.tags.append(REWRITTEN_TAG)
                    report.rewritten += 1
                else:
                    note.tags.append(UNRESOLVED_TAG)
                    report.unresolved += 1
                    report.unresolved_ids.append(note.id)
            note.tags.append(CHECKED_TAG)
        await store.update_note(note)

    await asyncio.gather(*(one(n) for n in notes))
    return report
