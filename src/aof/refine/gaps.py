"""Gap-driven research: find what a vault does not yet answer, and research exactly that.

1. **Decompose** a question into the specific facts a complete answer must cover (model-written; never the rubric).
2. **Grade** each part against the closest claims already in the *whole* vault (`answered | partial | unanswered`).
3. **Fill**: for each uncovered part, run the normal evidence pipeline with the part as its own question, then re-grade.

Retrieval is per part, so a landmark paper that one broad query never surfaces can still be found by a focused one.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from aof.memory.store import MemoryStore
from aof.memory.zettel import ZettelNote
from aof.refine.assess import live_claims
from aof.refine.claims import Embedder
from aof.refine.verify import claim_text
from aof.refine.vectors import cosine_to
from aof.specialists import CLASSIFY, GENERATE, SpecialistExhausted, SpecialistRegistry

logger = logging.getLogger(__name__)

GRADES = ("answered", "partial", "unanswered")

_DECOMPOSE_SYSTEM = (
    "You break a research question into the specific facts a complete answer must cover. "
    "Write short, self-contained questions (each answerable in one or two sentences from a scientific paper), "
    "one per line. Name the subject explicitly in every question. No numbering, no commentary."
)


@dataclass
class SubQuestion:
    text: str
    grade: str = "unanswered"
    evidence_ids: list[str] = field(default_factory=list)  # closest claims when graded
    filled: bool = False  # research was run for it


@dataclass
class GapReport:
    question: str
    parts: list[SubQuestion] = field(default_factory=list)
    rounds: int = 0

    def counts(self) -> dict[str, int]:
        out = {g: sum(1 for p in self.parts if p.grade == g) for g in GRADES}
        out["researched"] = sum(1 for p in self.parts if p.filled)
        return out


class VaultIndex:
    """Embeddings of every live claim in the vault, refreshed incrementally as research adds claims."""

    def __init__(self, store: MemoryStore, embed: Embedder) -> None:
        self._store, self._embed = store, embed
        self._vectors: dict[str, list[float]] = {}
        self.notes: list[ZettelNote] = []
        self._matrix: list[list[float]] = []

    async def refresh(self) -> None:
        self.notes = [n for n in await live_claims(self._store) if n.kind == "claim"]
        missing = [n for n in self.notes if n.id not in self._vectors]
        if missing:
            self._vectors.update(zip((n.id for n in missing), await self._embed([claim_text(n) for n in missing])))
        self._matrix = [self._vectors[n.id] for n in self.notes]

    async def nearest(self, text: str, k: int = 5) -> list[ZettelNote]:
        if not self.notes:
            return []
        query = (await self._embed([text]))[0]
        order = np.argsort(-cosine_to(query, self._matrix), kind="stable")[:k]
        return [self.notes[int(i)] for i in order]


def parse_parts(text: str, question: str, limit: int) -> list[str]:
    """Clean model output into distinct sub-questions (drops numbering, duplicates and the question itself)."""
    parts: list[str] = []
    seen = {question.strip().lower()}
    for line in text.splitlines():
        line = re.sub(r"^[\s\-\*\d.)]+", "", line).strip().strip('"')
        if not (15 <= len(line) <= 220):
            continue
        if not line.endswith("?"):
            line += "?"
        key = line.lower()
        if key in seen:
            continue
        seen.add(key)
        parts.append(line)
        if len(parts) >= limit:
            break
    return parts


async def decompose(question: str, registry: SpecialistRegistry, limit: int = 6) -> list[str]:
    try:
        text = (await registry.call(
            GENERATE, _DECOMPOSE_SYSTEM, f"Question: {question}\n\nWrite up to {limit} questions.", max_tokens=400,
        )).value
    except SpecialistExhausted:
        return []
    return parse_parts(text, question, limit)


async def grade_part(
    part: str, evidence: Sequence[ZettelNote], registry: SpecialistRegistry, only: str | None = None,
) -> str:
    """How well `evidence` answers `part`. Falls back to `unanswered` when no model can judge."""
    if not evidence:
        return "unanswered"
    listing = "\n".join(f"- {claim_text(n)[:240]}" for n in evidence)
    description = (
        f"Question: {part}\nEvidence sentences:\n{listing}\n"
        "answered = the evidence directly answers the question; partial = it addresses the subject but leaves "
        "part of the question open; unanswered = the evidence does not answer it."
    )
    try:
        return str((await registry.call(CLASSIFY, part, list(GRADES), description=description, only=only)).value)
    except SpecialistExhausted:
        return "unanswered"


async def find_gaps(
    question: str,
    index: VaultIndex,
    registry: SpecialistRegistry,
    *,
    max_parts: int = 6,
    judge_only: str | None = None,
) -> GapReport:
    report = GapReport(question)
    for text in await decompose(question, registry, max_parts):
        part = SubQuestion(text)
        evidence = await index.nearest(text, 5)
        part.evidence_ids = [n.id for n in evidence]
        part.grade = await grade_part(text, evidence, registry, judge_only)
        report.parts.append(part)
    return report


async def fill_gaps(
    report: GapReport,
    index: VaultIndex,
    registry: SpecialistRegistry,
    research,
    *,
    rounds: int = 2,
    max_gaps: int = 4,
    judge_only: str | None = None,
) -> GapReport:
    """Research uncovered parts and re-grade, up to `rounds` times.

    `research(sub_question, parent_question)` runs the evidence pipeline for one part (the caller owns sources and
    options). A part is researched at most once per call; parts still uncovered afterwards stay visible in the report.
    """
    for _ in range(rounds):
        todo = [p for p in report.parts if p.grade != "answered" and not p.filled][:max_gaps]
        if not todo:
            break
        report.rounds += 1
        for part in todo:
            await research(part.text, report.question)
            part.filled = True
        await index.refresh()
        for part in todo:
            evidence = await index.nearest(part.text, 5)
            part.evidence_ids = [n.id for n in evidence]
            part.grade = await grade_part(part.text, evidence, registry, judge_only)
    return report


def render_markdown(reports: Sequence[GapReport], title: str = "Gap-driven research") -> str:
    lines = [f"# {title}", ""]
    for r in reports:
        c = r.counts()
        lines += [f"## {r.question}", "", f"answered {c['answered']}, partial {c['partial']}, unanswered {c['unanswered']}; "
                  f"researched {c['researched']} in {r.rounds} round(s)", ""]
        lines += [f"- [{p.grade}{' *' if p.filled else ''}] {p.text}" for p in r.parts]
        lines.append("")
    return "\n".join(lines)
