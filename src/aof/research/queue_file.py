"""Load a curated question set (with optional assessment rubric) from a TOML file.

Format (see examples/queues/):

    [meta]
    name = "..."
    description = "..."

    [[questions]]
    id = "short-slug"
    question = "..."
    key_facts = ["a good answer states this", ...]   # optional rubric
    pitfalls  = ["common error to avoid", ...]        # optional rubric
    refs      = ["Author Year", ...]                  # optional hints for assessors

Only `question` is used to drive research; the rest is for assessing results and is never fed to agents.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from aof.research.queue import ResearchQueue


@dataclass(frozen=True)
class QuestionSpec:
    id: str
    question: str
    key_facts: tuple[str, ...] = ()
    pitfalls: tuple[str, ...] = ()
    refs: tuple[str, ...] = ()


@dataclass(frozen=True)
class QuestionSet:
    name: str
    description: str
    questions: tuple[QuestionSpec, ...] = field(default_factory=tuple)


def load_question_set(path: str | Path) -> QuestionSet:
    with open(path, "rb") as f:
        data = tomllib.load(f)
    meta = data.get("meta", {})
    specs: list[QuestionSpec] = []
    seen: set[str] = set()
    for i, raw in enumerate(data.get("questions", [])):
        question = str(raw.get("question", "")).strip()
        if not question:
            raise ValueError(f"{path}: questions[{i}] has no question text")
        qid = str(raw.get("id") or f"q{i + 1}")
        if qid in seen:
            raise ValueError(f"{path}: duplicate question id {qid!r}")
        seen.add(qid)
        specs.append(
            QuestionSpec(
                id=qid,
                question=question,
                key_facts=tuple(raw.get("key_facts", ())),
                pitfalls=tuple(raw.get("pitfalls", ())),
                refs=tuple(raw.get("refs", ())),
            )
        )
    return QuestionSet(name=meta.get("name", Path(path).stem), description=meta.get("description", ""), questions=tuple(specs))


def enqueue_question_set(queue: ResearchQueue, qset: QuestionSet) -> int:
    """Add questions not already in the queue (any status). Returns the number added."""
    existing = {
        item.question.strip().lower()
        for items in queue.list_all().values()
        for item in items
    }
    added = 0
    for spec in qset.questions:
        key = spec.question.strip().lower()
        if key in existing:
            continue
        queue.add(spec.question)
        existing.add(key)
        added += 1
    return added
