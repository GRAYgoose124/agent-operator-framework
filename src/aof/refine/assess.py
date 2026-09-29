"""Assess a vault against a question set's rubric.

For each rubric `key_fact`, retrieve the closest live claim notes from the *whole* vault (a fact may be covered by
a note gathered for another question) and ask the strongest available judge whether they support it. Pitfalls are
listed with the retrieved notes for a human to check; the rubric is a guide, not ground truth.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from aof.memory.store import MemoryStore
from aof.memory.zettel import ZettelNote
from aof.refine.claims import cosine
from aof.refine.pipeline import registry_embedder
from aof.refine.verify import claim_text
from aof.research.queue_file import QuestionSpec
from aof.specialists import JUDGE, SpecialistExhausted, SpecialistRegistry


@dataclass
class FactResult:
    fact: str
    verdict: str  # supported | contradicted | unsupported | needs_lookup | no_notes
    notes: list[ZettelNote] = field(default_factory=list)  # the retrieved candidates, best first
    judged_by: str = ""

    @property
    def covered(self) -> bool:
        return self.verdict == "supported"


@dataclass
class QuestionAssessment:
    spec: QuestionSpec
    facts: list[FactResult] = field(default_factory=list)
    pitfall_notes: dict[str, list[ZettelNote]] = field(default_factory=dict)  # pitfall -> closest notes (for a human)

    @property
    def coverage(self) -> float:
        return sum(f.covered for f in self.facts) / len(self.facts) if self.facts else 0.0


async def live_claims(store: MemoryStore) -> list[ZettelNote]:
    """Every non-archived claim note in the vault."""
    notes = await store.get_notes_by_tags(["claim"], limit=100000, exclude_tags=["archived"])
    return [n for n in notes if n.status != "archived"]


async def assess_question(
    spec: QuestionSpec,
    notes: Sequence[ZettelNote],
    vectors: dict[str, list[float]],
    *,
    registry: SpecialistRegistry,
    embed,
    top_k: int = 4,
    judge_only: str | None = None,
) -> QuestionAssessment:
    """Score `spec`'s key facts against `notes` (whose embeddings are in `vectors`, keyed by note id)."""
    result = QuestionAssessment(spec)
    if not notes:
        result.facts = [FactResult(f, "no_notes") for f in spec.key_facts]
        return result

    async def nearest(text: str) -> list[ZettelNote]:
        v = (await embed([text]))[0]
        ranked = sorted(notes, key=lambda n: -cosine(v, vectors[n.id]))
        return ranked[:top_k]

    for fact in spec.key_facts:
        best = await nearest(fact)
        evidence = "\n".join(f"- {claim_text(n)}" for n in best)
        try:
            judged = await registry.call(JUDGE, fact, evidence, only=judge_only)
            result.facts.append(FactResult(fact, judged.value["verdict"], best, judged.provider))
        except SpecialistExhausted:
            result.facts.append(FactResult(fact, "needs_lookup", best))
    for pitfall in spec.pitfalls:
        result.pitfall_notes[pitfall] = await nearest(pitfall)
    return result


async def assess_vault(
    specs: Sequence[QuestionSpec],
    store: MemoryStore,
    registry: SpecialistRegistry,
    *,
    judge_only: str | None = None,
    top_k: int = 4,
) -> list[QuestionAssessment]:
    notes = await live_claims(store)
    embed = registry_embedder(registry)
    if embed is None:
        raise RuntimeError("assessment needs an embedding provider (sentence-transformers)")
    vectors: dict[str, list[float]] = {}
    if notes:
        for note, vec in zip(notes, await embed([claim_text(n) for n in notes])):
            vectors[note.id] = vec
    return [
        await assess_question(s, notes, vectors, registry=registry, embed=embed, top_k=top_k, judge_only=judge_only)
        for s in specs
    ]


def render_markdown(assessments: Sequence[QuestionAssessment], *, title: str = "Vault assessment") -> str:
    lines = [f"# {title}", ""]
    total = sum(len(a.facts) for a in assessments)
    covered = sum(f.covered for a in assessments for f in a.facts)
    if total:
        lines += [f"**Rubric coverage: {covered}/{total} key facts ({covered / total:.0%})**", ""]
    lines += ["| question | coverage |", "|---|---|"]
    lines += [f"| {a.spec.id} | {sum(f.covered for f in a.facts)}/{len(a.facts)} |" for a in assessments]
    for a in assessments:
        lines += ["", f"## {a.spec.id}", "", a.spec.question, ""]
        for f in a.facts:
            mark = "OK " if f.covered else "-- "
            lines.append(f"- [{mark}] ({f.verdict}) {f.fact}")
            for n in f.notes[:2]:
                lines.append(f"    - {claim_text(n)[:200]}")
        if a.pitfall_notes:
            lines += ["", "Pitfalls to check by hand (closest notes):"]
            for pitfall, notes in a.pitfall_notes.items():
                lines.append(f"- *{pitfall}*")
                lines += [f"    - {claim_text(n)[:200]}" for n in notes[:2]]
    return "\n".join(lines) + "\n"
