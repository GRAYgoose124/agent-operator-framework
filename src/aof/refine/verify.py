"""Verification and curation of claim notes.

`verify_note` does the light "sanity research": look the claim up in *other* works, judge it against the best
matching sentences, and record what was found (corroboration or contradiction) as cited evidence. A cheap judge
goes first; a negative verdict is escalated to the large model before anything is flagged, so a small model's
false alarm cannot mark good knowledge as disputed.

`curate` is the relevance gate: claims that do not help answer the question are archived (never deleted).
"""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass, field
from typing import Sequence

from aof.memory.store import MemoryStore
from aof.memory.zettel import ZettelNote
from aof.refine.claims import Embedder, candidate_sentences
from aof.refine.vectors import cosine_to
from aof.refine.sources import Source, SourceDoc, doc_key, gather_documents
from aof.refine.text import keyword_query, lexical_overlap, normalize
from aof.specialists import CLASSIFY, JUDGE, SpecialistExhausted, SpecialistRegistry

logger = logging.getLogger(__name__)

LARGE = "role:large"
# A contradiction needs evidence that is about the same claim; loosely related sentences (same broad area,
# different subject) are only ever "unsupported". Measured: without this floor, all 4 flags in a 150-claim run
# were false alarms (MDMA users, toad vision, the cerebellum).
CONTRADICTION_MIN_SIM = 0.65


@dataclass
class Verification:
    verdict: str  # supported | contradicted | unsupported | needs_lookup | no_evidence
    evidence: list[tuple[str, SourceDoc]] = field(default_factory=list)  # (sentence, its document)
    judged_by: str = ""
    escalated: bool = False
    rationale: str = ""


def claim_text(note: ZettelNote) -> str:
    """The claim as a sentence: the note's headline (first paragraph), without its evidence block."""
    return note.content.split("\n\n**Evidence**")[0].strip()


def same_statement(a: str, b: str) -> bool:
    """True if two sentences are the same statement (identical after normalisation, or near-identical tokens)."""
    return normalize(a) == normalize(b) or lexical_overlap(a, b) >= 0.85


def own_quotes(note: ZettelNote) -> list[str]:
    """The verbatim quotes a note already rests on (its headline claim and its Evidence block)."""
    quotes = [claim_text(note)]
    quotes += re.findall(r'^- "(.*)" \u2014 ', note.content.split("**Corroboration**")[0], flags=re.MULTILINE)
    return quotes


def url_work_key(url: str) -> str:
    """Work identity for a stored source URL (DOI when the URL is a DOI link), matching `doc_key`."""
    if "doi.org/" in url:
        return "doi:" + re.sub(r"^https?://(dx\.)?doi\.org/", "", url.lower())
    return "url:" + url


async def _best_sentences(
    claim: str, docs: Sequence[SourceDoc], embed: Embedder | None, top_k: int, min_sim: float,
    exclude: Sequence[str] = (),
) -> list[tuple[str, SourceDoc, float]]:
    """Best-matching (sentence, doc, similarity), never the claim itself or a quote the note already rests on."""
    pool = [
        (s, d) for d in docs for s, _ in candidate_sentences(d)
        if not any(same_statement(s, q) for q in exclude)
    ]
    if not pool:
        return []
    if embed is not None:
        try:
            vectors = await embed([claim] + [s for s, _ in pool])
            scores = cosine_to(vectors[0], vectors[1:]).tolist()
        except Exception as e:
            logger.warning("embedding failed during verification (%s); using lexical match", e)
            scores = [lexical_overlap(claim, s) * 2 for s, _ in pool]
    else:
        scores = [lexical_overlap(claim, s) * 2 for s, _ in pool]
    ranked = sorted(range(len(pool)), key=lambda i: -scores[i])[:top_k]
    return [(*pool[i], float(scores[i])) for i in ranked if scores[i] >= min_sim]


async def verify_note(
    note: ZettelNote,
    *,
    registry: SpecialistRegistry,
    sources: list[Source],
    embed: Embedder | None = None,
    per_source: int = 4,
    top_k: int = 3,
    min_sim: float = 0.5,
) -> Verification:
    """Look the note's claim up in works other than its own sources and judge it against what turns up."""
    claim = claim_text(note)
    own = set(note.sources)
    own_keys = {url_work_key(u) for u in own}
    docs = await gather_documents(sources, [keyword_query(claim), claim], per_source)
    # Independent evidence only: skip the works this claim already came from (same URL or same DOI) ...
    docs = [d for d in docs if d.url not in own and doc_key(d) not in own_keys]
    # ... and, since the same paper reappears under other URLs, any sentence that repeats a quote we already have.
    scored = await _best_sentences(claim, docs, embed, top_k, min_sim, exclude=own_quotes(note))
    if not scored:
        return Verification("no_evidence")
    evidence = [(s, d) for s, d, _ in scored]
    closest = max(sim for _, _, sim in scored)
    text = "\n".join(f"- {s} [{d.title[:60]}]" for s, d in evidence)
    try:
        first = await registry.call(JUDGE, claim, text)
    except SpecialistExhausted:
        return Verification("no_evidence", evidence)
    verdict = first.value["verdict"]
    if verdict == "contradicted" and closest < CONTRADICTION_MIN_SIM:
        verdict = "unsupported"  # evidence too distant to be about the same claim: not a contradiction
    result = Verification(verdict, evidence, first.provider, rationale=str(first.value.get("rationale", "")))
    if verdict == "contradicted" and first.provider != LARGE:
        try:  # confirm a negative with the strongest judge before flagging anything
            second = await registry.call(JUDGE, claim, text, only=LARGE)
        except SpecialistExhausted:
            return result
        result = Verification(
            second.value["verdict"], evidence, second.provider, escalated=True,
            rationale=str(second.value.get("rationale", "")),
        )
    return result


async def apply_verification(store: MemoryStore, note: ZettelNote, v: Verification) -> ZettelNote:
    """Record a verification on the note: corroboration adds sources/evidence; a confirmed contradiction is flagged."""
    if v.verdict == "supported" and v.evidence:
        urls = list(dict.fromkeys(d.url for _, d in v.evidence))
        note.sources = list(dict.fromkeys([*note.sources, *urls]))
        note.content += "\n\n**Corroboration**\n" + "\n".join(
            f'- "{s}" — {d.title}, {d.url}' for s, d in v.evidence
        )
        note.status = "canonical"
    elif v.verdict == "contradicted" and v.evidence:
        note.content += "\n\n**Conflicting evidence**\n" + "\n".join(
            f'- "{s}" — {d.title}, {d.url}' for s, d in v.evidence
        )
        if "conflict" not in note.tags:
            note.tags.append("conflict")
    else:
        return note
    await store.update_note(note)
    return note


CURATE_LABELS = ("core", "supporting", "irrelevant")


async def curate(
    question: str,
    notes: Sequence[ZettelNote],
    *,
    store: MemoryStore,
    registry: SpecialistRegistry,
    only: str | None = None,
    confirm_negatives_with: str | None = None,
    concurrency: int = 1,
) -> dict[str, list[ZettelNote]]:
    """Grade each claim `core | supporting | irrelevant` for `question`; archive the irrelevant ones (kept, tagged).

    Archiving hides a claim from the working vault, so a wrongly rejected claim is the costly error. When
    `confirm_negatives_with` names a stronger provider, every "irrelevant" from the first pass is re-graded by it
    and its label is final.
    """
    graded: dict[str, list[ZettelNote]] = {label: [] for label in CURATE_LABELS}
    description = (
        f"Question: {question}\nGrade how the sentence relates to the question's subject. "
        "core = directly answers part of the question; supporting = related facts about the same subject "
        "(mechanism, anatomy, evidence, context, findings); irrelevant = about a different subject entirely."
    )
    gate = asyncio.Semaphore(concurrency)

    async def grade(note: ZettelNote) -> None:
        async with gate:
            try:
                label = (await registry.call(
                    CLASSIFY, claim_text(note), list(CURATE_LABELS), description=description, only=only,
                )).value
                if label == "irrelevant" and confirm_negatives_with:
                    label = (await registry.call(
                        CLASSIFY, claim_text(note), list(CURATE_LABELS), description=description,
                        only=confirm_negatives_with,
                    )).value
            except SpecialistExhausted:
                return
        graded[label].append(note)
        tag = f"grade:{label}"
        if tag not in note.tags:
            note.tags.append(tag)
        if label == "irrelevant":
            note.status = "archived"
            if "archived" not in note.tags:
                note.tags.append("archived")
            note.tags.append("off-topic")
        await store.update_note(note)

    await asyncio.gather(*(grade(n) for n in notes))
    return graded
