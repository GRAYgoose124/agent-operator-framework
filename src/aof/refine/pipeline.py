"""Evidence-first refinement for one question: plan -> gather -> extract -> dedupe -> enrich -> store.

Deterministic scaffolding drives the loop; models only do narrow, checkable jobs (embed, classify, rewrite, judge).
"""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass, field, replace
from typing import Sequence

from aof.memory.store import MemoryStore
from aof.memory.zettel import ZettelNote
from aof.refine.claims import Claim, Embedder, claim_quota, extract_claims
from aof.refine.decontext import decontextualize
from aof.refine.dedupe import cluster_claims, pick_canonical
from aof.refine.sources import Source, SourceDoc, gather_documents
from aof.refine.text import keyword_query
from aof.refine.vault import store_clusters
from aof.refine.verify import apply_verification, curate, verify_note
from aof.specialists import CLASSIFY, EMBED, GENERATE, SpecialistExhausted, SpecialistRegistry

logger = logging.getLogger(__name__)

_PLAN_SYSTEM = (
    "You write literature-search queries. Given a research question, write distinct keyword queries "
    "(3-7 words each) that would find scientific papers answering different parts of it. "
    "One query per line. No numbering, no quotes, no commentary."
)


@dataclass
class RefineReport:
    question: str
    queries: list[str] = field(default_factory=list)
    documents: list[SourceDoc] = field(default_factory=list)
    claims: list[Claim] = field(default_factory=list)
    clusters: list[list[int]] = field(default_factory=list)
    canonical: list[ZettelNote] = field(default_factory=list)
    archived: list[ZettelNote] = field(default_factory=list)
    graded: dict[str, int] = field(default_factory=dict)  # curation grade -> count
    verifications: list[tuple[str, str]] = field(default_factory=list)  # (note id, verdict)

    def summary(self) -> dict[str, int]:
        return {
            **self._counts(), **{f"grade_{k}": v for k, v in self.graded.items()},
            **{f"verified_{v}": sum(1 for _, x in self.verifications if x == v) for v in sorted({x for _, x in self.verifications})},
        }

    def _counts(self) -> dict[str, int]:
        return {
            "documents": len(self.documents), "claims": len(self.claims), "clusters": len(self.clusters),
            "canonical": len(self.canonical), "archived": len(self.archived),
            "corroborated": sum(1 for n in self.canonical if n.status == "canonical"),
            "standalone_rewrites": sum(1 for c in self.claims if c.statement),
        }


def registry_embedder(registry: SpecialistRegistry) -> Embedder | None:
    """Embedding function backed by the registry's `embed` chain, or None when no provider is available."""
    if not registry.providers_for(EMBED):
        return None

    async def embed(texts: list[str]) -> list[list[float]]:
        return (await registry.call(EMBED, texts)).value

    return embed


async def plan_queries(question: str, registry: SpecialistRegistry | None = None, *, max_queries: int = 4) -> list[str]:
    """Search queries for a question: the question, its keyword form, and (if a model is available) sub-queries."""
    queries = [question]
    keywords = keyword_query(question)
    if keywords.lower() != question.lower():
        queries.append(keywords)
    if registry is not None and registry.providers_for(GENERATE):
        try:
            text = (await registry.call(
                GENERATE, _PLAN_SYSTEM, f"Question: {question}\n\nWrite {max_queries} queries.", max_tokens=160,
            )).value
        except SpecialistExhausted:
            text = ""
        for line in text.splitlines():
            line = re.sub(r"^[\s\-\*\d.)]+", "", line).strip().strip('"')
            if 8 <= len(line) <= 100 and line.lower() not in {q.lower() for q in queries}:
                queries.append(line)
            if len(queries) >= max_queries + 2:
                break
    return queries


async def _classify_topics(
    registry: SpecialistRegistry, claims: list[Claim], indices: Sequence[int], topics: Sequence[str],
    concurrency: int = 1,
) -> list[Claim]:
    """Topic label for the given claims via the cheap-first classify chain (skipped when no topics are given)."""
    if not topics or not registry.providers_for(CLASSIFY):
        return claims
    labels = [*topics, "other"]
    out = list(claims)
    gate = asyncio.Semaphore(concurrency)

    async def one(i: int) -> None:
        async with gate:
            try:
                out[i] = replace(claims[i], topic=str((await registry.call(CLASSIFY, claims[i].text, labels)).value))
            except SpecialistExhausted:
                pass

    await asyncio.gather(*(one(i) for i in indices))
    return out


async def refine_question(
    question: str,
    *,
    store: MemoryStore,
    registry: SpecialistRegistry,
    sources: list[Source],
    topics: Sequence[str] = (),
    per_source: int = 4,
    claims_per_doc: int = 6,
    dedupe_threshold: float = 0.88,
    concurrency: int = 1,
    curate_only: str | None = None,
    curate_confirm: str | None = None,
    do_curate: bool = False,
    verify_top: int = 0,
) -> RefineReport:
    """Gather evidence for `question`, extract grounded claims, merge duplicates losslessly, store as notes."""
    report = RefineReport(question=question)
    report.queries = await plan_queries(question, registry)
    report.documents = await gather_documents(sources, report.queries, per_source)
    logger.info("refine: %d documents for %r", len(report.documents), question[:60])

    embed = registry_embedder(registry)
    for doc in report.documents:
        report.claims += await extract_claims(doc, question, embed=embed, top_k=claim_quota(doc, claims_per_doc))
    if not report.claims:
        return report

    vectors = None
    if embed:
        try:
            vectors = await embed([c.text for c in report.claims])
        except SpecialistExhausted as e:
            logger.warning("embedding unavailable (%s); clustering on token overlap instead", e)
    report.clusters = cluster_claims(report.claims, vectors, threshold=dedupe_threshold)

    # Enrich only each cluster's representative: it is the one that becomes the canonical note.
    reps = [pick_canonical(report.claims, members) for members in report.clusters]
    report.claims = await _classify_topics(registry, report.claims, reps, topics, concurrency)
    gate = asyncio.Semaphore(concurrency)

    async def standalone(i: int) -> tuple[int, Claim]:
        async with gate:
            return i, await decontextualize(report.claims[i], registry)

    for i, claim in await asyncio.gather(*(standalone(i) for i in reps)):
        report.claims[i] = claim

    report.canonical, report.archived = await store_clusters(store, question, report.claims, report.clusters)

    if do_curate:  # archive claims that do not help answer the question (kept, tagged off-topic)
        graded = await curate(
            question, report.canonical, store=store, registry=registry,
            only=curate_only, confirm_negatives_with=curate_confirm, concurrency=concurrency,
        )
        report.graded = {label: len(notes) for label, notes in graded.items()}
    if verify_top > 0:  # light sanity research on the most question-relevant surviving claims
        relevance = {n.id: report.claims[rep].relevance for n, rep in zip(report.canonical, reps)}
        live = [n for n in report.canonical if n.status != "archived"]
        for note in sorted(live, key=lambda n: -relevance[n.id])[:verify_top]:
            v = await verify_note(note, registry=registry, sources=sources, embed=embed)
            await apply_verification(store, note, v)
            report.verifications.append((note.id, v.verdict))
    return report
