"""Claim extraction: every claim is a verbatim sentence from a source (so it is grounded by construction).

Small models are unreliable at *writing* faithful claims, but reliable enough at narrow jobs. So extraction is
extractive: sentences are lifted from the source text, ranked by relevance to the question, and only then
annotated (subject/topic) by specialists. The quote is never paraphrased.
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass
from typing import Awaitable, Callable, Sequence

from aof.refine.sources import SourceDoc
from aof.refine.text import extract_entities, is_paper_meta, lexical_overlap, split_sentences

logger = logging.getLogger(__name__)

Embedder = Callable[[list[str]], Awaitable[list[list[float]]]]

_SECTION_RE = re.compile(r"^(Background|Objectives?|Aims?|Introduction|Methods?|Results?|Conclusions?|Discussion)\s*:\s*", re.I)
_PROCEDURAL = ("methods", "method")


@dataclass(frozen=True)
class Claim:
    text: str  # verbatim source sentence
    url: str
    title: str
    source: str
    authority: int
    year: str = ""
    section: str = ""  # abstract section label, lowercased ("results", "conclusions", ...)
    relevance: float = 0.0
    topic: str = ""
    statement: str = ""  # standalone rewrite of `text` (verified against it); empty when `text` already stands alone
    entities: tuple[str, ...] = ()  # acronyms / identifiers found in the quote; link keys between notes

    @property
    def standalone(self) -> str:
        """The sentence to show as the claim: the verified rewrite if there is one, else the verbatim quote."""
        return self.statement or self.text


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def candidate_sentences(doc: SourceDoc, *, min_len: int = 50, max_len: int = 420) -> list[tuple[str, str]]:
    """(sentence, section) pairs worth considering as claims; procedural and non-declarative text is dropped."""
    out: list[tuple[str, str]] = []
    section = ""
    for sentence in split_sentences(doc.text):
        m = _SECTION_RE.match(sentence)
        if m:
            section = m.group(1).lower()
            sentence = sentence[m.end():].strip()
        if section in _PROCEDURAL:
            continue
        if not (min_len <= len(sentence) <= max_len):
            continue
        if sentence.endswith("?") or not sentence.endswith("."):
            continue
        if is_paper_meta(sentence):
            continue  # about the paper, not the world
        if sum(c.isalpha() for c in sentence) < len(sentence) * 0.6:
            continue  # mostly numbers/symbols (tables, references)
        out.append((sentence, section))
    return out


def claim_quota(doc: SourceDoc, base: int, max_factor: int = 4) -> int:
    """How many claims to take from a document: `base` for an abstract, more for long articles (~1 per 1200 chars).

    A fixed quota starves long, information-dense sources (encyclopedia articles, reviews) while abstracts have
    fewer than `base` usable sentences anyway.
    """
    return min(base * max_factor, max(base, len(doc.text) // 1200))


async def extract_claims(
    doc: SourceDoc,
    question: str,
    *,
    embed: Embedder | None = None,
    top_k: int = 6,
    min_relevance: float = 0.25,
) -> list[Claim]:
    """Top-k question-relevant sentences from `doc` as claims (semantic relevance, lexical fallback)."""
    candidates = candidate_sentences(doc)
    if not candidates:
        return []
    sentences = [s for s, _ in candidates]
    scores = None
    if embed is not None:
        try:
            vectors = await embed([question] + sentences)
            scores = [cosine(vectors[0], v) for v in vectors[1:]]
        except Exception as e:  # embedder unavailable: degrade to lexical relevance rather than fail the run
            logger.warning("embedding failed (%s); using lexical relevance", e)
    if scores is None:
        scores = [lexical_overlap(question, s) * 2 for s in sentences]  # Jaccard runs low; scale toward cosine
    ranked = sorted(range(len(sentences)), key=lambda i: -scores[i])[:top_k]
    year = str(doc.meta.get("year") or "")
    return [
        Claim(
            text=sentences[i], url=doc.url, title=doc.title, source=doc.source, authority=doc.authority,
            year=year, section=candidates[i][1], relevance=round(scores[i], 4),
            entities=tuple(extract_entities(sentences[i])),
        )
        for i in ranked
        if scores[i] >= min_relevance
    ]
