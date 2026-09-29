"""Deterministic text utilities (tier 0): sentence splitting, keyword queries, normalisation."""

from __future__ import annotations

import re

_ABBREV = (
    "et al", "e.g", "i.e", "fig", "figs", "vs", "cf", "ca", "approx", "dr", "prof", "mr", "mrs", "ms",
    "no", "vol", "eq", "ref", "refs", "sp", "spp", "st", "inc", "ltd", "co", "resp", "viz",
)
_PLACEHOLDER = "\x00"
_ABBREV_RE = re.compile(r"\b(" + "|".join(re.escape(a) for a in _ABBREV) + r")\.", re.IGNORECASE)
_DECIMAL_RE = re.compile(r"(\d)\.(\d)")
_INITIAL_RE = re.compile(r"\b([A-Z])\.(?=\s+[A-Z])")
_SPLIT_RE = re.compile(r"(?<=[.!?])[\"')\]]*\s+(?=[\"'(\[]?[A-Z0-9])")

STOPWORDS = frozenset("""
a an the and or of in on at to for from by with without as is are was were be been being this that these those
it its what which who whom how why when where does do did can could should would may might will shall
about into over under between among within through during after before their there than then so such
also both each other some any most more less many much very not no nor only own same too
""".split())


def split_sentences(text: str) -> list[str]:
    """Split prose into sentences without breaking on 'et al.', decimals, or initials."""
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    protected = _ABBREV_RE.sub(lambda m: m.group(1) + _PLACEHOLDER, text)
    protected = _DECIMAL_RE.sub(lambda m: m.group(1) + _PLACEHOLDER + m.group(2), protected)
    protected = _INITIAL_RE.sub(lambda m: m.group(1) + _PLACEHOLDER, protected)
    return [p.replace(_PLACEHOLDER, ".").strip() for p in _SPLIT_RE.split(protected) if p.strip()]


def normalize(text: str) -> str:
    """Lowercase, strip punctuation and collapse whitespace (for exact/near-exact duplicate keys)."""
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", text.lower())).strip()


def content_tokens(text: str) -> list[str]:
    return [t for t in re.findall(r"[A-Za-z][A-Za-z0-9\-]+", text) if t.lower() not in STOPWORDS]


def keyword_query(question: str, max_terms: int = 6) -> str:
    """Salient keywords from a natural-language question, for engines that dislike full sentences.

    Prefers acronyms and long or hyphenated terms; keeps first-seen order among equals.
    """
    order: dict[str, int] = {}
    original: dict[str, str] = {}
    for tok in content_tokens(question):
        key = tok.lower()
        if key not in order:
            order[key] = len(order)
            original[key] = tok

    def weight(key: str) -> tuple[int, int]:
        raw = original[key]
        score = (3 if raw.isupper() and len(raw) > 1 else 0) + (2 if "-" in raw else 0) + min(len(raw), 12) // 4
        return (-score, order[key])

    chosen = sorted(order, key=weight)[:max_terms]
    return " ".join(original[k] for k in sorted(chosen, key=lambda k: order[k]))


def lexical_overlap(a: str, b: str) -> float:
    """Jaccard overlap of content tokens (fallback relevance/similarity when no embedder is available)."""
    ta = {t.lower() for t in content_tokens(a)}
    tb = {t.lower() for t in content_tokens(b)}
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)
