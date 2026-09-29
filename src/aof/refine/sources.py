"""Evidence sources: peer-reviewed abstracts (PubMed, OpenAlex), Wikipedia, and general web search.

Every source returns `SourceDoc`s carrying verbatim text to quote from, a stable URL, and an
`authority` level that later steps use to break ties (never to hide lower-authority evidence).
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Protocol
from urllib.parse import urlparse

import aiohttp

logger = logging.getLogger(__name__)

USER_AGENT = "AOF-Refine/0.1 (https://github.com/GRAYgoose124/agent-operator-framework; local research framework)"
AUTHORITY = {"pubmed": 3, "openalex": 3, "wikipedia": 2, "web": 1}


@dataclass(frozen=True)
class SourceDoc:
    url: str
    title: str
    text: str  # verbatim text claims will be quoted from
    source: str  # provider name
    authority: int = 1
    meta: dict = field(default_factory=dict)  # year, doi, journal, cited_by, ...


class Source(Protocol):
    name: str

    async def search(self, query: str, limit: int = 5) -> list[SourceDoc]: ...


# Minimum seconds between requests per host (NCBI allows ~3/s without an API key; OpenAlex asks for politeness).
_MIN_INTERVAL = {"eutils.ncbi.nlm.nih.gov": 0.4, "api.openalex.org": 0.2, "en.wikipedia.org": 0.15}


class _Throttle:
    """Spaces requests to one host; safe to share between concurrent tasks."""

    def __init__(self, min_interval: float) -> None:
        self._min = min_interval
        self._next = 0.0
        self._lock = asyncio.Lock()

    async def wait(self) -> None:
        async with self._lock:
            now = time.monotonic()
            delay = self._next - now
            self._next = max(now, self._next) + self._min
        if delay > 0:
            await asyncio.sleep(delay)


_throttles: dict[tuple[str, int], _Throttle] = {}
_cooldowns: dict[str, float] = {}  # host -> monotonic time before which it should not be called
MAX_INLINE_WAIT = 15.0  # a Retry-After longer than this puts the host on cooldown instead of stalling the run


def _throttle_for(url: str) -> _Throttle:
    host = urlparse(url).netloc
    key = (host, id(asyncio.get_running_loop()))  # asyncio primitives are bound to one event loop
    if key not in _throttles:
        _throttles[key] = _Throttle(_MIN_INTERVAL.get(host, 0.0))
    return _throttles[key]


def _retry_after(resp: aiohttp.ClientResponse, attempt: int) -> float:
    try:
        return min(float(resp.headers.get("Retry-After", "")), 30.0)
    except ValueError:
        return min(1.5 * 2**attempt, 30.0)


async def _fetch(
    session: aiohttp.ClientSession, url: str, params: dict, *, as_json: bool, timeout: int = 30, attempts: int = 4,
):
    """GET with per-host throttling and backoff on 429/5xx. Returns None when the source cannot be reached.

    A silent 429 would quietly cost recall, so rate limits are retried rather than dropped.
    """
    throttle = _throttle_for(url)
    name = url.split("?")[0]
    host = urlparse(url).netloc
    for attempt in range(attempts):
        if time.monotonic() < _cooldowns.get(host, 0.0):
            return None  # rate-limited recently: skip this source for now rather than block every caller
        await throttle.wait()
        try:
            async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=timeout)) as resp:
                if resp.status in (429, 502, 503, 504):
                    delay = _retry_after(resp, attempt)
                    if delay > MAX_INLINE_WAIT:
                        _cooldowns[host] = time.monotonic() + delay
                        logger.warning("%s rate limited (HTTP %s); skipping it for %.0fs", host, resp.status, delay)
                        return None
                    logger.info("%s -> HTTP %s; retrying in %.1fs", name, resp.status, delay)
                    await asyncio.sleep(delay)
                    continue
                if resp.status != 200:
                    logger.warning("%s -> HTTP %s", name, resp.status)
                    return None
                return await (resp.json(content_type=None) if as_json else resp.text())
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            logger.info("%s failed (%s); attempt %d/%d", name, e, attempt + 1, attempts)
            await asyncio.sleep(min(1.5 * 2**attempt, 30.0))
    logger.warning("%s unavailable after %d attempts (rate limited?)", name, attempts)
    return None


async def _get_json(session: aiohttp.ClientSession, url: str, params: dict) -> dict | None:
    return await _fetch(session, url, params, as_json=True)


class PubMedSource:
    """PubMed via NCBI E-utilities: titles and structured abstracts of peer-reviewed papers."""

    name = "pubmed"
    _ESEARCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    _EFETCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

    async def search(self, query: str, limit: int = 5) -> list[SourceDoc]:
        async with aiohttp.ClientSession(headers={"User-Agent": USER_AGENT}) as session:
            found = await _get_json(session, self._ESEARCH, {
                "db": "pubmed", "term": query, "retmax": limit, "retmode": "json", "sort": "relevance",
            })
            ids = (found or {}).get("esearchresult", {}).get("idlist", [])
            if not ids:
                return []
            xml = await _fetch(
                session, self._EFETCH, {"db": "pubmed", "id": ",".join(ids), "retmode": "xml"},
                as_json=False, timeout=45,
            )
        return parse_pubmed_xml(xml) if xml else []


def parse_pubmed_xml(xml: str) -> list[SourceDoc]:
    docs: list[SourceDoc] = []
    for art in ET.fromstring(xml).iter("PubmedArticle"):
        pmid = (art.findtext(".//PMID") or "").strip()
        title_el = art.find(".//ArticleTitle")
        title = "".join(title_el.itertext()).strip() if title_el is not None else ""
        parts = []
        for ab in art.findall(".//Abstract/AbstractText"):
            text = "".join(ab.itertext()).strip()
            if text:
                label = ab.get("Label")
                unlabeled = not label or label.upper() in ("UNLABELLED", "UNLABELED")
                parts.append(text if unlabeled else f"{label.capitalize()}: {text}")
        if not (pmid and parts):
            continue  # nothing to quote from
        doi = next((i.text for i in art.findall(".//ArticleId") if i.get("IdType") == "doi"), None)
        year = art.findtext(".//JournalIssue/PubDate/Year") or (art.findtext(".//JournalIssue/PubDate/MedlineDate") or "")[:4]
        docs.append(SourceDoc(
            url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/", title=title, text=" ".join(parts),
            source="pubmed", authority=AUTHORITY["pubmed"],
            meta={"pmid": pmid, "doi": doi, "year": year, "journal": art.findtext(".//Journal/Title") or ""},
        ))
    return docs


class OpenAlexSource:
    """OpenAlex works with abstracts (reconstructed from the inverted index); carries citation counts."""

    name = "openalex"
    _URL = "https://api.openalex.org/works"

    async def search(self, query: str, limit: int = 5) -> list[SourceDoc]:
        params = {
            # OpenAlex rejects some punctuation (colons, parentheses, question marks) in `search`
            "search": re.sub(r"[^\w\s\-]", " ", query), "per-page": limit, "filter": "has_abstract:true",
            "select": "id,title,publication_year,doi,cited_by_count,abstract_inverted_index",
        }
        if os.environ.get("OPENALEX_API_KEY"):  # optional free key for uninterrupted access; never stored
            params["api_key"] = os.environ["OPENALEX_API_KEY"]
        async with aiohttp.ClientSession(headers={"User-Agent": USER_AGENT}) as session:
            data = await _get_json(session, self._URL, params)
        docs: list[SourceDoc] = []
        for w in (data or {}).get("results", []):
            text = rebuild_abstract(w.get("abstract_inverted_index") or {})
            if not text:
                continue
            docs.append(SourceDoc(
                url=w.get("doi") or w.get("id", ""), title=w.get("title") or "", text=text,
                source="openalex", authority=AUTHORITY["openalex"],
                meta={"year": w.get("publication_year"), "cited_by": w.get("cited_by_count", 0)},
            ))
        return docs


def rebuild_abstract(inverted: dict[str, list[int]]) -> str:
    """OpenAlex stores abstracts as {word: [positions]}; restore the running text."""
    if not inverted:
        return ""
    slots: dict[int, str] = {}
    for word, positions in inverted.items():
        for pos in positions:
            slots[pos] = word
    return " ".join(slots[i] for i in sorted(slots))


class WikipediaSource:
    """Wikipedia plain-text extracts: good for background definitions, medium authority."""

    name = "wikipedia"
    _API = "https://en.wikipedia.org/w/api.php"

    async def search(self, query: str, limit: int = 3) -> list[SourceDoc]:
        async with aiohttp.ClientSession(headers={"User-Agent": USER_AGENT}) as session:
            hits = await _get_json(session, self._API, {
                "action": "query", "list": "search", "srsearch": query, "srlimit": limit, "format": "json",
            })
            titles = [h["title"] for h in (hits or {}).get("query", {}).get("search", [])]
            if not titles:
                return []
            pages = await _get_json(session, self._API, {
                "action": "query", "prop": "extracts", "explaintext": 1, "exlimit": len(titles),
                "titles": "|".join(titles), "format": "json", "redirects": 1,
            })
        docs: list[SourceDoc] = []
        for page in ((pages or {}).get("query", {}).get("pages", {})).values():
            text = re.sub(r"\n=+ .*? =+\n", "\n", page.get("extract", "")).strip()
            if text:
                title = page["title"]
                docs.append(SourceDoc(
                    url="https://en.wikipedia.org/wiki/" + title.replace(" ", "_"), title=title,
                    text=text[:12000], source="wikipedia", authority=AUTHORITY["wikipedia"],
                ))
        return docs


class WebSource:
    """General web search (DuckDuckGo) + page fetch. Lowest authority; often rate-limited."""

    name = "web"

    async def search(self, query: str, limit: int = 3) -> list[SourceDoc]:
        from aof.tools.builtin.scraper import _sync_scrape
        from aof.tools.builtin.search import _sync_search

        results = await asyncio.to_thread(_sync_search, query, limit)
        if not isinstance(results, list):
            return []
        docs: list[SourceDoc] = []
        for r in results:
            if not r.get("url"):
                continue  # the "No results" hint row
            page = await asyncio.to_thread(_sync_scrape, r["url"], 12000)
            text = page.get("content") or r.get("snippet", "")
            if text:
                docs.append(SourceDoc(
                    url=r["url"], title=r.get("title", ""), text=text, source="web", authority=AUTHORITY["web"],
                ))
        return docs


SOURCE_CLASSES = {"pubmed": PubMedSource, "openalex": OpenAlexSource, "wikipedia": WikipediaSource, "web": WebSource}


def build_sources(names: tuple[str, ...] | list[str]) -> list[Source]:
    unknown = [n for n in names if n not in SOURCE_CLASSES]
    if unknown:
        raise ValueError(f"unknown source(s) {unknown}; choose from {sorted(SOURCE_CLASSES)}")
    return [SOURCE_CLASSES[n]() for n in names]


def doc_key(doc: SourceDoc) -> str:
    """Identity of the underlying work: DOI when known, else normalised title, else URL.

    The same paper arrives via PubMed *and* OpenAlex; counting it twice would fake independent corroboration.
    """
    doi = str(doc.meta.get("doi") or (doc.url if "doi.org/" in doc.url else "") or "").lower()
    if doi:
        return "doi:" + re.sub(r"^https?://(dx\.)?doi\.org/", "", doi)
    title = re.sub(r"[^a-z0-9]+", " ", doc.title.lower()).strip()
    return "title:" + title if len(title) >= 20 else "url:" + doc.url


async def gather_documents(sources: list[Source], queries: list[str], per_source: int = 4) -> list[SourceDoc]:
    """Run every query on every source concurrently; de-duplicate by underlying work (see `doc_key`).

    Among copies of the same work the highest authority wins, then PubMed (labelled abstracts).
    """
    jobs = [s.search(q, per_source) for s in sources for q in queries]
    batches = await asyncio.gather(*jobs, return_exceptions=True)
    best: dict[str, SourceDoc] = {}
    for batch in batches:
        if isinstance(batch, Exception):
            logger.warning("source search raised %s: %s", type(batch).__name__, batch)
            continue
        for doc in batch:
            key = doc_key(doc)
            rank = (doc.authority, doc.source == "pubmed")
            if key not in best or rank > (best[key].authority, best[key].source == "pubmed"):
                best[key] = doc
    return sorted(best.values(), key=lambda d: (-d.authority, d.url))
