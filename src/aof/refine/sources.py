"""Evidence sources: peer-reviewed abstracts (PubMed, OpenAlex), Wikipedia, and general web search.

Every source returns `SourceDoc`s carrying verbatim text to quote from, a stable URL, and an
`authority` level that later steps use to break ties (never to hide lower-authority evidence).
"""

from __future__ import annotations

import asyncio
import logging
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Protocol

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


async def _get_json(session: aiohttp.ClientSession, url: str, params: dict) -> dict | None:
    try:
        async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=30)) as resp:
            if resp.status != 200:
                logger.warning("%s -> HTTP %s", url.split("?")[0], resp.status)
                return None
            return await resp.json(content_type=None)
    except (aiohttp.ClientError, asyncio.TimeoutError) as e:
        logger.warning("%s failed: %s", url.split("?")[0], e)
        return None


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
            await asyncio.sleep(0.35)  # NCBI allows ~3 requests/s without an API key
            try:
                async with session.get(
                    self._EFETCH, params={"db": "pubmed", "id": ",".join(ids), "retmode": "xml"},
                    timeout=aiohttp.ClientTimeout(total=45),
                ) as resp:
                    if resp.status != 200:
                        return []
                    xml = await resp.text()
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                logger.warning("PubMed efetch failed: %s", e)
                return []
        return parse_pubmed_xml(xml)


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
        async with aiohttp.ClientSession(headers={"User-Agent": USER_AGENT}) as session:
            data = await _get_json(session, self._URL, {
                "search": query, "per-page": limit, "filter": "has_abstract:true",
                "select": "id,title,publication_year,doi,cited_by_count,abstract_inverted_index",
            })
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


async def gather_documents(sources: list[Source], queries: list[str], per_source: int = 4) -> list[SourceDoc]:
    """Run every query on every source concurrently; de-duplicate by URL (highest authority wins)."""
    jobs = [s.search(q, per_source) for s in sources for q in queries]
    batches = await asyncio.gather(*jobs, return_exceptions=True)
    best: dict[str, SourceDoc] = {}
    for batch in batches:
        if isinstance(batch, Exception):
            logger.warning("source search raised %s: %s", type(batch).__name__, batch)
            continue
        for doc in batch:
            if doc.url not in best or doc.authority > best[doc.url].authority:
                best[doc.url] = doc
    return sorted(best.values(), key=lambda d: (-d.authority, d.url))
