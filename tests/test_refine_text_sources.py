"""Refine text utilities and evidence sources (network tests skip when the service is unreachable)."""

from __future__ import annotations

import pytest

from aof.refine.sources import (
    OpenAlexSource,
    PubMedSource,
    WikipediaSource,
    build_sources,
    gather_documents,
    parse_pubmed_xml,
    rebuild_abstract,
)
from aof.refine.text import keyword_query, lexical_overlap, normalize, split_sentences


def test_split_sentences_handles_abbreviations_and_decimals():
    text = (
        "Spindles were first described by Steriade et al. in 1993. They peak at 12.5 Hz in humans. "
        "See Fig. 2 for details. J. Smith disagreed."
    )
    assert split_sentences(text) == [
        "Spindles were first described by Steriade et al. in 1993.",
        "They peak at 12.5 Hz in humans.",
        "See Fig. 2 for details.",
        "J. Smith disagreed.",
    ]


def test_keyword_query_prefers_salient_terms():
    q = keyword_query("What is the thalamic reticular nucleus (TRN) and how does it generate sleep spindles?")
    words = q.lower().split()
    assert "trn" in words and "thalamic" in words and "spindles" in words
    assert "what" not in words and "the" not in words
    assert len(words) <= 6


def test_normalize_and_overlap():
    assert normalize("The TRN's  GABAergic, shell!") == "the trn s gabaergic shell"
    assert lexical_overlap("GABAergic thalamic shell", "thalamic GABAergic shell") == 1.0
    assert lexical_overlap("grid cells", "sharp wave ripples") == 0.0


def test_rebuild_abstract_orders_words_by_position():
    assert rebuild_abstract({"cells": [1], "Grid": [0], "fire": [2]}) == "Grid cells fire"


def test_parse_pubmed_xml_labels_and_skips_abstractless():
    xml = """<PubmedArticleSet>
      <PubmedArticle><MedlineCitation><PMID>111</PMID><Article><Journal><Title>J Neuro</Title>
        <JournalIssue><PubDate><Year>2017</Year></PubDate></JournalIssue></Journal>
        <ArticleTitle>TRN cell types</ArticleTitle>
        <Abstract><AbstractText Label="BACKGROUND">The TRN is inhibitory.</AbstractText>
                  <AbstractText Label="RESULTS">PV and SOM cells differ.</AbstractText></Abstract>
      </Article></MedlineCitation><PubmedData><ArticleIdList><ArticleId IdType="doi">10.1/x</ArticleId></ArticleIdList></PubmedData></PubmedArticle>
      <PubmedArticle><MedlineCitation><PMID>222</PMID><Article><ArticleTitle>No abstract</ArticleTitle></Article></MedlineCitation></PubmedArticle>
    </PubmedArticleSet>"""
    docs = parse_pubmed_xml(xml)
    assert len(docs) == 1
    d = docs[0]
    assert d.url == "https://pubmed.ncbi.nlm.nih.gov/111/" and d.authority == 3
    assert d.text == "Background: The TRN is inhibitory. Results: PV and SOM cells differ."
    assert d.meta["doi"] == "10.1/x" and d.meta["year"] == "2017" and d.meta["journal"] == "J Neuro"


def test_build_sources_rejects_unknown():
    with pytest.raises(ValueError, match="unknown source"):
        build_sources(["pubmed", "nope"])


@pytest.mark.parametrize("source", [PubMedSource(), OpenAlexSource(), WikipediaSource()], ids=lambda s: s.name)
async def test_real_source_returns_quotable_neuro_documents(source):
    docs = await source.search("thalamic reticular nucleus sleep spindles", 3)
    if not docs:
        pytest.skip(f"{source.name} unreachable or returned nothing")
    for d in docs:
        assert d.url.startswith("http") and len(d.text) > 200 and d.source == source.name


async def test_gather_documents_dedupes_by_url_and_orders_by_authority():
    docs = await gather_documents(build_sources(["wikipedia", "pubmed"]), ["thalamic reticular nucleus spindles"], 3)
    if not docs:
        pytest.skip("no source reachable")
    assert len({d.url for d in docs}) == len(docs)
    assert [d.authority for d in docs] == sorted((d.authority for d in docs), reverse=True)


def test_same_paper_from_pubmed_and_openalex_is_one_document():
    from aof.refine.sources import SourceDoc, doc_key

    pubmed = SourceDoc("https://pubmed.ncbi.nlm.nih.gov/1/", "TRN and spindles: a study", "t", "pubmed", 3,
                       {"doi": "10.1523/JNEUROSCI.0001-20.2020"})
    alex = SourceDoc("https://doi.org/10.1523/jneurosci.0001-20.2020", "TRN and spindles: a study", "t", "openalex", 3)
    assert doc_key(pubmed) == doc_key(alex)
    # no DOI: fall back to the normalised title, but never merge short/generic titles
    a = SourceDoc("u1", "Thalamic Reticular Nucleus: A Review of Cell Types!", "t", "wikipedia", 2)
    b = SourceDoc("u2", "thalamic reticular nucleus - a review of cell types", "t", "web", 1)
    assert doc_key(a) == doc_key(b)
    assert doc_key(SourceDoc("u3", "Review", "t", "web", 1)) == "url:u3"


async def test_gather_documents_merges_copies_of_one_work_preferring_pubmed():
    from aof.refine.sources import SourceDoc, gather_documents

    class Fake:
        def __init__(self, name, doc):
            self.name, self._doc = name, doc

        async def search(self, query, limit=5):
            return [self._doc]

    pm = SourceDoc("https://pubmed.ncbi.nlm.nih.gov/1/", "Same paper", "pm text", "pubmed", 3, {"doi": "10.1/abc"})
    oa = SourceDoc("https://doi.org/10.1/abc", "Same paper", "oa text", "openalex", 3)
    docs = await gather_documents([Fake("openalex", oa), Fake("pubmed", pm)], ["q"], 3)
    assert [d.source for d in docs] == ["pubmed"]


async def test_fetch_retries_rate_limits_then_succeeds():
    """A real local HTTP server answers 429 twice, then 200: the fetch must retry, not give up."""
    import aiohttp
    from aiohttp import web

    from aof.refine import sources

    calls = {"n": 0}

    async def handler(request):
        calls["n"] += 1
        if calls["n"] <= 2:
            return web.Response(status=429, headers={"Retry-After": "0"})
        return web.json_response({"ok": True})

    app = web.Application()
    app.router.add_get("/x", handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    try:
        async with aiohttp.ClientSession() as session:
            got = await sources._fetch(session, f"http://127.0.0.1:{port}/x", {}, as_json=True)
            assert got == {"ok": True} and calls["n"] == 3
            calls["n"] = -100  # always 429 now: gives up after the attempt budget and returns None
            assert await sources._fetch(session, f"http://127.0.0.1:{port}/x", {}, as_json=True, attempts=2) is None
    finally:
        await runner.cleanup()


async def test_throttle_spaces_concurrent_requests():
    import asyncio
    import time

    from aof.refine.sources import _Throttle

    t = _Throttle(0.1)
    started = time.monotonic()
    await asyncio.gather(*(t.wait() for _ in range(4)))
    assert time.monotonic() - started >= 0.29  # 4 slots => at least 3 gaps of 0.1s


async def test_long_retry_after_puts_host_on_cooldown_instead_of_stalling():
    import time

    import aiohttp
    from aiohttp import web

    from aof.refine import sources

    hits = {"n": 0}

    async def handler(request):
        hits["n"] += 1
        return web.Response(status=429, headers={"Retry-After": "120"})

    app = web.Application()
    app.router.add_get("/x", handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    url = f"http://127.0.0.1:{port}/x"
    try:
        async with aiohttp.ClientSession() as session:
            started = time.monotonic()
            assert await sources._fetch(session, url, {}, as_json=True) is None
            assert await sources._fetch(session, url, {}, as_json=True) is None  # cooling down: no second request
            assert time.monotonic() - started < 5 and hits["n"] == 1
    finally:
        sources._cooldowns.pop(f"127.0.0.1:{port}", None)
        await runner.cleanup()


async def test_openalex_query_is_sanitised_and_key_comes_from_env(monkeypatch):
    """Real local server records what OpenAlexSource actually sends."""
    from aiohttp import web

    from aof.refine import sources

    seen: list[dict] = []

    async def handler(request):
        seen.append(dict(request.query))
        return web.json_response({"results": []})

    app = web.Application()
    app.router.add_get("/works", handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    source = sources.OpenAlexSource()
    source._URL = f"http://127.0.0.1:{port}/works"
    try:
        monkeypatch.delenv("OPENALEX_API_KEY", raising=False)
        await source.search("What is the TRN (thalamic reticular nucleus): anatomy?", 3)
        assert seen[0]["search"].split() == "What is the TRN thalamic reticular nucleus anatomy".split()
        assert "api_key" not in seen[0]
        monkeypatch.setenv("OPENALEX_API_KEY", "k")
        await source.search("x y", 3)
        assert seen[1]["api_key"] == "k"
    finally:
        await runner.cleanup()
