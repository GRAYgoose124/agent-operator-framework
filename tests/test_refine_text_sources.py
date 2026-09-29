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
