"""Tests for BeautifulSoup4 HTML extraction tools."""

from __future__ import annotations

import pytest

from aof.tools.registry import ToolRegistry


@pytest.fixture
def tools_with_html():
    from aof.tools.builtin import html_tools

    reg = ToolRegistry()
    html_tools.register(reg)
    return reg


@pytest.mark.asyncio
async def test_extract_links(tools_with_html):
    html = '<html><body><a href="/about">About</a><a href="https://example.com/page">Page</a></body></html>'
    result = await tools_with_html.invoke(
        "extract_links",
        {"url": "https://example.com/", "html": html},
    )
    assert "links" in result
    assert len(result["links"]) >= 1
    urls = [l["url"] for l in result["links"]]
    assert any("about" in u or "page" in u for u in urls)


@pytest.mark.asyncio
async def test_extract_meta(tools_with_html):
    html = '<html><head><title>Test Page</title><meta name="description" content="A test"></head><body></body></html>'
    result = await tools_with_html.invoke(
        "extract_meta",
        {"url": "https://example.com/", "html": html},
    )
    assert "title" in result
    assert result["title"] == "Test Page"
    assert "description" in result
    assert "test" in result["description"]


@pytest.mark.asyncio
async def test_parse_html_fragment(tools_with_html):
    html = '<div class="content"><p>First</p><p>Second</p></div>'
    result = await tools_with_html.invoke(
        "parse_html_fragment",
        {"html": html, "selector": "p"},
    )
    assert result["matches"] == 2
    assert "First" in result["texts"]
    assert "Second" in result["texts"]
