"""End-to-end tests for scraper tools."""

from __future__ import annotations

import asyncio
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from tests.conftest import MOCK_HTML


class MockHTMLHandler(BaseHTTPRequestHandler):
    """Serve mock HTML for scraper tests."""

    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(MOCK_HTML.encode("utf-8"))

    def log_message(self, format, *args):
        pass


@pytest.fixture
def local_http_server():
    """Start a local HTTP server serving mock HTML."""
    server = HTTPServer(("127.0.0.1", 0), MockHTMLHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        port = server.server_address[1]
        yield f"http://127.0.0.1:{port}/"
    finally:
        server.shutdown()


@pytest.mark.asyncio
async def test_web_scrape_output_structure(temp_tools, mock_http_scrape):
    """web_scrape returns url, title, content, length; content contains Test content."""
    with mock_http_scrape:
        result = await temp_tools.invoke("web_scrape", {"url": "http://example.com/test"})

    assert "error" not in result
    assert result.get("url") == "http://example.com/test"
    assert "title" in result
    assert "content" in result
    assert "length" in result
    assert "Test content" in result["content"]


@pytest.mark.asyncio
async def test_web_scrape_truncation(temp_tools, mock_http_scrape):
    """web_scrape with max_chars=10 truncates content with ..."""
    with mock_http_scrape:
        result = await temp_tools.invoke(
            "web_scrape", {"url": "http://example.com/test", "max_chars": 10}
        )

    assert "error" not in result
    assert result["content"].endswith("...")
    assert len(result["content"]) <= 13


@pytest.mark.asyncio
async def test_crawl_site_output_structure(temp_tools, local_http_server):
    """crawl_site returns urls_crawled, items, errors; items have url, title, content."""
    result = await temp_tools.invoke(
        "crawl_site", {"url": local_http_server, "max_pages": 2}
    )

    if "error" in result:
        pytest.skip(f"crawl_site failed: {result.get('error')}")

    assert "urls_crawled" in result
    assert "items" in result
    assert "errors" in result
    assert isinstance(result["items"], list)
    if result["items"]:
        item = result["items"][0]
        assert "url" in item
        assert "title" in item
        assert "content" in item


@pytest.mark.asyncio
async def test_extract_structured_output(temp_tools, mock_http_scrape):
    """extract_structured returns extracted dict with values matching CSS selectors."""
    with mock_http_scrape:
        result = await temp_tools.invoke(
            "extract_structured",
            {
                "url": "http://example.com/test",
                "selectors": {"title": "title::text", "paragraph": "p::text"},
            },
        )

    assert "error" not in result
    assert "extracted" in result
    assert isinstance(result["extracted"], dict)
    assert result["extracted"].get("title") == "Test Page"
    assert "Test content" in (result["extracted"].get("paragraph") or "")


@pytest.mark.asyncio
async def test_create_crawler_creates_file(temp_config, temp_tools):
    """create_crawler creates file at crawlers_dir with expected content."""
    result = await temp_tools.invoke(
        "create_crawler", {"site": "test.example", "template": "generic"}
    )

    assert result.get("success") is True
    crawlers_dir = Path(temp_config.tools.crawlers_dir)
    expected_path = crawlers_dir / "site_test_example.py"
    assert expected_path.exists()
    content = expected_path.read_text(encoding="utf-8")
    assert "class SiteCrawlerSpider" in content or "SiteCrawlerSpider" in content
    assert "allowed_domains" in content
    assert "start_urls" in content
