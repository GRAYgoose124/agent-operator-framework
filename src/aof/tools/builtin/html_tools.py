"""HTML extraction tools using BeautifulSoup4."""

from __future__ import annotations

import asyncio
import logging
from typing import Any
from urllib.parse import urljoin, urlparse

from aof.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


def _sync_extract_links(url: str, html: str, base_url: str | None = None) -> dict:
    """Extract href links from HTML. Returns list of {url, text}."""
    try:
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html, "html.parser")
        base = base_url or url
        links = []
        for a in soup.find_all("a", href=True):
            href = a["href"].strip()
            if not href or href.startswith("#") or href.startswith("javascript:"):
                continue
            full_url = urljoin(base, href)
            try:
                parsed = urlparse(full_url)
                if parsed.scheme in ("http", "https"):
                    text = (a.get_text() or "").strip()[:200]
                    links.append({"url": full_url, "text": text or "(no text)"})
            except Exception:
                continue
        return {"url": url, "links": links[:50], "count": len(links)}
    except ImportError:
        return {"error": "beautifulsoup4 required. pip install beautifulsoup4"}
    except Exception as e:
        logger.warning("extract_links failed: %s", e)
        return {"error": str(e)}


def _sync_extract_meta(url: str, html: str) -> dict:
    """Extract meta tags: title, description, og:title, og:description, etc."""
    try:
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html, "html.parser")
        result: dict[str, str] = {"url": url}

        title_elem = soup.find("title")
        if title_elem:
            result["title"] = title_elem.get_text().strip()[:500]

        for meta in soup.find_all("meta", attrs={"name": True}) or []:
            name = meta.get("name", "").lower()
            if name in ("description", "keywords", "author"):
                content = meta.get("content", "")
                if content:
                    result[name] = content.strip()[:500]

        for meta in soup.find_all("meta", attrs={"property": True}) or []:
            prop = meta.get("property", "").lower()
            if prop.startswith("og:"):
                key = prop.replace(":", "_")
                content = meta.get("content", "")
                if content:
                    result[key] = content.strip()[:500]

        return result
    except ImportError:
        return {"error": "beautifulsoup4 required. pip install beautifulsoup4"}
    except Exception as e:
        logger.warning("extract_meta failed: %s", e)
        return {"error": str(e)}


def _sync_parse_html_fragment(html: str, selector: str) -> dict:
    """Parse HTML string and extract text by CSS selector."""
    try:
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html, "html.parser")
        elements = soup.select(selector)
        texts = []
        for el in elements[:20]:
            t = el.get_text(separator=" ", strip=True)
            if t:
                texts.append(t[:500])
        return {"selector": selector, "matches": len(elements), "texts": texts}
    except ImportError:
        return {"error": "beautifulsoup4 required. pip install beautifulsoup4"}
    except Exception as e:
        logger.warning("parse_html_fragment failed: %s", e)
        return {"error": str(e)}


def register(registry: ToolRegistry) -> None:
    """Register BeautifulSoup4-based HTML extraction tools."""

    @registry.register_function(
        name="extract_links",
        description="Extract links from HTML. Provide URL and HTML content (or fetch first with web_scrape). Returns list of {url, text}.",
        parameters={
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "Source URL for resolving relative links",
                },
                "html": {
                    "type": "string",
                    "description": "HTML content to parse",
                },
                "base_url": {
                    "type": "string",
                    "description": "Optional base URL for relative links (defaults to url)",
                },
            },
            "required": ["url", "html"],
        },
    )
    async def extract_links(url: str, html: str, base_url: str | None = None) -> Any:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, _sync_extract_links, url, html, base_url
        )

    @registry.register_function(
        name="extract_meta",
        description="Extract meta tags from HTML: title, description, og:title, og:description. Useful for citation metadata.",
        parameters={
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "Source URL",
                },
                "html": {
                    "type": "string",
                    "description": "HTML content to parse",
                },
            },
            "required": ["url", "html"],
        },
    )
    async def extract_meta(url: str, html: str) -> Any:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, _sync_extract_meta, url, html)

    @registry.register_function(
        name="parse_html_fragment",
        description="Parse HTML string and extract text by CSS selector. Use when you have raw HTML from another source.",
        parameters={
            "type": "object",
            "properties": {
                "html": {
                    "type": "string",
                    "description": "HTML content to parse",
                },
                "selector": {
                    "type": "string",
                    "description": "CSS selector (e.g. 'article', '.content', 'h1')",
                },
            },
            "required": ["html", "selector"],
        },
    )
    async def parse_html_fragment(html: str, selector: str) -> Any:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, _sync_parse_html_fragment, html, selector)
