"""Web scraping tools using Scrapy (with optional Playwright for JS rendering)."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from aof.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


def register(registry: ToolRegistry) -> None:
    """Register web scraping tools."""

    @registry.register_function(
        name="web_scrape",
        description=(
            "Scrape text content from a single URL using a lightweight HTTP + Scrapy "
            "selector pipeline. Use this for mostly static HTML pages. If the page "
            "loads content via JavaScript or remains empty, fall back to "
            "scrape_with_playwright."
        ),
        parameters={
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The URL to scrape",
                },
                "max_chars": {
                    "type": "integer",
                    "description": "Maximum characters to return (default 5000)",
                },
            },
            "required": ["url"],
        },
    )
    async def web_scrape(url: str, max_chars: int = 5000) -> Any:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, _sync_scrape, url, max_chars)

    @registry.register_function(
        name="crawl_site",
        description=(
            "Crawl a domain starting from a seed URL. Follows same-domain links up to "
            "max_pages. Use when you need broad coverage of a site that does not have "
            "a convenient sitemap.xml. Returns urls_crawled, items (url, title, "
            "content), and errors."
        ),
        parameters={
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "Seed URL to start crawling",
                },
                "max_pages": {
                    "type": "integer",
                    "description": "Maximum pages to crawl (default 10)",
                },
                "max_chars_per_page": {
                    "type": "integer",
                    "description": "Max characters to extract per page (default 3000)",
                },
            },
            "required": ["url"],
        },
    )
    async def crawl_site(url: str, max_pages: int = 10, max_chars_per_page: int = 3000) -> Any:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, _sync_crawl_site, url, max_pages, max_chars_per_page
        )

    @registry.register_function(
        name="crawl_sitemap",
        description=(
            "Parse sitemap.xml from a URL and fetch listed pages. Use this when the "
            "site exposes a sitemap and you want structured coverage of key pages. "
            "Returns urls_crawled, items, and errors."
        ),
        parameters={
            "type": "object",
            "properties": {
                "sitemap_url": {
                    "type": "string",
                    "description": "URL of sitemap.xml (e.g. https://example.com/sitemap.xml)",
                },
                "max_pages": {
                    "type": "integer",
                    "description": "Maximum pages to fetch (default 10)",
                },
                "max_chars_per_page": {
                    "type": "integer",
                    "description": "Max characters per page (default 3000)",
                },
            },
            "required": ["sitemap_url"],
        },
    )
    async def crawl_sitemap(
        sitemap_url: str, max_pages: int = 10, max_chars_per_page: int = 3000
    ) -> Any:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, _sync_crawl_sitemap, sitemap_url, max_pages, max_chars_per_page
        )

    @registry.register_function(
        name="scrape_with_playwright",
        description=(
            "Scrape a JS-rendered page using Playwright. Use this only when page "
            "content loads via JavaScript or when web_scrape returns empty/partial "
            "content, as it is heavier and slower than simple HTTP scraping."
        ),
        parameters={
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The URL to scrape",
                },
                "max_chars": {
                    "type": "integer",
                    "description": "Maximum characters to return (default 5000)",
                },
            },
            "required": ["url"],
        },
    )
    async def scrape_with_playwright(url: str, max_chars: int = 5000) -> Any:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, _sync_scrape_playwright, url, max_chars)

    @registry.register_function(
        name="follow_links",
        description="Crawl from a seed URL following links that match a pattern. Returns urls_crawled, items, and errors.",
        parameters={
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "Seed URL",
                },
                "link_pattern": {
                    "type": "string",
                    "description": "Regex pattern for link href to follow (e.g. '/blog/'). Empty = follow all same-domain links.",
                },
                "max_pages": {
                    "type": "integer",
                    "description": "Maximum pages (default 10)",
                },
                "max_chars_per_page": {
                    "type": "integer",
                    "description": "Max characters per page (default 3000)",
                },
            },
            "required": ["url"],
        },
    )
    async def follow_links(
        url: str,
        link_pattern: str = "",
        max_pages: int = 10,
        max_chars_per_page: int = 3000,
    ) -> Any:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, _sync_follow_links, url, link_pattern, max_pages, max_chars_per_page
        )

    @registry.register_function(
        name="extract_structured",
        description="Fetch a URL and extract content using CSS or XPath selectors. Returns structured dict of extracted values.",
        parameters={
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "URL to fetch",
                },
                "selectors": {
                    "type": "object",
                    "description": "Dict of label -> CSS selector or XPath. E.g. {\"title\": \"h1::text\", \"links\": \"a::attr(href)\"}. For XPath use /xpath/ prefix.",
                },
            },
            "required": ["url", "selectors"],
        },
    )
    async def extract_structured(url: str, selectors: dict[str, str]) -> Any:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, _sync_extract_structured, url, selectors)


def _sync_scrape(url: str, max_chars: int) -> dict:
    """Scrape a page using Scrapy's fetch mechanism."""
    try:
        from scrapy.http import HtmlResponse
        from scrapy.selector import Selector
        import urllib.request

        req = urllib.request.Request(url, headers={"User-Agent": "AOF-Agent/0.1"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            body = resp.read()
            encoding = resp.headers.get_content_charset() or "utf-8"

        response = HtmlResponse(url=url, body=body, encoding=encoding)
        sel = Selector(response=response)

        # Extract title
        title = sel.css("title::text").get("No title")

        # Extract text content, filtering out script/style
        for tag in ["script", "style", "nav", "footer", "header"]:
            sel.css(tag).drop()

        text_parts = sel.css("body *::text").getall()
        text = " ".join(t.strip() for t in text_parts if t.strip())

        # Truncate
        if len(text) > max_chars:
            text = text[:max_chars] + "..."

        return {
            "url": url,
            "title": title.strip(),
            "content": text,
            "length": len(text),
        }

    except ImportError:
        return {"error": "scrapy not installed. pip install scrapy"}
    except Exception as e:
        logger.warning("Scrape failed for %s: %s", url, e)
        return {
            "error": (
                f"Scrape failed: {e}. If the page uses heavy JavaScript or "
                "dynamic rendering, try scrape_with_playwright instead."
            )
        }


def _extract_text_from_response(sel, max_chars: int) -> str:
    """Extract body text from a Scrapy selector."""
    for tag in ["script", "style", "nav", "footer", "header"]:
        sel.css(tag).drop()
    text_parts = sel.css("body *::text").getall()
    text = " ".join(t.strip() for t in text_parts if t.strip())
    if len(text) > max_chars:
        text = text[:max_chars] + "..."
    return text


def _sync_crawl_site(url: str, max_pages: int, max_chars_per_page: int) -> dict:
    """Crawl a domain using Scrapy CrawlerProcess."""
    try:
        import scrapy
        from scrapy.crawler import CrawlerProcess
        from scrapy.utils.project import get_project_settings
        from urllib.parse import urlparse

        collected: list[dict] = []
        errors: list[str] = []

        class GenericCrawlSpider(scrapy.Spider):
            name = "generic_crawl"
            allowed_domains: list[str] = []
            custom_settings = {
                "CONCURRENT_REQUESTS": 2,
                "DOWNLOAD_DELAY": 0.5,
                "DEPTH_LIMIT": 3,
                "ROBOTSTXT_OBEY": True,
            }

            def __init__(self, *a, collected_ref=None, max_chars=3000, **kw):
                super().__init__(*a, **kw)
                self._collected = collected_ref or []
                self._max_chars = max_chars
                self._seen = 0
                self._max_pages = max_pages

            def parse(self, response):
                if self._seen >= self._max_pages:
                    return
                self._seen += 1
                try:
                    sel = response.selector
                    title = sel.css("title::text").get("") or ""
                    content = _extract_text_from_response(sel, self._max_chars)
                    self._collected.append({
                        "url": response.url,
                        "title": title.strip(),
                        "content": content,
                    })
                except Exception as e:
                    errors.append(f"{response.url}: {e}")

                if self._seen < self._max_pages:
                    for href in response.css("a::attr(href)").getall():
                        if href and href.startswith(("http://", "https://")):
                            parsed = urlparse(href)
                            if parsed.netloc in self.allowed_domains:
                                yield response.follow(href, self.parse)
                        elif href and href.startswith("/"):
                            yield response.follow(href, self.parse)

        parsed = urlparse(url)
        domain = parsed.netloc or parsed.path
        if not domain:
            return {"error": "Invalid URL", "urls_crawled": [], "items": [], "errors": []}

        settings = {
            "CONCURRENT_REQUESTS": 2,
            "DOWNLOAD_DELAY": 0.5,
            "DEPTH_LIMIT": 3,
            "ROBOTSTXT_OBEY": True,
            "LOG_LEVEL": "ERROR",
        }
        process = CrawlerProcess(settings)
        process.crawl(
            GenericCrawlSpider,
            start_urls=[url],
            allowed_domains=[domain],
            collected_ref=collected,
            max_chars=max_chars_per_page,
            max_pages=max_pages,
        )
        process.start()

        return {
            "urls_crawled": [i["url"] for i in collected],
            "items": collected,
            "errors": errors,
        }
    except ImportError as e:
        return {
            "error": f"scrapy not installed: {e}",
            "urls_crawled": [],
            "items": [],
            "errors": [],
        }
    except Exception as e:
        logger.warning("Crawl failed for %s: %s", url, e)
        return {
            "error": (
                f"Crawl failed: {e}. You can fall back to web_scrape on a few "
                "key URLs or use crawl_sitemap if the site exposes sitemap.xml."
            ),
            "urls_crawled": [],
            "items": [],
            "errors": [],
        }


def _sync_crawl_sitemap(
    sitemap_url: str, max_pages: int, max_chars_per_page: int
) -> dict:
    """Parse sitemap and fetch pages."""
    try:
        import scrapy
        from scrapy.crawler import CrawlerProcess
        from scrapy.spiders import SitemapSpider
        import xml.etree.ElementTree as ET
        import urllib.request

        # Fetch sitemap
        req = urllib.request.Request(
            sitemap_url, headers={"User-Agent": "AOF-Agent/0.1"}
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            body = resp.read()
        root = ET.fromstring(body)
        ns = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}
        urls = []
        for loc in root.findall(".//sm:loc", ns):
            if loc is not None and loc.text:
                urls.append(loc.text.strip())
        for loc in root.findall(".//{http://www.sitemaps.org/schemas/sitemap/0.9}loc"):
            if loc is not None and loc.text and loc.text not in urls:
                urls.append(loc.text.strip())
        urls = urls[:max_pages]

        if not urls:
            return {"error": "No URLs in sitemap", "urls_crawled": [], "items": [], "errors": []}

        collected: list[dict] = []
        errors: list[str] = []

        class SitemapFetchSpider(scrapy.Spider):
            name = "sitemap_fetch"

            def __init__(self, *a, urls_to_fetch=None, collected_ref=None, max_chars=3000, **kw):
                super().__init__(*a, **kw)
                self._urls = urls_to_fetch or []
                self._collected = collected_ref or []
                self._max_chars = max_chars

            def start_requests(self):
                for u in self._urls:
                    yield scrapy.Request(u, callback=self.parse)

            def parse(self, response):
                try:
                    sel = response.selector
                    title = sel.css("title::text").get("") or ""
                    content = _extract_text_from_response(sel, self._max_chars)
                    self._collected.append({
                        "url": response.url,
                        "title": title.strip(),
                        "content": content,
                    })
                except Exception as e:
                    errors.append(f"{response.url}: {e}")

        settings = {"CONCURRENT_REQUESTS": 2, "DOWNLOAD_DELAY": 0.5, "LOG_LEVEL": "ERROR"}
        process = CrawlerProcess(settings)
        process.crawl(
            SitemapFetchSpider,
            urls_to_fetch=urls,
            collected_ref=collected,
            max_chars=max_chars_per_page,
        )
        process.start()

        return {
            "urls_crawled": [i["url"] for i in collected],
            "items": collected,
            "errors": errors,
        }
    except ImportError as e:
        return {
            "error": f"scrapy not installed: {e}",
            "urls_crawled": [],
            "items": [],
            "errors": [],
        }
    except Exception as e:
        logger.warning("Sitemap crawl failed for %s: %s", sitemap_url, e)
        return {
            "error": (
                f"Sitemap crawl failed: {e}. If the sitemap is missing or invalid, "
                "try crawl_site from a representative seed URL instead."
            ),
            "urls_crawled": [],
            "items": [],
            "errors": [],
        }


def _sync_scrape_playwright(url: str, max_chars: int) -> dict:
    """Scrape JS-rendered page using scrapy-playwright."""
    try:
        from scrapy.http import HtmlResponse
        from scrapy.selector import Selector
        from playwright.sync_api import sync_playwright

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            try:
                page = browser.new_page()
                page.goto(url, wait_until="networkidle", timeout=30000)
                content = page.content()
                page.close()
            finally:
                browser.close()

        response = HtmlResponse(url=url, body=content.encode(), encoding="utf-8")
        sel = Selector(response=response)
        title = sel.css("title::text").get("No title")
        text = _extract_text_from_response(sel, max_chars)

        return {
            "url": url,
            "title": title.strip(),
            "content": text,
            "length": len(text),
        }
    except ImportError:
        return {
            "error": (
                "playwright not installed. "
                "pip install playwright && playwright install chromium"
            )
        }
    except Exception as e:
        logger.warning("Playwright scrape failed for %s: %s", url, e)
        return {
            "error": (
                f"Playwright scrape failed: {e}. As a fallback, you can try "
                "web_scrape (for static HTML) or crawl_site/crawl_sitemap for "
                "broader coverage."
            )
        }


def _sync_follow_links(
    url: str, link_pattern: str, max_pages: int, max_chars_per_page: int
) -> dict:
    """Crawl following links matching a pattern."""
    try:
        import re
        import scrapy
        from scrapy.crawler import CrawlerProcess
        from urllib.parse import urlparse, urljoin

        collected: list[dict] = []
        errors: list[str] = []
        parsed = urlparse(url)
        domain = parsed.netloc or parsed.path
        if not domain:
            return {"error": "Invalid URL", "urls_crawled": [], "items": [], "errors": []}

        pattern_re = re.compile(link_pattern) if link_pattern else None

        class LinkFollowSpider(scrapy.Spider):
            name = "link_follow"
            allowed_domains = [domain]
            custom_settings = {
                "CONCURRENT_REQUESTS": 2,
                "DOWNLOAD_DELAY": 0.5,
                "DEPTH_LIMIT": 3,
                "ROBOTSTXT_OBEY": True,
                "LOG_LEVEL": "ERROR",
            }

            def __init__(self, *a, collected_ref=None, max_chars=3000, max_pages=10, pattern=None, **kw):
                super().__init__(*a, **kw)
                self._collected = collected_ref or []
                self._max_chars = max_chars
                self._max_pages = max_pages
                self._seen = 0
                self._pattern = pattern

            def start_requests(self):
                yield scrapy.Request(url, callback=self.parse)

            def parse(self, response):
                if self._seen >= self._max_pages:
                    return
                self._seen += 1
                try:
                    sel = response.selector
                    title = sel.css("title::text").get("") or ""
                    content = _extract_text_from_response(sel, self._max_chars)
                    self._collected.append({
                        "url": response.url,
                        "title": title.strip(),
                        "content": content,
                    })
                except Exception as e:
                    errors.append(f"{response.url}: {e}")

                if self._seen < self._max_pages:
                    for href in response.css("a::attr(href)").getall():
                        if not href:
                            continue
                        full_url = urljoin(response.url, href)
                        if self._pattern and not self._pattern.search(href):
                            continue
                        try:
                            p = urlparse(full_url)
                            if p.netloc == domain or (p.netloc == "" and p.path):
                                yield response.follow(href, self.parse)
                        except Exception:
                            pass

        settings = {
            "CONCURRENT_REQUESTS": 2,
            "DOWNLOAD_DELAY": 0.5,
            "DEPTH_LIMIT": 3,
            "ROBOTSTXT_OBEY": True,
            "LOG_LEVEL": "ERROR",
        }
        process = CrawlerProcess(settings)
        process.crawl(
            LinkFollowSpider,
            collected_ref=collected,
            max_chars=max_chars_per_page,
            max_pages=max_pages,
            pattern=pattern_re,
        )
        process.start()

        return {
            "urls_crawled": [i["url"] for i in collected],
            "items": collected,
            "errors": errors,
        }
    except ImportError as e:
        return {
            "error": f"scrapy not installed: {e}",
            "urls_crawled": [],
            "items": [],
            "errors": [],
        }
    except Exception as e:
        logger.warning("Follow links failed for %s: %s", url, e)
        return {
            "error": (
                f"Follow-links crawl failed: {e}. You can narrow the link_pattern "
                "or switch to web_scrape on specific URLs instead."
            ),
            "urls_crawled": [],
            "items": [],
            "errors": [],
        }


def _sync_extract_structured(url: str, selectors: dict[str, str]) -> dict:
    """Fetch URL and extract using CSS or XPath selectors."""
    try:
        from scrapy.http import HtmlResponse
        from scrapy.selector import Selector
        import urllib.request

        req = urllib.request.Request(url, headers={"User-Agent": "AOF-Agent/0.1"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            body = resp.read()
            encoding = resp.headers.get_content_charset() or "utf-8"

        response = HtmlResponse(url=url, body=body, encoding=encoding)
        sel = Selector(response=response)

        result: dict[str, Any] = {}
        for label, selector in selectors.items():
            if selector.startswith("/") or selector.startswith("("):
                elems = sel.xpath(selector)
                vals = [e.get() for e in elems] if elems else []
            else:
                vals = sel.css(selector).getall()
            if isinstance(vals, list) and len(vals) == 1:
                result[label] = vals[0]
            else:
                result[label] = vals

        return {"url": url, "extracted": result}
    except ImportError:
        return {
            "error": "scrapy not installed. pip install scrapy",
            "url": url,
            "extracted": {},
        }
    except Exception as e:
        logger.warning("Extract failed for %s: %s", url, e)
        return {
            "error": (
                f"Structured extraction failed: {e}. Check that your selectors "
                "match the page, or run web_scrape first to inspect the HTML."
            ),
            "url": url,
            "extracted": {},
        }
