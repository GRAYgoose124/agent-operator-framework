"""Agent-callable tool for creating new Scrapy crawlers."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from aof.tools.registry import ToolRegistry
from aof.tools.sandbox import Sandbox

if TYPE_CHECKING:
    from aof.config import AppConfig

logger = logging.getLogger(__name__)

GENERIC_TEMPLATE = '''"""Agent-created crawler for {domain}."""
import scrapy


class SiteCrawlerSpider(scrapy.Spider):
    name = "crawl_{site_slug}"
    allowed_domains = ["{domain}"]
    start_urls = ["{start_url}"]
    custom_settings = {{"CONCURRENT_REQUESTS": 2, "DOWNLOAD_DELAY": 0.5, "ROBOTSTXT_OBEY": True}}

    def parse(self, response):
        yield {{"url": response.url, "title": response.css("title::text").get("") or "", "content": " ".join(t.strip() for t in response.css("body *::text").getall() if t.strip())[:3000]}}
        for href in response.css("a::attr(href)").getall():
            if href and (href.startswith(("http://", "https://", "/"))):
                yield response.follow(href, self.parse, errback=lambda _: None)
'''


def register(registry: ToolRegistry, crawlers_dir: str | Path = "data/crawlers") -> None:
    """Register crawler creation and update tools."""

    @registry.register_function(
        name="create_crawler",
        description="Create a new Scrapy crawler for a site. Use when you need a site-specific crawler. Saves to data/crawlers/.",
        parameters={
            "type": "object",
            "properties": {
                "site": {
                    "type": "string",
                    "description": "Domain or site (e.g. example.com)",
                },
                "template": {
                    "type": "string",
                    "description": "Template: generic, sitemap, or playwright",
                    "enum": ["generic", "sitemap", "playwright"],
                },
                "start_url": {
                    "type": "string",
                    "description": "Optional start URL override",
                },
            },
            "required": ["site"],
        },
    )
    async def create_crawler(
        site: str,
        template: str = "generic",
        start_url: str | None = None,
    ) -> dict:
        try:
            site = site.strip().lower()
            site_slug = site.replace(".", "_").replace("-", "_")
            if not site_slug.replace("_", "").isalnum():
                return {"error": "Site must be alphanumeric with dots/dashes"}

            domain = site if "." in site else f"{site}.com"
            url = start_url or f"https://{domain}/"
            crawlers_path = Path(crawlers_dir)
            crawlers_path.mkdir(parents=True, exist_ok=True)
            path = crawlers_path / f"site_{site_slug}.py"

            if path.exists():
                return {"error": f"Crawler already exists: {path}", "path": str(path)}

            if template == "generic":
                code = GENERIC_TEMPLATE.format(
                    domain=domain,
                    site_slug=site_slug,
                    start_url=url,
                )
            else:
                return {"error": f"Template '{template}' not yet supported; use generic"}

            sandbox = Sandbox()
            issues = sandbox.validate_safety(code)
            if issues:
                return {"error": f"Validation failed: {issues}", "path": None}

            path.write_text(code, encoding="utf-8")
            logger.info("Created crawler: %s", path)
            return {"success": True, "path": str(path), "tool_name": f"crawl_site_{site_slug}"}
        except Exception as e:
            logger.warning("create_crawler failed: %s", e)
            return {"error": str(e), "path": None}

    @registry.register_function(
        name="propose_crawler_update",
        description=(
            "Propose an update to a Scrapy crawler for a site. Use this when crawling "
            "a site yields empty content, repeated errors, or clearly outdated "
            "selectors. Saves a markdown proposal under data/crawlers/proposals/ for "
            "later review and application."
        ),
        parameters={
            "type": "object",
            "properties": {
                "site": {
                    "type": "string",
                    "description": "Domain or site (e.g. example.com) the crawler targets.",
                },
                "reason": {
                    "type": "string",
                    "description": (
                        "Why an update is needed (e.g. empty content, blocked by robots, "
                        "layout changed, missing sections)."
                    ),
                },
                "instructions": {
                    "type": "string",
                    "description": (
                        "Proposed change(s): new CSS/XPath selectors, allowed_domains, "
                        "throttling settings, Playwright usage, etc."
                    ),
                },
                "crawler_path": {
                    "type": "string",
                    "description": (
                        "Optional explicit crawler file path if known. If omitted, "
                        "the tool infers site_<slug>.py under the crawlers directory."
                    ),
                },
            },
            "required": ["site", "reason", "instructions"],
        },
    )
    async def propose_crawler_update(
        site: str,
        reason: str,
        instructions: str,
        crawler_path: str | None = None,
    ) -> dict:
        """Write a crawler update proposal markdown file for later review."""
        from datetime import datetime

        try:
            site = (site or "").strip().lower()
            if not site:
                return {"error": "Site cannot be empty."}

            site_slug = site.replace(".", "_").replace("-", "_")
            if not site_slug.replace("_", "").isalnum():
                return {"error": "Site must be alphanumeric with dots/dashes"}

            crawlers_path = Path(crawlers_dir)
            crawlers_path.mkdir(parents=True, exist_ok=True)

            if crawler_path:
                base_path = Path(crawler_path)
                if not base_path.is_absolute():
                    base_path = crawlers_path / base_path
            else:
                base_path = crawlers_path / f"site_{site_slug}.py"

            exists = base_path.exists()

            proposals_dir = crawlers_path / "proposals"
            proposals_dir.mkdir(parents=True, exist_ok=True)

            ts = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
            stem = base_path.stem or f"site_{site_slug}"
            proposal_path = proposals_dir / f"{stem}_update_{ts}.md"

            content_lines = [
                "# Crawler update proposal",
                "",
                f"- Site: {site}",
                f"- Inferred crawler_path: {base_path}",
                f"- Crawler exists: {'yes' if exists else 'no'}",
                "",
                "## Reason",
                "",
                reason.strip(),
                "",
                "## Proposed changes",
                "",
                instructions.strip(),
                "",
                "_This file was generated by propose_crawler_update; apply changes "
                "after review and testing._",
                "",
            ]
            proposal_path.write_text("\n".join(content_lines), encoding="utf-8")
            logger.info("Wrote crawler update proposal: %s", proposal_path)
            return {
                "success": True,
                "proposal_path": str(proposal_path),
                "crawler_path": str(base_path),
                "crawler_exists": exists,
            }
        except Exception as e:
            logger.warning("propose_crawler_update failed: %s", e)
            return {"error": str(e)}
