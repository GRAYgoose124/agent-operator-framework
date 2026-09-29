"""DuckDuckGo web search tool with retry, backoff, and TTL cache."""

from __future__ import annotations

import asyncio
import logging
import random
import time
from typing import Any

from aof.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)

# In-memory TTL cache: (query, max_results) -> (timestamp, result)
_search_cache: dict[tuple[str, int], tuple[float, Any]] = {}
_news_cache: dict[tuple[str, int], tuple[float, Any]] = {}
_CACHE_TTL = 300  # 5 minutes


def _cache_get(cache: dict, key: tuple) -> Any | None:
    """Return cached result if within TTL, else None."""
    entry = cache.get(key)
    if entry is None:
        return None
    ts, result = entry
    if time.monotonic() - ts > _CACHE_TTL:
        del cache[key]
        return None
    return result


def _cache_set(cache: dict, key: tuple, result: Any) -> None:
    cache[key] = (time.monotonic(), result)


def _retry_with_backoff(func, *args, max_attempts: int = 3):
    """Call func with exponential backoff + jitter on failure."""
    for attempt in range(max_attempts):
        try:
            return func(*args)
        except Exception as e:
            if attempt == max_attempts - 1:
                raise
            delay = (2 ** attempt) + random.uniform(0, 1)
            logger.warning(
                "Search attempt %d/%d failed: %s. Retrying in %.1fs",
                attempt + 1, max_attempts, e, delay,
            )
            time.sleep(delay)


def register(registry: ToolRegistry) -> None:
    """Register web search tools."""

    @registry.register_function(
        name="web_search",
        description=(
            "General-purpose DuckDuckGo web search. Use for background facts, "
            "documentation, and static pages. Returns a list of results with "
            "title, url, and snippet. After using this tool, you will usually "
            "pick 1-3 promising URLs to inspect more closely with web_scrape "
            "or crawl_site."
        ),
        parameters={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "The search query. Use neutral, descriptive keywords; "
                        "for time-sensitive topics prefer web_news."
                    ),
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results (default 5).",
                },
            },
            "required": ["query"],
        },
    )
    async def web_search(query: str, max_results: int = 5) -> Any:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, _sync_search, query, max_results)

    @registry.register_function(
        name="web_news",
        description=(
            "Search recent news using DuckDuckGo. Use this for time-sensitive "
            "or breaking topics (markets, politics, product launches) and use "
            "web_search for timeless background information. Returns news "
            "articles with title, url, date, source, and snippet."
        ),
        parameters={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "The news search query. Include time qualifiers when "
                        "useful (e.g. 'in 2024', 'last month')."
                    ),
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results (default 5).",
                },
            },
            "required": ["query"],
        },
    )
    async def web_news(query: str, max_results: int = 5) -> Any:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, _sync_news, query, max_results)


def _sync_search(query: str, max_results: int) -> list[dict] | dict:
    cache_key = (query, max_results)
    cached = _cache_get(_search_cache, cache_key)
    if cached is not None:
        return cached
    try:
        from duckduckgo_search import DDGS

        def _do_search():
            results = DDGS().text(query, max_results=max_results)
            return [
                {
                    "title": r.get("title", ""),
                    "url": r.get("href", ""),
                    "snippet": r.get("body", ""),
                }
                for r in results
            ]

        result = _retry_with_backoff(_do_search)
        if not result:
            logger.warning("Web search returned no results for query=%r (rate limit or empty DDG response)", query[:80])
            # Return a clear hint so the agent can try rephrasing or web_news
            hint = [
                {
                    "title": "No results",
                    "url": "",
                    "snippet": "Try rephrasing the query or using web_news for time-sensitive topics. DuckDuckGo may return empty for some queries or regions.",
                }
            ]
            return hint
        # Hint to downstream agents about how to continue the research flow.
        for item in result:
            if isinstance(item, dict) and "url" in item:
                item.setdefault(
                    "suggested_next_step",
                    "Consider scraping 1-3 of these URLs with web_scrape for deeper content.",
                )
        _cache_set(_search_cache, cache_key, result)
        return result
    except ImportError:
        logger.warning("duckduckgo-search not installed for query=%r", query[:80])
        return {"error": "duckduckgo-search not installed. pip install duckduckgo-search"}
    except Exception as e:
        logger.warning("Web search failed after retries for query=%r: %s", query[:80], e)
        return {"error": f"Search failed: {e}"}


def _sync_news(query: str, max_results: int) -> list[dict] | dict:
    cache_key = (query, max_results)
    cached = _cache_get(_news_cache, cache_key)
    if cached is not None:
        return cached
    try:
        from duckduckgo_search import DDGS

        def _do_news():
            results = DDGS().news(query, max_results=max_results)
            return [
                {
                    "title": r.get("title", ""),
                    "url": r.get("url", ""),
                    "date": r.get("date", ""),
                    "snippet": r.get("body", ""),
                    "source": r.get("source", ""),
                }
                for r in results
            ]

        result = _retry_with_backoff(_do_news)
        if not result:
            logger.warning("Web news returned no results for query=%r", query[:80])
            return [
                {
                    "title": "No results",
                    "url": "",
                    "date": "",
                    "snippet": "Try rephrasing the query or using web_search for general background.",
                    "source": "",
                }
            ]
        # Hint for follow-up actions when dealing with time-sensitive topics.
        for item in result:
            if isinstance(item, dict) and "url" in item:
                item.setdefault(
                    "suggested_next_step",
                    (
                        "For deeper context, consider scraping key articles with "
                        "web_scrape or running a broader web_search on important entities."
                    ),
                )
        _cache_set(_news_cache, cache_key, result)
        return result
    except ImportError:
        logger.warning("duckduckgo-search not installed for news query=%r", query[:80])
        return {"error": "duckduckgo-search not installed. pip install duckduckgo-search"}
    except Exception as e:
        logger.warning("News search failed after retries for query=%r: %s", query[:80], e)
        return {"error": f"News search failed: {e}"}
