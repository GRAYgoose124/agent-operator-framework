"""Registry for discovering and loading agent-created Scrapy crawlers."""

from __future__ import annotations

import importlib.util
import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import scrapy

logger = logging.getLogger(__name__)


class CrawlerRegistry:
    """Discover Scrapy spiders from data/crawlers/ and expose them as tools."""

    def __init__(self, crawlers_dir: Path) -> None:
        self._dir = Path(crawlers_dir)
        self._spiders: dict[str, type] = {}

    def discover(self) -> list[str]:
        """Find all spider modules in crawlers_dir. Returns list of spider names."""
        self._spiders.clear()
        if not self._dir.exists():
            return []

        for path in self._dir.glob("*.py"):
            if path.name.startswith("_"):
                continue
            try:
                spec = importlib.util.spec_from_file_location(
                    f"crawler_{path.stem}", path
                )
                if spec and spec.loader:
                    mod = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(mod)
                    import scrapy
                    for attr in dir(mod):
                        obj = getattr(mod, attr)
                        if (isinstance(obj, type) and
                                hasattr(obj, "name") and
                                issubclass(obj, scrapy.Spider)):
                            name = getattr(obj, "name", path.stem)
                            tool_name = f"crawl_{path.stem}"
                            self._spiders[tool_name] = obj
                            logger.debug("Discovered crawler: %s", tool_name)
            except Exception as e:
                logger.warning("Failed to load crawler %s: %s", path, e)

        return list(self._spiders.keys())

    def get_spider(self, name: str, /) -> type | None:
        """Get spider class by tool name (e.g. crawl_site_example_com)."""
        return self._spiders.get(name)

    def list_names(self) -> list[str]:
        """List all discovered crawler tool names."""
        return list(self._spiders.keys())
