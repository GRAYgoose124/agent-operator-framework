"""Optional adapter for A-MEM (agentic memory) from A-mem-sys.

When agentic_enabled and the agentic-memory package is installed, mirrors
Zettel notes into A-MEM for LLM-generated keywords/context/tags and
evolution. Search can be delegated to A-MEM with fallback to FTS.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from aof.config import MemoryConfig

logger = logging.getLogger(__name__)


def _create_agentic_system(config: MemoryConfig) -> Any | None:
    """Create AgenticMemorySystem from config. Returns None if disabled or import fails."""
    if not getattr(config, "agentic_enabled", False):
        return None
    try:
        from agentic_memory.memory_system import AgenticMemorySystem
    except ImportError:
        logger.warning(
            "Agentic memory enabled but agentic-memory not installed. "
            "Install with: uv sync --extra agentic"
        )
        return None

    model_name = getattr(config, "agentic_embedding_model", None) or config.embedding_model
    llm_backend = getattr(config, "agentic_llm_backend", "ollama") or "ollama"
    llm_model = getattr(config, "agentic_llm_model", "") or "llama2"
    api_key = getattr(config, "agentic_api_key", "") or None

    kwargs: dict[str, Any] = {
        "model_name": model_name,
        "llm_backend": llm_backend,
        "llm_model": llm_model,
    }
    if api_key and llm_backend in ("openai", "openrouter"):
        kwargs["api_key"] = api_key

    try:
        system = AgenticMemorySystem(**kwargs)
        logger.info(
            "Agentic memory (A-MEM) initialized: backend=%s model=%s",
            llm_backend,
            llm_model,
        )
        return system
    except Exception as e:
        logger.warning("Failed to initialize agentic memory: %s", e)
        return None


class AgenticAdapter:
    """Thin wrapper around A-MEM for mirroring notes and search."""

    def __init__(self, config: MemoryConfig) -> None:
        self._config = config
        self._system = _create_agentic_system(config)

    @property
    def enabled(self) -> bool:
        return self._system is not None

    async def add_note(
        self,
        content: str,
        *,
        title: str = "",
        tags: list[str] | None = None,
        note_id: str | None = None,
    ) -> None:
        """Mirror a note into A-MEM. LLM will generate keywords/context if omitted.
        Runs in executor to avoid blocking the event loop.
        """
        if not self._system:
            return
        try:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(
                None,
                lambda: self._add_note_sync(content, title=title, tags=tags),
            )
        except Exception as e:
            logger.debug("Agentic add_note failed (non-fatal): %s", e)

    def _add_note_sync(
        self,
        content: str,
        *,
        title: str = "",
        tags: list[str] | None = None,
    ) -> str | None:
        """Synchronous add for run_in_executor. Returns A-MEM memory id or None."""
        if not self._system:
            return None
        full_content = (f"# {title}\n\n" if title else "") + (content or "").strip()
        if not full_content:
            return None
        try:
            kwargs: dict[str, Any] = {}
            if tags:
                kwargs["tags"] = tags
            if title:
                kwargs["context"] = title
            return self._system.add_note(full_content, **kwargs)
        except Exception as e:
            logger.debug("Agentic add_note failed: %s", e)
            return None

    def search(self, query: str, k: int = 5) -> list[dict[str, Any]]:
        """Search A-MEM; returns list of dicts with id, title, content, tags.
        Empty list if disabled or on error.
        """
        if not self._system or not (query or "").strip():
            return []
        try:
            results = self._system.search(query, k=k)
        except Exception as e:
            logger.debug("Agentic search failed: %s", e)
            return []
        out: list[dict[str, Any]] = []
        for r in results:
            if isinstance(r, dict):
                content = r.get("content", "") or ""
                title = (content.split("\n")[0] or "Untitled").strip().lstrip("#").strip()
                out.append({
                    "id": r.get("id", ""),
                    "title": r.get("title") or title,
                    "content": content,
                    "tags": r.get("tags") or [],
                })
            else:
                content = getattr(r, "content", "") or ""
                title = (content.split("\n")[0] or "Untitled").strip().lstrip("#").strip()
                out.append({
                    "id": getattr(r, "id", ""),
                    "title": getattr(r, "title", None) or title,
                    "content": content,
                    "tags": getattr(r, "tags", None) or [],
                })
        return out

    def search_agentic(self, query: str, k: int = 5) -> list[dict[str, Any]]:
        """Agentic search when available; same return shape as search."""
        if not self._system or not (query or "").strip():
            return []
        try:
            results = self._system.search_agentic(query, k=k)
        except Exception as e:
            logger.debug("Agentic search_agentic failed: %s", e)
            return self.search(query, k)
        out: list[dict[str, Any]] = []
        for r in results:
            if isinstance(r, dict):
                content = r.get("content", "") or ""
                title = (content.split("\n")[0] or "Untitled").strip().lstrip("#").strip()
                out.append({
                    "id": r.get("id", ""),
                    "title": r.get("title") or title,
                    "content": content,
                    "tags": r.get("tags") or [],
                })
            else:
                content = getattr(r, "content", "") or ""
                title = (content.split("\n")[0] or "Untitled").strip().lstrip("#").strip()
                out.append({
                    "id": getattr(r, "id", ""),
                    "title": getattr(r, "title", None) or title,
                    "content": content,
                    "tags": getattr(r, "tags", None) or [],
                })
        return out
