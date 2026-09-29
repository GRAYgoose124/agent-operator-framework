"""Sentence-transformers embedding provider (the framework's default embedding path)."""

from __future__ import annotations

import asyncio
import logging

from aof.specialists.base import EMBED, SpecialistResult

logger = logging.getLogger(__name__)


class SentenceTransformerProvider:
    capabilities = (EMBED,)

    def __init__(self, model: str = "all-MiniLM-L6-v2") -> None:
        self.name = "sentence-transformers"
        self._model_name = model
        self._model = None
        self._lock = asyncio.Lock()

    def available(self) -> bool:
        try:
            import sentence_transformers  # noqa: F401
        except ImportError:
            return False
        return True

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self._model_name)
        return self._model

    async def embed(self, texts: list[str]) -> SpecialistResult | None:
        async with self._lock:
            try:
                vectors = await asyncio.to_thread(
                    lambda: self._load().encode(texts, normalize_embeddings=True).tolist()
                )
            except Exception as e:  # e.g. model download failed while offline
                logger.warning("sentence-transformers embed failed: %s", e)
                return None
        return SpecialistResult(vectors, self.name)

    async def close(self) -> None:
        return None
