"""Needle (cactus-needle) provider: tiny native model for fielded extraction and classification.

Notes from measurement (see docs/VAULT_OVERHAUL.md):
  * Extraction is *span-level*: use it for fields (subject, topic, entities), not for writing claims.
  * The engine withholds calls it is unsure about (`suppressed_calls`); we treat that as abstaining.
  * Confidence scales differ per generation, so no numeric threshold is applied.
  * Needle 3 embeddings barely discriminate semantically; embed is opt-in, never in a default chain.
  * The binary reports telemetry unless NEEDLE_TELEMETRY=0 and DO_NOT_TRACK=1; we set both.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, Sequence

from aof.specialists.base import ANNOTATE, CLASSIFY, EMBED, SpecialistResult, label_schema, validate_fields

logger = logging.getLogger(__name__)


def _import_needle():
    os.environ.setdefault("NEEDLE_TELEMETRY", "0")
    os.environ.setdefault("DO_NOT_TRACK", "1")
    import needle

    return needle


class NeedleProvider:
    """`needle:<generation>` provider (generation 3 preferred, 2 as fallback)."""

    capabilities = (ANNOTATE, CLASSIFY, EMBED)

    def __init__(self, generation: int = 3) -> None:
        self.generation = generation
        self.name = f"needle:{generation}"
        self._needle: Any = None
        self._lock = asyncio.Lock()  # the native engine is not re-entrant
        self._checked: bool | None = None

    def available(self) -> bool:
        if self._checked is None:
            try:
                self._needle = _import_needle()
                self._checked = True
            except Exception as e:  # ImportError, or a platform without an engine
                logger.info("Needle unavailable (%s: %s)", type(e).__name__, e)
                self._checked = False
        return self._checked

    def _complete(self, text: str, schema: dict) -> dict:
        agent = self._needle.Needle(tools=[schema], generation=self.generation, stateless=True)
        try:
            return agent.complete(text)
        finally:
            agent.close()

    async def annotate(self, text: str, schema: dict) -> SpecialistResult | None:
        async with self._lock:
            try:
                response = await asyncio.to_thread(self._complete, text, schema)
            except Exception as e:
                logger.warning("%s failed (%s: %s); escalating", self.name, type(e).__name__, str(e)[:120])
                return None
        calls = response.get("function_calls") or []
        if not calls:
            return None  # withheld: the engine is not confident enough to act
        fields = calls[0].get("arguments") or {}
        if not validate_fields(schema, fields):
            return None
        return SpecialistResult(fields, self.name, confidence=response.get("confidence"))

    async def classify(self, text: str, labels: Sequence[str], *, description: str = "") -> SpecialistResult | None:
        schema = label_schema(labels, description or "Pick the single best label for the text.")
        result = await self.annotate(text, schema)
        if result is None:
            return None
        return SpecialistResult(result.value["label"], self.name, result.confidence)

    async def embed(self, texts: list[str]) -> SpecialistResult | None:
        if self.generation < 3:
            return None  # embeddings require a Needle 3 model
        async with self._lock:
            def _run() -> list[list[float]]:
                agent = self._needle.Needle(generation=3)
                try:
                    return [agent.embed(t) for t in texts]
                finally:
                    agent.close()

            try:
                return SpecialistResult(await asyncio.to_thread(_run), self.name)
            except Exception as e:
                logger.warning("%s embed failed: %s", self.name, e)
                return None

    async def close(self) -> None:
        return None
