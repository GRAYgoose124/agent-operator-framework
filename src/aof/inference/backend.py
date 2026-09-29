"""Abstract inference backend protocol."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable


@dataclass
class CompletionResult:
    """Structured result from an inference call."""

    text: str
    tokens_used: int
    finish_reason: str
    raw: dict[str, Any] | None = None


@runtime_checkable
class InferenceBackend(Protocol):
    """Protocol that all inference backends must satisfy.

    Implementations:
      - LlamaBackend: llama-cpp-python with model pool + thread dispatch
      - OpenAIBackend: aiohttp client for vLLM / Ollama / LM Studio
    """

    async def start(self) -> None:
        """Load models and prepare for inference."""
        ...

    async def complete(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        stop: list[str] | None = None,
    ) -> CompletionResult:
        """Run chat completion and return structured result."""
        ...

    async def complete_json(
        self,
        messages: list[dict[str, str]],
        schema: dict | None = None,
    ) -> dict:
        """Run chat completion constrained to JSON output.

        If the backend supports grammar/schema constraints, use them.
        Otherwise, parse JSON from the text output with fallback extraction.
        """
        ...

    async def shutdown(self) -> None:
        """Release model resources."""
        ...

    def model_info(self) -> dict[str, Any]:
        """Return metadata about the loaded model."""
        ...
