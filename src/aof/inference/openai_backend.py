"""Local server inference backend (LM Studio, Ollama — OpenAI-compatible API, fully local)."""

from __future__ import annotations

import json
import logging
from typing import Any

import aiohttp

from aof.config import LocalServerConfig
from aof.inference.backend import CompletionResult, InferenceBackend
from aof.inference.parsing import extract_json

logger = logging.getLogger(__name__)


class LocalServerBackend:
    """Async HTTP client for local OpenAI-compatible servers.

    Works with:
      - LM Studio (local OpenAI-compatible endpoint)
      - Ollama (true parallel requests via OLLAMA_NUM_PARALLEL)

    Both expose the same /v1/chat/completions endpoint locally.
    """

    def __init__(
        self,
        config: LocalServerConfig,
        *,
        n_ctx: int = 2048,
        max_tokens: int = 512,
    ) -> None:
        self._config = config
        self._n_ctx = n_ctx
        self._max_tokens = max_tokens
        self._session: aiohttp.ClientSession | None = None

    async def start(self) -> None:
        self._session = aiohttp.ClientSession(
            headers={
                "Authorization": f"Bearer {self._config.api_key}",
                "Content-Type": "application/json",
            },
            timeout=aiohttp.ClientTimeout(total=self._config.timeout),
        )
        logger.info("OpenAI backend ready: %s (model=%s)", self._config.base_url, self._config.model)

    async def shutdown(self) -> None:
        if self._session:
            await self._session.close()
            self._session = None

    async def complete(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        stop: list[str] | None = None,
    ) -> CompletionResult:
        assert self._session is not None, "Call start() first"

        payload: dict[str, Any] = {
            "model": self._config.model,
            "messages": messages,
            "temperature": temperature if temperature is not None else 0.7,
            "max_tokens": max_tokens or self._max_tokens,
        }
        if stop:
            payload["stop"] = stop
        if self._config.enable_thinking is not None:
            payload["chat_template_kwargs"] = {"enable_thinking": self._config.enable_thinking}

        url = f"{self._config.base_url}/chat/completions"

        async with self._session.post(url, json=payload) as resp:
            if resp.status != 200:
                error_text = await resp.text()
                raise RuntimeError(f"OpenAI API error {resp.status}: {error_text}")

            data = await resp.json()

        choice = data["choices"][0]
        return CompletionResult(
            text=choice["message"]["content"] or "",
            tokens_used=data.get("usage", {}).get("total_tokens", 0),
            finish_reason=choice.get("finish_reason", "unknown"),
            raw=data,
        )

    async def complete_json(
        self,
        messages: list[dict[str, str]],
        schema: dict | None = None,
    ) -> dict:
        # Add JSON constraint hint
        json_messages = list(messages)
        hint = "\n\nRespond with ONLY valid JSON. No markdown, no explanation."
        if schema:
            hint += f"\nRequired schema: {json.dumps(schema)}"
        if json_messages and json_messages[-1]["role"] == "user":
            json_messages[-1] = {
                **json_messages[-1],
                "content": json_messages[-1]["content"] + hint,
            }
        else:
            json_messages.append({"role": "user", "content": hint})

        result = await self.complete(json_messages, temperature=0.1)
        # Strip <think>...</think> tags before JSON extraction (Qwen3 thinking models)
        from aof.inference.parsing import parse_response

        clean_text = parse_response(result.text).text
        return extract_json(clean_text)

    def model_info(self) -> dict[str, Any]:
        return {
            "backend": "openai-compatible",
            "base_url": self._config.base_url,
            "model": self._config.model,
            "n_ctx": self._n_ctx,
            "max_tokens": self._max_tokens,
        }
