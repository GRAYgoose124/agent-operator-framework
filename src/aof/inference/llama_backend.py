"""llama-cpp-python inference backend with model pool for concurrent async access."""

from __future__ import annotations

import asyncio
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from llama_cpp import Llama

from aof.config import ModelConfig, PoolConfig
from aof.inference.backend import CompletionResult, InferenceBackend
from aof.inference.parsing import extract_json

logger = logging.getLogger(__name__)


class LlamaBackend:
    """Pool of Llama model instances behind asyncio semaphore + thread pool.

    Each Qwen3-0.6B Q4_K_M instance is ~462 MB. The pool_size determines
    how many can run truly in parallel (typically 2-4 depending on RAM).
    """

    def __init__(self, model_config: ModelConfig, pool_config: PoolConfig) -> None:
        self._model_config = model_config
        self._pool_size = pool_config.inference_pool_size
        self._executor = ThreadPoolExecutor(
            max_workers=pool_config.thread_pool_workers
        )
        self._models: list[Llama] = []
        self._slots: asyncio.Queue[int] = asyncio.Queue()
        self._started = False

    # -- lifecycle ----------------------------------------------------------

    async def start(self) -> None:
        if self._started:
            return
        loop = asyncio.get_running_loop()
        logger.info(
            "Loading %d model instance(s) from %s",
            self._pool_size,
            self._model_config.path,
        )
        futs = [
            loop.run_in_executor(self._executor, self._load_model)
            for _ in range(self._pool_size)
        ]
        models = await asyncio.gather(*futs)
        self._models = list(models)
        for i in range(self._pool_size):
            self._slots.put_nowait(i)
        self._started = True
        logger.info("Inference engine ready (%d slots)", self._pool_size)

    async def shutdown(self) -> None:
        self._executor.shutdown(wait=False)
        self._models.clear()
        self._started = False

    # -- inference ----------------------------------------------------------

    async def complete(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        stop: list[str] | None = None,
    ) -> CompletionResult:
        slot = await self._slots.get()
        try:
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(
                self._executor,
                self._sync_complete,
                slot,
                messages,
                temperature,
                max_tokens,
                stop,
            )
            return result
        finally:
            self._slots.put_nowait(slot)

    async def complete_json(
        self,
        messages: list[dict[str, str]],
        schema: dict | None = None,
    ) -> dict:
        # Append JSON instruction to the last user message
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

        result = await self.complete(
            json_messages, temperature=0.1, max_tokens=self._model_config.max_tokens
        )
        # Strip <think>...</think> tags before JSON extraction (Qwen3 thinking models)
        from aof.inference.parsing import parse_response

        clean_text = parse_response(result.text).text
        return extract_json(clean_text)

    def model_info(self) -> dict[str, Any]:
        return {
            "backend": "llama-cpp-python",
            "model_path": self._model_config.path,
            "n_ctx": self._model_config.n_ctx,
            "pool_size": self._pool_size,
            "chat_format": self._model_config.chat_format,
        }

    # -- private ------------------------------------------------------------

    def _load_model(self) -> Llama:
        return Llama(
            model_path=self._model_config.path,
            n_ctx=self._model_config.n_ctx,
            n_threads=self._model_config.n_threads,
            n_gpu_layers=self._model_config.n_gpu_layers,
            chat_format=self._model_config.chat_format,
            verbose=False,
        )

    def _sync_complete(
        self,
        slot: int,
        messages: list[dict[str, str]],
        temperature: float | None,
        max_tokens: int | None,
        stop: list[str] | None,
    ) -> CompletionResult:
        """Blocking call executed in the thread pool."""
        model = self._models[slot]
        kwargs: dict[str, Any] = {}
        if temperature is not None:
            kwargs["temperature"] = temperature
        else:
            kwargs["temperature"] = self._model_config.temperature
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        else:
            kwargs["max_tokens"] = self._model_config.max_tokens
        if stop:
            kwargs["stop"] = stop

        # Overflow protection: estimate prompt tokens and truncate if needed
        effective_max_tokens = kwargs["max_tokens"]
        n_ctx = self._model_config.n_ctx
        # Reserve tokens for generation + chat template overhead (~10 tokens per message)
        template_overhead = len(messages) * 10
        max_prompt_tokens = n_ctx - effective_max_tokens - template_overhead
        if max_prompt_tokens > 0:
            total_chars = sum(
                len(m.get("content", "")) for m in messages
                if isinstance(m.get("content"), str)
            )
            estimated_tokens = total_chars // 3  # conservative 3 chars/token
            if estimated_tokens > max_prompt_tokens:
                messages = self._truncate_for_fit(messages, max_prompt_tokens)
                logger.warning(
                    "Overflow protection: truncated prompt from ~%d to ~%d est. tokens "
                    "(n_ctx=%d, max_tokens=%d)",
                    estimated_tokens, max_prompt_tokens, n_ctx, effective_max_tokens,
                )

        try:
            response = model.create_chat_completion(messages=messages, **kwargs)
        except Exception as e:
            err = str(e)
            if "llama_decode" in err or "n_ctx" in err:
                logger.error("Context overflow in llama_decode: %s", err)

                # Second-chance retry: aggressively truncate the longest user message
                # and try once more before surfacing an error to the caller.
                try:
                    if max_prompt_tokens > 0:
                        reduced_tokens = max(128, max_prompt_tokens // 2)
                        retry_messages = self._truncate_for_fit(
                            messages, reduced_tokens
                        )
                        logger.warning(
                            "Retrying after context overflow with stricter prompt cap "
                            "~%d tokens (n_ctx=%d)",
                            reduced_tokens,
                            n_ctx,
                        )
                        response = model.create_chat_completion(
                            messages=retry_messages, **kwargs
                        )
                        choice = response["choices"][0]
                        return CompletionResult(
                            text=choice["message"]["content"] or "",
                            tokens_used=response.get("usage", {}).get(
                                "total_tokens", 0
                            ),
                            finish_reason=choice.get("finish_reason", "unknown"),
                            raw=response,
                        )
                except Exception as retry_err:
                    # If the retry also fails, fall through and surface an error
                    # message with the latest underlying exception.
                    err = str(retry_err)
                    logger.error(
                        "Context overflow retry failed in llama_decode: %s", err
                    )

                return CompletionResult(
                    text=f"[Error: context overflow — prompt too large for {n_ctx}-token window]",
                    tokens_used=0,
                    finish_reason="error",
                    raw={"error": err},
                )
            raise

        choice = response["choices"][0]
        return CompletionResult(
            text=choice["message"]["content"] or "",
            tokens_used=response.get("usage", {}).get("total_tokens", 0),
            finish_reason=choice.get("finish_reason", "unknown"),
            raw=response,
        )

    def _truncate_for_fit(
        self, messages: list[dict[str, Any]], max_prompt_tokens: int
    ) -> list[dict[str, Any]]:
        """Truncate the longest user message so total fits within max_prompt_tokens."""
        max_chars = max_prompt_tokens * 3
        messages = [dict(m) for m in messages]

        # Find the longest user message to truncate
        longest_idx = -1
        longest_len = 0
        for i, m in enumerate(messages):
            content = m.get("content", "")
            if m.get("role") == "user" and isinstance(content, str) and len(content) > longest_len:
                longest_len = len(content)
                longest_idx = i

        if longest_idx < 0:
            return messages

        total_chars = sum(
            len(m.get("content", "")) for m in messages
            if isinstance(m.get("content"), str)
        )
        excess = total_chars - max_chars
        if excess <= 0:
            return messages

        content = messages[longest_idx]["content"]
        new_len = max(100, len(content) - excess)
        messages[longest_idx]["content"] = content[:new_len] + "\n[...truncated]"
        return messages
