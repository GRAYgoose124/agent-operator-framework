"""Context window builder with token budget management for small models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from aof.inference.parsing import format_tools_for_prompt


def _normalize_image_url(url: str) -> str:
    """Normalize image URL for multimodal models. Supports file:// and data URLs."""
    if url.startswith("file://"):
        return url
    if url.startswith("data:"):
        return url
    if "://" not in url:
        return f"file://{url}"
    return url


@dataclass
class ContextBudget:
    """Token budget allocation for context window segments."""

    total: int  # Total context size (e.g. 2048)
    system: int  # System prompt + tools
    memory: int  # Retrieved memory notes
    history: int  # Conversation history
    task: int  # Final user message (task / goal)
    generation: int  # Reserved for model output

    HISTORY_MIN_TOKENS = 256

    @classmethod
    def from_total(
        cls,
        total: int,
        max_tokens: int = 512,
        task_reserve: int = 384,
    ) -> ContextBudget:
        """Allocate budget so system+memory+history+task+generation <= total.

        task_reserve is the token budget for the final user message. For thinking
        models (large generation reserve), use a smaller task_reserve so input fits.
        """
        generation = max_tokens
        task_reserve_actual = min(task_reserve, max(0, total - generation))
        remaining = total - generation - task_reserve_actual
        system = min(384, remaining // 3)
        memory = remaining // 4
        history = remaining - system - memory
        if history < cls.HISTORY_MIN_TOKENS and remaining >= cls.HISTORY_MIN_TOKENS:
            need = cls.HISTORY_MIN_TOKENS - history
            system_cap = max(128, system - need // 2)
            memory_cap = max(64, memory - (need - (system - system_cap)))
            system = system_cap
            memory = memory_cap
            history = remaining - system - memory
        return cls(
            total=total,
            system=system,
            memory=memory,
            history=history,
            task=task_reserve_actual,
            generation=generation,
        )


class ContextBuilder:
    """Assembles a messages list within a token budget.

    Uses a character-based heuristic (chars_per_token) to estimate token count
    without coupling to a specific model tokenizer. Default 4; some models use ~3.
    """

    CHARS_PER_TOKEN = 3

    def __init__(
        self,
        budget: ContextBudget,
        model_family: str = "qwen3",
        chars_per_token: int | None = None,
    ) -> None:
        self._budget = budget
        self._model_family = model_family
        self._chars_per_token = chars_per_token if chars_per_token is not None else self.CHARS_PER_TOKEN

    def build(
        self,
        system_prompt: str,
        task: str,
        *,
        tools: list[dict] | None = None,
        memory_notes: list[str] | None = None,
        history: list[dict[str, Any]] | None = None,
        guidance: str | None = None,
        image_urls: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Build a messages list within the token budget.

        Supports multimodal (VL) models: pass image_urls for vision tasks.
        image_urls: list of file paths, file:// URLs, or data: URLs.

        Priority (last to be truncated):
          1. System prompt + task (always included)
          2. Tool definitions (if any)
          3. Recent history (most recent kept)
          4. Memory notes (most relevant first)
          5. Evaluator guidance (if agent is underperforming)
        """
        messages: list[dict[str, Any]] = []

        # 1. System message: prompt + tools + guidance (task goes in final user message)
        # When tools are present, keep the full tools block (format + list) and truncate only
        # the non-tools part so the model always sees how to call tools.
        if tools:
            tools_text = format_tools_for_prompt(tools, model_family=self._model_family)
            tools_tokens = self.estimate_tokens(tools_text) + 1  # +1 for leading newline
            budget_for_prompt = max(0, self._budget.system - tools_tokens)
            prompt_part = system_prompt
            if guidance:
                prompt_part += f"\nIMPORTANT: {guidance}"
            system_content = self._fit(prompt_part, budget_for_prompt) + "\n" + tools_text
        else:
            system_parts = [system_prompt]
            if guidance:
                system_parts.append(f"\nIMPORTANT: {guidance}")
            system_content = self._fit("\n".join(system_parts), self._budget.system)
        messages.append({"role": "system", "content": system_content})

        # 2. Memory context
        if memory_notes:
            memory_text = "\n---\n".join(memory_notes)
            memory_content = self._fit(
                f"Relevant context from memory:\n{memory_text}",
                self._budget.memory,
            )
            messages.append({"role": "user", "content": memory_content})
            messages.append(
                {"role": "assistant", "content": "I've reviewed the context. Ready to proceed."}
            )

        # 3. Conversation history (keep most recent turns)
        if history:
            trimmed = self._trim_history(history, self._budget.history)
            messages.extend(trimmed)

        # 4. Final user message: task (invocation) so backends always have a clear current turn
        if image_urls:
            content_parts: list[dict[str, Any]] = []
            for url in image_urls:
                content_parts.append({
                    "type": "image_url",
                    "image_url": {"url": _normalize_image_url(url)},
                })
            content_parts.append({"type": "text", "text": task})
            messages.append({"role": "user", "content": content_parts})
        else:
            task_content = self._fit(task, self._budget.task)
            messages.append({"role": "user", "content": task_content})

        return messages

    def _fit(self, text: str, token_budget: int) -> str:
        """Truncate text to fit within token budget."""
        if token_budget <= 0:
            return ""
        max_chars = token_budget * self._chars_per_token
        if len(text) <= max_chars:
            return text
        return text[: max_chars - 3] + "..."

    def _msg_content_len(self, msg: dict[str, Any]) -> int:
        """Character count for a message (handles text or multimodal content list)."""
        content = msg.get("content", "")
        if isinstance(content, str):
            return len(content)
        if isinstance(content, list):
            return sum(
                len(part.get("text", "")) for part in content
                if isinstance(part, dict) and part.get("type") == "text"
            )
        return 0

    def _trim_history(
        self, history: list[dict[str, Any]], token_budget: int
    ) -> list[dict[str, Any]]:
        """Keep the most recent messages that fit within budget."""
        max_chars = token_budget * self._chars_per_token
        result: list[dict[str, Any]] = []
        chars_used = 0

        for msg in reversed(history):
            msg_chars = self._msg_content_len(msg)
            if chars_used + msg_chars > max_chars:
                break
            result.insert(0, msg)
            chars_used += msg_chars

        return result

    def estimate_tokens(self, text: str) -> int:
        """Rough token count estimate."""
        return len(text) // self._chars_per_token
