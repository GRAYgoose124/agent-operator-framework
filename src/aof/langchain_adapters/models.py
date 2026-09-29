"""LangChain chat model factory for AOF. Uses local_server (LM Studio / Ollama) by default."""

from __future__ import annotations

from typing import TYPE_CHECKING

from langchain_openai import ChatOpenAI

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

    from aof.config import AppConfig


def create_chat_model(config: AppConfig, role: str = "general") -> BaseChatModel:
    """Create a LangChain chat model from AOF config.

    Uses local_server config for OpenAI-compatible endpoints (LM Studio, Ollama,
    vLLM, etc.). Supports role-based model selection when configured.

    Args:
        config: AOF application config
        role: Agent role for model routing (general, micro, small, fallback, etc.)

    Returns:
        BaseChatModel instance configured for the given role
    """
    local = config.local_server

    # Use local_server for OpenAI-compatible endpoints
    return ChatOpenAI(
        base_url=local.base_url,
        api_key=local.api_key or "not-needed",
        model=local.model,
        temperature=0.7,
        max_tokens=config.model.max_tokens,
    )
