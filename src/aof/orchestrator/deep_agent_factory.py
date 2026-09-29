"""Factory for creating AOF Deep Agents with LangChain/LangGraph."""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import TYPE_CHECKING

from deepagents import create_deep_agent
from deepagents.backends import CompositeBackend, StateBackend, StoreBackend
from langgraph.checkpoint.memory import MemorySaver

from aof.langchain_adapters.models import create_chat_model
from aof.langchain_adapters.memory_store import ZettelStoreAdapter
from aof.langchain_adapters.tools import tool_registry_to_langchain

if TYPE_CHECKING:
    from aof.config import AppConfig
    from aof.memory.store import MemoryStore
    from aof.tools.registry import ToolRegistry


def create_aof_deep_agent(
    config: AppConfig,
    tools: ToolRegistry,
    memory: MemoryStore,
    *,
    system_prompt: str | None = None,
    tool_names: list[str] | None = None,
    role: str = "general",
) -> object:
    """Create a Deep Agent configured for AOF with Zettelkasten memory.

    Args:
        config: AOF application config
        tools: AOF tool registry (web_search, scraper, etc.)
        memory: Initialized MemoryStore (Zettelkasten)
        system_prompt: Optional custom system prompt
        tool_names: Optional list of tool names to include; None = all
        role: Model role for routing

    Returns:
        Compiled LangGraph StateGraph (invoke/stream as usual)
    """
    model = create_chat_model(config, role=role)
    langchain_tools = tool_registry_to_langchain(tools, tool_names=tool_names)

    zettel_store = ZettelStoreAdapter(memory)

    def make_backend(rt):
        return CompositeBackend(
            default=StateBackend(rt),
            routes={"/memories/": StoreBackend(rt)},
        )

    default_system_prompt = (
        "You are a focused, efficient agent. Complete tasks step by step. "
        "Use tools when they help accomplish the task. "
        "Be concise. Save important findings to /memories/ for long-term storage. "
        "When using a tool, call it with the correct arguments."
    )

    agent = create_deep_agent(
        model=model,
        tools=langchain_tools,
        system_prompt=system_prompt or default_system_prompt,
        backend=make_backend,
        store=zettel_store,
        checkpointer=MemorySaver(),
    )
    return agent


def create_aof_pipeline_agent(
    config: AppConfig,
    tools: ToolRegistry,
    memory: MemoryStore,
    pipeline_path: str | Path,
) -> object:
    """Create a Deep Agent configured from a pipeline TOML (all tools, combined prompt)."""
    with open(pipeline_path, "rb") as f:
        data = tomllib.load(f)
    steps = data.get("steps", [])
    all_tool_names: list[str] = []
    for step in steps:
        all_tool_names.extend(step.get("tools", []))
    tool_names = list(dict.fromkeys(all_tool_names))  # unique, preserve order
    prompts = [s.get("system_prompt", "") for s in steps if s.get("system_prompt")]
    combined_prompt = (
        "You execute a multi-step pipeline. "
        + " ".join(prompts)
        + " First gather information, then synthesize into a clear final report."
    )
    return create_aof_deep_agent(
        config=config,
        tools=tools,
        memory=memory,
        system_prompt=combined_prompt,
        tool_names=tool_names if tool_names else None,
    )
