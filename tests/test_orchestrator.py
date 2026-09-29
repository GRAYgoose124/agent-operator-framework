"""Tests for orchestration layer (Deep Agent factory)."""

import tempfile
from pathlib import Path

import pytest

from aof.config import MemoryConfig, load_config
from aof.memory.store import MemoryStore
from aof.orchestrator.deep_agent_factory import create_aof_deep_agent, create_aof_pipeline_agent
from aof.tools.registry import ToolRegistry


@pytest.mark.asyncio
async def test_create_aof_deep_agent():
    """create_aof_deep_agent returns a compiled LangGraph."""
    config = load_config()
    with tempfile.TemporaryDirectory() as tmpdir:
        mem_config = MemoryConfig(
            db_path=str(Path(tmpdir) / "mem.db"),
            notes_dir=str(Path(tmpdir) / "notes"),
        )
        memory = MemoryStore(mem_config)
        await memory.initialize()

        tools = ToolRegistry()
        agent = create_aof_deep_agent(config, tools, memory)
        assert agent is not None
        assert hasattr(agent, "invoke")
        await memory.close()


@pytest.mark.asyncio
async def test_create_aof_pipeline_agent():
    """create_aof_pipeline_agent returns a compiled LangGraph from pipeline TOML."""
    config = load_config()
    with tempfile.TemporaryDirectory() as tmpdir:
        mem_config = MemoryConfig(
            db_path=str(Path(tmpdir) / "mem.db"),
            notes_dir=str(Path(tmpdir) / "notes"),
        )
        memory = MemoryStore(mem_config)
        await memory.initialize()

        tools = ToolRegistry()
        pipeline_path = Path(__file__).parent.parent / "examples" / "research.toml"
        if pipeline_path.exists():
            agent = create_aof_pipeline_agent(config, tools, memory, pipeline_path)
            assert agent is not None
            assert hasattr(agent, "invoke")
        await memory.close()
