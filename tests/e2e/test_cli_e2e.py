"""End-to-end tests for CLI commands."""

from __future__ import annotations

import pytest

from tests.conftest import run_cli


def test_cli_version():
    """aof --version"""
    result = run_cli(["--version"])
    assert result.returncode == 0
    assert "aof" in result.stdout
    assert "0.1" in result.stdout or "version" in result.stdout.lower()


def test_cli_tools_list():
    """aof tools list — assert output contains expected tools."""
    result = run_cli(["tools", "list"])
    assert result.returncode == 0
    output = result.stdout + result.stderr
    assert "web_search" in output or "web_scrape" in output


@pytest.mark.asyncio
async def test_cli_memory_search(temp_dir, temp_config):
    """aof memory search 'test' — use temp config, pre-populate notes."""
    from aof.config import load_config
    from aof.memory.store import MemoryStore

    # Write temp config (use forward slashes for TOML path compatibility)
    config_path = temp_dir / "test_config.toml"
    db_path = (temp_dir / "memory.db").as_posix()
    notes_dir = (temp_dir / "memory").as_posix()
    queue_path = (temp_dir / "research_queue.json").as_posix()
    config_path.write_text(
        f"""[memory]
db_path = "{db_path}"
notes_dir = "{notes_dir}"
memory_strategy = "search"
memory_search_limit = 5

[research]
queue_path = "{queue_path}"
""",
        encoding="utf-8",
    )

    # Pre-populate memory
    config = load_config(config_path)
    memory = MemoryStore(config.memory)
    await memory.initialize()
    await memory.create_note(
        title="CLI Test",
        content="This is a test note for CLI search.",
        tags=["test"],
        agent_id="cli-e2e",
    )
    await memory.close()

    config_arg = config_path.resolve().as_posix()
    result = run_cli(
        ["memory", "--config", config_arg, "search", "CLI Test"],
    )
    assert result.returncode == 0
    output = result.stdout + result.stderr
    assert "CLI Test" in output
    assert "This is a test note" in output or "test note" in output


def test_cli_research_add_list(temp_dir):
    """aof research add 'Q' then aof research list."""
    config_path = temp_dir / "test_config.toml"
    db_path = (temp_dir / "memory.db").as_posix()
    notes_dir = (temp_dir / "memory").as_posix()
    queue_path = (temp_dir / "research_queue.json").as_posix()
    config_path.write_text(
        f"""[memory]
db_path = "{db_path}"
notes_dir = "{notes_dir}"

[research]
queue_path = "{queue_path}"
""",
        encoding="utf-8",
    )

    add_result = run_cli(["--config", str(config_path), "research", "add", "What is Python?"])
    assert add_result.returncode == 0

    list_result = run_cli(["--config", str(config_path), "research", "list"])
    assert list_result.returncode == 0
    output = list_result.stdout + list_result.stderr
    assert "Python" in output or "backlog" in output.lower()


def test_cli_discovery_add_list(temp_dir):
    """aof discovery add 'seed' then aof discovery list."""
    config_path = temp_dir / "test_config.toml"
    db_path = (temp_dir / "memory.db").as_posix()
    notes_dir = (temp_dir / "memory").as_posix()
    research_path = (temp_dir / "research_queue.json").as_posix()
    discovery_path = (temp_dir / "discovery_queue.json").as_posix()
    config_path.write_text(
        f"""[memory]
db_path = "{db_path}"
notes_dir = "{notes_dir}"

[research]
queue_path = "{research_path}"

[discovery]
queue_path = "{discovery_path}"
""",
        encoding="utf-8",
    )

    add_result = run_cli(["--config", str(config_path), "discovery", "add", "Topic: AI agents"])
    assert add_result.returncode == 0

    list_result = run_cli(["--config", str(config_path), "discovery", "list"])
    assert list_result.returncode == 0
    output = list_result.stdout + list_result.stderr
    assert "AI agents" in output or "backlog" in output.lower()
