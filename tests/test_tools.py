"""Tests for tool registry and sandbox."""

import asyncio
import tempfile
from pathlib import Path

import pytest

from aof.tools.registry import ToolDefinition, ToolRegistry
from aof.tools.sandbox import Sandbox


def test_register_and_invoke():
    registry = ToolRegistry()

    async def add(a: int, b: int) -> int:
        return a + b

    registry.register(
        ToolDefinition(
            name="add",
            description="Add two numbers",
            parameters={"type": "object", "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}}},
            handler=add,
        )
    )

    assert registry.get("add") is not None
    assert "add" in registry.list_names()


@pytest.mark.asyncio
async def test_invoke_tool():
    registry = ToolRegistry()

    async def multiply(x: int, y: int) -> int:
        return x * y

    registry.register(
        ToolDefinition(
            name="multiply",
            description="Multiply",
            parameters={},
            handler=multiply,
        )
    )

    result = await registry.invoke("multiply", {"x": 3, "y": 4})
    assert result == 12


@pytest.mark.asyncio
async def test_invoke_unknown_tool():
    registry = ToolRegistry()
    result = await registry.invoke("nonexistent", {})
    assert isinstance(result, dict)
    assert "error" in result


def test_to_prompt_format():
    registry = ToolRegistry()
    registry.register(
        ToolDefinition(
            name="test_tool",
            description="A test tool",
            parameters={"type": "object"},
            handler=lambda: None,
        )
    )

    fmt = registry.to_prompt_format()
    assert len(fmt) == 1
    assert fmt[0]["function"]["name"] == "test_tool"


def test_filtered_registry():
    registry = ToolRegistry()
    for name in ["a", "b", "c"]:
        registry.register(
            ToolDefinition(name=name, description=name, parameters={}, handler=lambda: None)
        )

    filtered = registry.filtered(["a", "c"])
    assert "a" in filtered.list_names()
    assert "c" in filtered.list_names()
    assert "b" not in filtered.list_names()


def test_sandbox_safety_validation():
    sandbox = Sandbox()

    # Safe code
    assert sandbox.validate_safety("x = 1 + 2\nprint(x)") == []

    # Unsafe: os import
    issues = sandbox.validate_safety("import os\nos.system('rm -rf /')")
    assert len(issues) > 0

    # Unsafe: eval
    issues = sandbox.validate_safety("result = eval(input())")
    assert len(issues) > 0

    # Unsafe: subprocess
    issues = sandbox.validate_safety("from subprocess import run")
    assert len(issues) > 0


@pytest.mark.asyncio
async def test_sandbox_execute_script():
    sandbox = Sandbox(timeout=10)

    # Create a simple test script
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, encoding="utf-8"
    ) as f:
        f.write('import sys, json\nargs = json.loads(sys.stdin.read())\nresult = args["a"] + args["b"]\nprint(json.dumps({"sum": result}))\n')
        f.flush()
        script_path = Path(f.name)

    result = await sandbox.execute_script(script_path, {"a": 3, "b": 4})
    assert result["success"]
    import json
    output = json.loads(result["stdout"])
    assert output["sum"] == 7

    script_path.unlink()


@pytest.mark.asyncio
async def test_sandbox_timeout():
    sandbox = Sandbox(timeout=2)

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, encoding="utf-8"
    ) as f:
        f.write("import time\ntime.sleep(10)\n")
        f.flush()
        script_path = Path(f.name)

    result = await sandbox.execute_script(script_path, {})
    assert not result["success"]
    assert "timeout" in result.get("error", "").lower() or not result["success"]

    script_path.unlink()
