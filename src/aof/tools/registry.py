"""Tool registration, discovery, and invocation."""

from __future__ import annotations

import asyncio
import importlib.util
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)


@dataclass
class ToolDefinition:
    """Describes a tool for the LLM and provides its runtime handler."""

    name: str
    description: str
    parameters: dict  # JSON Schema
    handler: Callable[..., Any]
    source: str = "builtin"  # "builtin" | "script" | "agent-created"


class ToolRegistry:
    """Central registry for all agent-callable tools."""

    def __init__(self) -> None:
        self._tools: dict[str, ToolDefinition] = {}

    def register(self, tool: ToolDefinition) -> None:
        self._tools[tool.name] = tool
        logger.debug("Registered tool: %s (%s)", tool.name, tool.source)

    def register_function(
        self,
        name: str,
        description: str,
        parameters: dict,
        source: str = "builtin",
    ) -> Callable:
        """Decorator for registering a function as a tool."""

        def decorator(fn: Callable) -> Callable:
            self._tools[name] = ToolDefinition(
                name=name,
                description=description,
                parameters=parameters,
                handler=fn,
                source=source,
            )
            return fn

        return decorator

    def unregister(self, name: str) -> None:
        self._tools.pop(name, None)

    def get(self, name: str) -> ToolDefinition | None:
        return self._tools.get(name)

    def list_tools(self) -> list[ToolDefinition]:
        return list(self._tools.values())

    def list_names(self) -> list[str]:
        return list(self._tools.keys())

    def to_prompt_format(self, tool_names: list[str] | None = None) -> list[dict]:
        """Convert tools to the Qwen3 function-calling format for prompts.

        If tool_names is provided, only include those tools.
        """
        tools = self._tools.values()
        if tool_names is not None:
            tools = [t for t in tools if t.name in tool_names]

        return [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                },
            }
            for t in tools
        ]

    async def invoke(self, name: str, arguments: dict[str, Any]) -> Any:
        """Invoke a registered tool by name with the given arguments.

        Handles both sync and async handlers.
        """
        tool = self._tools.get(name)
        if not tool:
            return {"error": f"Unknown tool: {name}"}

        try:
            result = tool.handler(**arguments)
            if asyncio.iscoroutine(result):
                result = await result
            return result
        except TypeError as e:
            return {"error": f"Invalid arguments for {name}: {e}"}
        except Exception as e:
            logger.error("Tool %s failed: %s", name, e)
            return {"error": f"Tool {name} failed: {e}"}

    def load_scripts(self, scripts_dir: Path) -> int:
        """Discover and load agent-created tool scripts from a directory.

        Scripts must have a docstring header with @tool metadata:
            '''
            @tool
            name: tool_name
            description: what it does
            parameters: {"param": {"type": "string"}}
            '''

        Returns the number of scripts loaded.
        """
        count = 0
        if not scripts_dir.exists():
            return count

        for script in scripts_dir.glob("*.py"):
            meta = self._extract_meta(script)
            if meta:
                self.register(
                    ToolDefinition(
                        name=meta["name"],
                        description=meta["description"],
                        parameters=meta["parameters"],
                        handler=self._make_script_handler(script),
                        source="agent-created",
                    )
                )
                count += 1
                logger.info("Loaded agent-created tool: %s from %s", meta["name"], script.name)

        return count

    def filtered(self, tool_names: list[str]) -> ToolRegistry:
        """Return a new registry containing only the named tools."""
        filtered_reg = ToolRegistry()
        for name in tool_names:
            tool = self._tools.get(name)
            if tool:
                filtered_reg.register(tool)
        return filtered_reg

    # -- private ------------------------------------------------------------

    def _extract_meta(self, script: Path) -> dict | None:
        """Extract @tool metadata from a script's docstring."""
        try:
            text = script.read_text(encoding="utf-8")
        except Exception:
            return None

        # Match triple-quoted docstring with @tool marker
        match = re.search(
            r'(?:"""|\'\'\')(.*?@tool.*?)(?:"""|\'\'\')', text, re.DOTALL
        )
        if not match:
            return None

        block = match.group(1)
        meta: dict[str, Any] = {}

        for key in ("name", "description"):
            m = re.search(rf"^{key}:\s*(.+)$", block, re.MULTILINE)
            if m:
                meta[key] = m.group(1).strip()

        # Parameters as JSON
        p_match = re.search(r"parameters:\s*(\{.*\})", block, re.DOTALL)
        if p_match:
            try:
                meta["parameters"] = json.loads(p_match.group(1))
            except json.JSONDecodeError:
                meta["parameters"] = {}
        else:
            meta["parameters"] = {}

        if "name" not in meta or "description" not in meta:
            return None

        return meta

    def _make_script_handler(self, script: Path) -> Callable:
        """Create an async handler that runs the script in a subprocess."""
        from aof.tools.sandbox import Sandbox

        sandbox = Sandbox()

        async def handler(**kwargs: Any) -> Any:
            result = await sandbox.execute_script(script, kwargs)
            if result["success"]:
                try:
                    return json.loads(result["stdout"])
                except json.JSONDecodeError:
                    return result["stdout"]
            return {"error": result.get("stderr", result.get("error", "Unknown error"))}

        return handler
