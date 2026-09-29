"""File system tools for reading, writing, and listing files."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from aof.tools.registry import ToolRegistry


def register(registry: ToolRegistry) -> None:
    """Register file system tools."""

    @registry.register_function(
        name="file_read",
        description="Read the contents of a text file. Returns the text content and file size.",
        parameters={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the file to read",
                },
                "max_chars": {
                    "type": "integer",
                    "description": "Maximum characters to read (default 10000)",
                },
            },
            "required": ["path"],
        },
    )
    async def file_read(path: str, max_chars: int = 10000) -> Any:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, _sync_read, path, max_chars)

    @registry.register_function(
        name="file_write",
        description="Write text content to a file. Creates parent directories if needed.",
        parameters={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to write the file",
                },
                "content": {
                    "type": "string",
                    "description": "Content to write",
                },
            },
            "required": ["path", "content"],
        },
    )
    async def file_write(path: str, content: str) -> Any:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, _sync_write, path, content)

    @registry.register_function(
        name="file_list",
        description="List files and directories at a given path.",
        parameters={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Directory path to list (default: current directory)",
                },
                "pattern": {
                    "type": "string",
                    "description": "Glob pattern to filter (default: '*')",
                },
            },
            "required": [],
        },
    )
    async def file_list(path: str = ".", pattern: str = "*") -> Any:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, _sync_list, path, pattern)


def _sync_read(path: str, max_chars: int) -> dict:
    try:
        p = Path(path)
        if not p.exists():
            return {"error": f"File not found: {path}"}
        if not p.is_file():
            return {"error": f"Not a file: {path}"}
        text = p.read_text(encoding="utf-8", errors="replace")
        truncated = len(text) > max_chars
        return {
            "path": str(p.resolve()),
            "content": text[:max_chars],
            "size": p.stat().st_size,
            "truncated": truncated,
        }
    except Exception as e:
        return {"error": f"Read failed: {e}"}


def _sync_write(path: str, content: str) -> dict:
    try:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return {
            "path": str(p.resolve()),
            "size": p.stat().st_size,
            "written": True,
        }
    except Exception as e:
        return {"error": f"Write failed: {e}"}


def _sync_list(path: str, pattern: str) -> dict:
    try:
        p = Path(path)
        if not p.exists():
            return {"error": f"Path not found: {path}"}
        if not p.is_dir():
            return {"error": f"Not a directory: {path}"}

        entries = []
        for item in sorted(p.glob(pattern))[:100]:  # cap at 100 entries
            entries.append({
                "name": item.name,
                "type": "dir" if item.is_dir() else "file",
                "size": item.stat().st_size if item.is_file() else 0,
            })
        return {"path": str(p.resolve()), "entries": entries, "count": len(entries)}
    except Exception as e:
        return {"error": f"List failed: {e}"}
