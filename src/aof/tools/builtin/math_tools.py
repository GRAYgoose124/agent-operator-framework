"""NumPy-based math and statistics tools."""

from __future__ import annotations

import asyncio
from typing import Any

from aof.tools.registry import ToolRegistry


def register(registry: ToolRegistry) -> None:
    """Register math/statistics tools."""

    @registry.register_function(
        name="numpy_stats",
        description="Compute descriptive statistics on a list of numbers: mean, std, median, min, max, sum, count.",
        parameters={
            "type": "object",
            "properties": {
                "numbers": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "List of numbers to analyze",
                },
            },
            "required": ["numbers"],
        },
    )
    async def numpy_stats(numbers: list[float]) -> Any:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, _sync_stats, numbers)

    @registry.register_function(
        name="numpy_calc",
        description="Evaluate a mathematical expression. Supports standard math operations: +, -, *, /, **, sqrt, sin, cos, log, abs, pi, e.",
        parameters={
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Mathematical expression to evaluate, e.g. 'sqrt(144) + 3**2'",
                },
            },
            "required": ["expression"],
        },
    )
    async def numpy_calc(expression: str) -> Any:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, _sync_calc, expression)


def _sync_stats(numbers: list[float]) -> dict:
    try:
        import numpy as np

        arr = np.array(numbers, dtype=float)
        return {
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "median": float(np.median(arr)),
            "min": float(arr.min()),
            "max": float(arr.max()),
            "sum": float(arr.sum()),
            "count": len(arr),
        }
    except Exception as e:
        return {"error": f"Stats computation failed: {e}"}


def _sync_calc(expression: str) -> dict:
    """Evaluate a math expression safely using numpy."""
    try:
        import numpy as np

        # Only allow safe names
        allowed = {
            "sqrt": np.sqrt, "abs": np.abs, "sin": np.sin, "cos": np.cos,
            "tan": np.tan, "log": np.log, "log2": np.log2, "log10": np.log10,
            "exp": np.exp, "pi": np.pi, "e": np.e, "ceil": np.ceil,
            "floor": np.floor, "round": round, "sum": sum, "min": min, "max": max,
        }
        # Restricted eval with only math functions
        result = eval(expression, {"__builtins__": {}}, allowed)  # noqa: S307
        return {"expression": expression, "result": float(result)}
    except Exception as e:
        return {"error": f"Calculation failed: {e}"}
