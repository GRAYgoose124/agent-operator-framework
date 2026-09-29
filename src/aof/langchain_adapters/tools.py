"""Convert AOF ToolRegistry to LangChain BaseTool instances."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from langchain_core.tools import StructuredTool

if TYPE_CHECKING:
    from aof.tools.registry import ToolRegistry


def tool_registry_to_langchain(
    registry: ToolRegistry,
    tool_names: list[str] | None = None,
) -> list[StructuredTool]:
    """Convert ToolRegistry definitions to LangChain tools.

    Args:
        registry: AOF tool registry with registered tools
        tool_names: Optional list of tool names to include; None = all tools

    Returns:
        List of LangChain StructuredTool instances
    """
    tools = registry.list_tools()
    if tool_names is not None:
        tools = [t for t in tools if t.name in tool_names]

    result: list[StructuredTool] = []
    for defn in tools:
        tool = _make_tool(registry, defn)
        result.append(tool)
    return result


def _make_tool(registry: ToolRegistry, defn: Any) -> StructuredTool:
    """Create a StructuredTool from a ToolDefinition."""
    name = defn.name
    description = defn.description
    parameters = defn.parameters

    # Ensure parameters has required structure for JSON schema
    args_schema = parameters if isinstance(parameters, dict) else {"type": "object", "properties": {}}

    async def _invoke(**kwargs: Any) -> Any:
        return await registry.invoke(name, kwargs)

    return StructuredTool.from_function(
        coroutine=_invoke,
        name=name,
        description=description,
        args_schema=args_schema,
        infer_schema=False,
    )
