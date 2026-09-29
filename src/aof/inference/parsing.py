"""Structured output parsing for small models (Qwen3 + LFM2.5 tool call formats).

Supported formats:
  - Qwen3: <tool_call>{"name": ..., "arguments": ...}</tool_call>
  - LFM2.5: <|tool_call_start|>[func(arg=val)]<|tool_call_end|>  (Pythonic)
  - LFM2.5: <|tool_call_start|>{"name": ..., "arguments": ...}<|tool_call_end|>  (JSON mode)
  - Bare JSON: {"name": ..., "arguments": ...} without tags
"""

from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass, field


@dataclass
class ToolCall:
    """A single tool invocation parsed from model output."""

    name: str
    arguments: dict


@dataclass
class ParsedResponse:
    """Parsed model output with extracted tool calls and thinking."""

    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    thinking: str | None = None


# -- public API -------------------------------------------------------------


def parse_response(raw: str) -> ParsedResponse:
    """Parse model output, extracting tool calls and thinking blocks.

    Handles:
      - <think>...</think> reasoning blocks (Qwen3 thinking mode)
      - <tool_call>{"name": ..., "arguments": ...}</tool_call> (Qwen3 tool format)
      - Fallback JSON extraction for malformed output
    """
    thinking = _extract_thinking(raw)
    tool_calls = _extract_tool_calls(raw)
    clean = _strip_tags(raw)
    return ParsedResponse(text=clean, tool_calls=tool_calls, thinking=thinking)


def extract_json(text: str) -> dict:
    """Extract a JSON object from text, with fallback for noisy output.

    Tries in order:
      1. Direct parse of the entire text
      2. Find the outermost {...} and parse that
      3. Return {"raw": text} as last resort
    """
    stripped = text.strip()

    # Try direct parse
    try:
        parsed = json.loads(stripped)
        if isinstance(parsed, list):
            return {"steps": parsed}
        return parsed
    except (json.JSONDecodeError, ValueError):
        pass

    # Try extracting outermost braces
    match = re.search(r"\{.*\}", stripped, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group())
            if isinstance(parsed, list):
                return {"steps": parsed}
            return parsed
        except (json.JSONDecodeError, ValueError):
            pass

    # Try extracting from markdown code blocks
    md_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", stripped, re.DOTALL)
    if md_match:
        try:
            parsed = json.loads(md_match.group(1))
            if isinstance(parsed, list):
                return {"steps": parsed}
            return parsed
        except (json.JSONDecodeError, ValueError):
            pass

    return {"raw": stripped}


def format_tools_for_prompt(tools: list[dict], model_family: str = "qwen3") -> str:
    """Format tool definitions for the model's expected tool call format.

    Args:
        tools: List of tool definitions, each with keys:
               type, function.name, function.description, function.parameters
        model_family: "qwen3" for XML format, "lfm2" for LFM2.5 JSON format

    Qwen3 uses: <tools>[...]</tools> + <tool_call>{...}</tool_call>
    LFM2.5 uses: JSON tool defs + <|tool_call_start|>...<|tool_call_end|>
    """
    tools_json = json.dumps(tools, indent=2)
    if model_family == "lfm2":
        # LFM2.5: instruct to use JSON tool call format for reliable parsing
        return (
            f"You have access to the following tools:\n{tools_json}\n\n"
            "To call a tool, respond with a JSON object inside tool call tags:\n"
            '<|tool_call_start|>{"name": "tool_name", "arguments": {"key": "value"}}<|tool_call_end|>'
        )
    # Qwen3: include tool list and explicit response format so the model emits tool calls
    return (
        f"<tools>\n{tools_json}\n</tools>\n\n"
        'To call a tool, respond with: <tool_call>{"name": "tool_name", "arguments": {...}}</tool_call>'
    )


def detect_model_family(model_path: str) -> str:
    """Detect model family from the GGUF file path.

    Returns:
        "lfm2" for LiquidAI LFM2.5 models
        "qwen3" for Qwen3 models (default)
    """
    path_lower = model_path.lower()
    if "lfm2" in path_lower or "lfm-2" in path_lower or "liquid" in path_lower:
        return "lfm2"
    return "qwen3"


# -- private helpers --------------------------------------------------------

_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)
_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
# LFM2.5 tool call format: <|tool_call_start|>...<|tool_call_end|>
_LFM_TOOL_CALL_RE = re.compile(
    r"<\|tool_call_start\|>(.*?)<\|tool_call_end\|>", re.DOTALL
)
# LFM2.5 Pythonic call: [func_name(key="value", ...)]
_LFM_PYTHONIC_RE = re.compile(r"\[(\w+)\((.*?)\)\]", re.DOTALL)
_TAG_RE = re.compile(
    r"</?(?:think|tool_call|tools)>|<\|(?:tool_call_start|tool_call_end|im_start|im_end|startoftext)\|>",
    re.DOTALL,
)


def _extract_thinking(text: str) -> str | None:
    match = _THINK_RE.search(text)
    return match.group(1).strip() if match else None


def _extract_tool_calls(text: str) -> list[ToolCall]:
    """Extract tool calls from model output.

    Supports multiple formats:
      - Qwen3:  <tool_call>{"name": ..., "arguments": ...}</tool_call>
      - LFM2.5: <|tool_call_start|>[func(args)]<|tool_call_end|>  (Pythonic)
      - LFM2.5: <|tool_call_start|>{"name": ..., ...}<|tool_call_end|>  (JSON)
      - Bare JSON: {"name": ..., "arguments": ...} without tags
    """
    calls: list[ToolCall] = []

    # 1. Qwen3 format: <tool_call>{...}</tool_call>
    for match in _TOOL_CALL_RE.finditer(text):
        raw_json = match.group(1)
        try:
            data = json.loads(raw_json)
            name = data.get("name", "")
            arguments = data.get("arguments", data.get("parameters", {}))
            if name:
                calls.append(ToolCall(name=name, arguments=arguments))
        except (json.JSONDecodeError, ValueError):
            fuzzy = _fuzzy_parse_tool_call(raw_json)
            if fuzzy:
                calls.append(fuzzy)

    # 2. LFM2.5 format: <|tool_call_start|>...<|tool_call_end|>
    for match in _LFM_TOOL_CALL_RE.finditer(text):
        raw = match.group(1).strip()
        parsed = _parse_lfm_tool_call(raw)
        if parsed:
            calls.append(parsed)

    # 3. Fallback: bare tool call JSON without XML tags
    if not calls:
        calls.extend(_extract_bare_tool_calls(text))
    return calls


def _fuzzy_parse_tool_call(text: str) -> ToolCall | None:
    """Attempt to extract tool call from malformed JSON."""
    # Try to find name field
    name_match = re.search(r'"name"\s*:\s*"([^"]+)"', text)
    if not name_match:
        return None
    name = name_match.group(1)

    # Try to find arguments
    args_match = re.search(r'"arguments"\s*:\s*(\{.*?\})', text, re.DOTALL)
    if args_match:
        try:
            arguments = json.loads(args_match.group(1))
        except (json.JSONDecodeError, ValueError):
            arguments = {}
    else:
        arguments = {}

    return ToolCall(name=name, arguments=arguments)


def _extract_bare_tool_calls(text: str) -> list[ToolCall]:
    """Look for JSON objects that look like tool calls outside of XML tags."""
    calls: list[ToolCall] = []
    # Pattern: {"name": "tool_name", "arguments": {...}}
    pattern = r'\{"name"\s*:\s*"[^"]+"\s*,\s*"arguments"\s*:\s*\{.*?\}\s*\}'
    for match in re.finditer(pattern, text, re.DOTALL):
        try:
            data = json.loads(match.group())
            calls.append(ToolCall(name=data["name"], arguments=data.get("arguments", {})))
        except (json.JSONDecodeError, ValueError, KeyError):
            pass
    return calls


def _parse_lfm_tool_call(raw: str) -> ToolCall | None:
    """Parse an LFM2.5 tool call (Pythonic or JSON format).

    Pythonic: [func_name(key="value", key2=123)]
    JSON: {"name": "func_name", "arguments": {...}}
    """
    # Try JSON first
    try:
        data = json.loads(raw)
        name = data.get("name", "")
        arguments = data.get("arguments", data.get("parameters", {}))
        if name:
            return ToolCall(name=name, arguments=arguments)
    except (json.JSONDecodeError, ValueError):
        pass

    # Try Pythonic format: [func_name(key="value", ...)]
    match = _LFM_PYTHONIC_RE.search(raw)
    if match:
        func_name = match.group(1)
        args_str = match.group(2).strip()
        arguments = _parse_pythonic_args(args_str)
        return ToolCall(name=func_name, arguments=arguments)

    # Try bare function-like: func_name(key="value")
    bare = re.match(r"(\w+)\((.*)\)", raw.strip(), re.DOTALL)
    if bare:
        func_name = bare.group(1)
        args_str = bare.group(2).strip()
        arguments = _parse_pythonic_args(args_str)
        return ToolCall(name=func_name, arguments=arguments)

    return None


def _parse_pythonic_args(args_str: str) -> dict:
    """Parse Python-style keyword arguments: key="value", key2=123."""
    if not args_str:
        return {}
    try:
        # Use ast.literal_eval on a dict expression
        dict_expr = "{" + re.sub(r"(\w+)\s*=", r'"\1":', args_str) + "}"
        return ast.literal_eval(dict_expr)
    except (ValueError, SyntaxError):
        pass
    # Fallback: regex-based extraction
    result = {}
    for m in re.finditer(r'(\w+)\s*=\s*("(?:[^"\\]|\\.)*"|\'(?:[^\'\\]|\\.)*\'|[^,\)]+)', args_str):
        key = m.group(1)
        val = m.group(2).strip().strip("'\"")
        # Try to parse as number/bool
        try:
            result[key] = ast.literal_eval(val)
        except (ValueError, SyntaxError):
            result[key] = val
    return result


def _strip_tags(text: str) -> str:
    """Remove XML-style tags and tool call blocks from display text."""
    # Remove think blocks entirely
    result = _THINK_RE.sub("", text)
    # Remove Qwen3 tool call blocks
    result = _TOOL_CALL_RE.sub("", result)
    # Remove LFM2.5 tool call blocks
    result = _LFM_TOOL_CALL_RE.sub("", result)
    # Remove remaining tags
    result = _TAG_RE.sub("", result)
    return result.strip()
