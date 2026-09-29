"""Tests for structured output parsing (Qwen3 + LFM2.5 formats)."""

from aof.inference.parsing import (
    ParsedResponse,
    ToolCall,
    detect_model_family,
    extract_json,
    format_tools_for_prompt,
    parse_response,
)


def test_parse_clean_text():
    result = parse_response("Hello, world!")
    assert result.text == "Hello, world!"
    assert result.tool_calls == []
    assert result.thinking is None


def test_parse_thinking_block():
    raw = "<think>Let me consider this...</think>The answer is 42."
    result = parse_response(raw)
    assert result.thinking == "Let me consider this..."
    assert "42" in result.text
    assert "<think>" not in result.text


def test_parse_tool_call():
    raw = '<tool_call>{"name": "web_search", "arguments": {"query": "python"}}</tool_call>'
    result = parse_response(raw)
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].name == "web_search"
    assert result.tool_calls[0].arguments == {"query": "python"}


def test_parse_tool_call_with_thinking():
    raw = (
        '<think>I need to search for this.</think>'
        'Let me search. '
        '<tool_call>{"name": "web_search", "arguments": {"query": "test"}}</tool_call>'
    )
    result = parse_response(raw)
    assert result.thinking == "I need to search for this."
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].name == "web_search"
    assert "<tool_call>" not in result.text
    assert "<think>" not in result.text


def test_parse_multiple_tool_calls():
    raw = (
        '<tool_call>{"name": "tool_a", "arguments": {"x": 1}}</tool_call>'
        '<tool_call>{"name": "tool_b", "arguments": {"y": 2}}</tool_call>'
    )
    result = parse_response(raw)
    assert len(result.tool_calls) == 2
    assert result.tool_calls[0].name == "tool_a"
    assert result.tool_calls[1].name == "tool_b"


def test_parse_malformed_tool_call():
    # Missing closing brace — fuzzy parser should still extract name
    raw = '<tool_call>{"name": "broken_tool", "arguments": {"k": "v"}</tool_call>'
    result = parse_response(raw)
    # May or may not extract depending on regex, but shouldn't crash
    assert isinstance(result, ParsedResponse)


def test_extract_json_clean():
    assert extract_json('{"key": "value"}') == {"key": "value"}


def test_extract_json_with_noise():
    text = 'Here is the result: {"steps": ["step1", "step2"]} and more text.'
    result = extract_json(text)
    assert result == {"steps": ["step1", "step2"]}


def test_extract_json_markdown_code_block():
    text = '```json\n{"color": "blue"}\n```'
    result = extract_json(text)
    assert result == {"color": "blue"}


def test_extract_json_fallback():
    result = extract_json("not json at all")
    assert "raw" in result


def test_format_tools_for_prompt():
    tools = [
        {
            "type": "function",
            "function": {
                "name": "test_tool",
                "description": "A test",
                "parameters": {"type": "object"},
            },
        }
    ]
    formatted = format_tools_for_prompt(tools)
    assert "<tools>" in formatted
    assert "test_tool" in formatted
    assert "</tools>" in formatted


def test_format_tools_lfm2():
    tools = [
        {
            "type": "function",
            "function": {
                "name": "web_search",
                "description": "Search the web",
                "parameters": {"type": "object"},
            },
        }
    ]
    formatted = format_tools_for_prompt(tools, model_family="lfm2")
    assert "<tools>" not in formatted
    assert "tool_call_start" in formatted
    assert "web_search" in formatted


# --- LFM2.5 tool call format tests ---


def test_parse_lfm2_json_tool_call():
    """LFM2.5 JSON format: <|tool_call_start|>{...}<|tool_call_end|>"""
    raw = '<|tool_call_start|>{"name": "web_search", "arguments": {"query": "python"}}<|tool_call_end|>'
    result = parse_response(raw)
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].name == "web_search"
    assert result.tool_calls[0].arguments == {"query": "python"}
    assert "<|tool_call_start|>" not in result.text


def test_parse_lfm2_pythonic_tool_call():
    """LFM2.5 Pythonic format: <|tool_call_start|>[func(args)]<|tool_call_end|>"""
    raw = '<|tool_call_start|>[web_search(query="machine learning")]<|tool_call_end|>'
    result = parse_response(raw)
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].name == "web_search"
    assert result.tool_calls[0].arguments == {"query": "machine learning"}


def test_parse_lfm2_pythonic_multiple_args():
    raw = '<|tool_call_start|>[numpy_stats(values="1,2,3", operation="mean")]<|tool_call_end|>'
    result = parse_response(raw)
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].name == "numpy_stats"
    assert result.tool_calls[0].arguments["operation"] == "mean"


def test_parse_lfm2_with_text():
    """LFM2.5 tool call mixed with regular text."""
    raw = (
        "Let me search for that. "
        '<|tool_call_start|>{"name": "web_search", "arguments": {"query": "test"}}<|tool_call_end|>'
    )
    result = parse_response(raw)
    assert len(result.tool_calls) == 1
    assert "search for that" in result.text
    assert "<|tool_call_start|>" not in result.text


def test_detect_model_family_qwen():
    assert detect_model_family("Qwen3-0.6B-Q4_K_M.gguf") == "qwen3"
    assert detect_model_family("path/to/Qwen3-4B-Instruct.gguf") == "qwen3"


def test_detect_model_family_lfm2():
    assert detect_model_family("LFM2.5-1.2B-Instruct-Q8_0.gguf") == "lfm2"
    assert detect_model_family("path/to/LFM2.5-1.2B-Thinking-Q8_0.gguf") == "lfm2"


def test_detect_model_family_unknown():
    assert detect_model_family("some-other-model.gguf") == "qwen3"  # default


def test_qwen35_native_function_call():
    from aof.inference.parsing import parse_response

    raw = (
        "Checking.\n<tool_call>\n<function=web_search>\n<parameter=query>\nLuhmann index cards\n</parameter>\n"
        "<parameter=max_results>\n5\n</parameter>\n</function>\n</tool_call>"
    )
    parsed = parse_response(raw)
    assert [(c.name, c.arguments) for c in parsed.tool_calls] == [
        ("web_search", {"query": "Luhmann index cards", "max_results": 5})
    ]
    assert parsed.text == "Checking."


def test_lone_closing_think_tag_is_split_off():
    from aof.inference.parsing import parse_response

    parsed = parse_response("The user wants a definition.\nLet me recall...\n</think>\n\nA Zettelkasten is a note system.")
    assert parsed.text == "A Zettelkasten is a note system."
    assert "The user wants" in parsed.thinking


def test_detect_qwen35_family():
    from aof.inference.parsing import detect_model_family

    assert detect_model_family("models/Qwythos-9B-Claude-Mythos-5-1M-Q4_K_M.gguf") == "qwen35"
    assert detect_model_family("Qwen3.5-4B-Q4.gguf") == "qwen35"
    assert detect_model_family("Qwen3-8B-Q4_K_M.gguf") == "qwen3"
