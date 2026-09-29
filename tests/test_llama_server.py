"""Tests for the managed llama-server backend and config layering.

The real-model test is opt-in: set AOF_LLAMA_SERVER (binary or its directory) and
AOF_TEST_LARGE_MODEL (path to a GGUF) to run it.
"""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

from aof.config import LlamaServerConfig, load_config, resolve_role_path
from aof.inference.llama_server import (
    ENV_BINARY,
    LlamaServerBackend,
    LlamaServerNotFound,
    find_llama_server,
)


def test_find_llama_server_missing(monkeypatch):
    monkeypatch.delenv(ENV_BINARY, raising=False)
    monkeypatch.setenv("PATH", "")
    with pytest.raises(LlamaServerNotFound, match="docs/llama-cpp.md"):
        find_llama_server("")


def test_find_llama_server_from_env_dir(monkeypatch, tmp_path):
    name = "llama-server.exe" if sys.platform == "win32" else "llama-server"
    exe = tmp_path / name
    exe.write_text("")
    monkeypatch.setenv(ENV_BINARY, str(tmp_path))
    assert find_llama_server("") == str(exe)


def test_local_config_is_merged_over_base(tmp_path):
    (tmp_path / "config.toml").write_text('[llama_server]\nroles = ["a"]\nn_gpu_layers = 10\n')
    (tmp_path / "config.local.toml").write_text('[llama_server]\nroles = ["large"]\n')
    cfg = load_config(tmp_path / "config.toml")
    assert cfg.llama_server.roles == ("large",)
    assert cfg.llama_server.n_gpu_layers == 10


def test_absolute_role_path_wins_over_models_dir(tmp_path):
    model = tmp_path / "big.gguf"
    (tmp_path / "config.toml").write_text(
        f'[models]\ndirectory = "somewhere/else"\n[roles]\nlarge = "{model.as_posix()}"\n'
    )
    cfg = load_config(tmp_path / "config.toml")
    assert os.path.normcase(resolve_role_path(cfg, "large")) == os.path.normcase(str(model))


def test_llama_server_command_splits_context_across_slots():
    server = LlamaServerConfig(n_parallel=4, extra_args=("--flash-attn", "on"))
    cmd = LlamaServerBackend("m.gguf", server, n_ctx=4096)._command("llama-server", 9000)
    assert cmd[cmd.index("-c") + 1] == "16384"
    assert cmd[cmd.index("-np") + 1] == "4"
    assert cmd[-2:] == ["--flash-attn", "on"]


LARGE_MODEL = os.environ.get("AOF_TEST_LARGE_MODEL", "")


@pytest.mark.skipif(
    not (os.environ.get(ENV_BINARY) and LARGE_MODEL and os.path.exists(LARGE_MODEL)),
    reason="set AOF_LLAMA_SERVER and AOF_TEST_LARGE_MODEL to run against a real llama-server",
)
async def test_real_llama_server_answers_without_thinking():
    backend = LlamaServerBackend(LARGE_MODEL, LlamaServerConfig(enable_thinking=False), n_ctx=4096, max_tokens=200)
    await backend.start()
    try:
        pid = backend._proc.pid
        result = await backend.complete(
            [{"role": "user", "content": "Reply with exactly one word: the capital of France."}],
            temperature=0.0,
        )
        assert "paris" in result.text.lower()
        assert len(result.text) < 200  # a direct answer, not untagged chain-of-thought
    finally:
        await backend.shutdown()
    assert backend._proc is None
    if sys.platform == "win32":
        out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}"], capture_output=True, text=True).stdout
        assert str(pid) not in out


_REQUIRES_REAL = pytest.mark.skipif(
    not (os.environ.get(ENV_BINARY) and LARGE_MODEL and os.path.exists(LARGE_MODEL)),
    reason="set AOF_LLAMA_SERVER and AOF_TEST_LARGE_MODEL to run against a real llama-server",
)
_CARD_SAMPLING = dict(top_p=0.95, top_k=20, repeat_penalty=1.05, min_temperature=0.6)


@_REQUIRES_REAL
async def test_real_model_emits_parseable_tool_call_in_aof_prompt_format():
    from aof.inference.parsing import detect_model_family, format_tools_for_prompt, parse_response

    tools = [{
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for a query and return top results.",
            "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
        },
    }]
    family = detect_model_family(LARGE_MODEL)
    system = "You verify claims for a knowledge vault. " + format_tools_for_prompt(tools, family)
    backend = LlamaServerBackend(
        LARGE_MODEL, LlamaServerConfig(enable_thinking=False, **_CARD_SAMPLING), n_ctx=4096, max_tokens=300
    )
    await backend.start()
    try:
        result = await backend.complete([
            {"role": "system", "content": system},
            {"role": "user", "content": "Claim: 'Luhmann kept about 90,000 index cards.' Look up the number of cards."},
        ])
    finally:
        await backend.shutdown()
    parsed = parse_response(result.text)
    assert parsed.tool_calls, f"no tool call parsed from: {result.text!r}"
    assert parsed.tool_calls[0].name == "web_search"
    assert isinstance(parsed.tool_calls[0].arguments.get("query"), str)


@_REQUIRES_REAL
async def test_real_model_thinking_is_separated_from_answer():
    from aof.inference.parsing import parse_response

    backend = LlamaServerBackend(
        LARGE_MODEL, LlamaServerConfig(enable_thinking=True, **_CARD_SAMPLING), n_ctx=4096, max_tokens=3000
    )
    await backend.start()
    try:
        result = await backend.complete(
            [{"role": "user", "content": "In one sentence, what is a Zettelkasten?"}], temperature=0.1
        )
    finally:
        await backend.shutdown()
    parsed = parse_response(result.text)
    assert result.raw["choices"][0]["finish_reason"] == "stop", "thinking exhausted the token budget"
    assert "zettelkasten" in parsed.text.lower()
    assert "</think>" not in parsed.text
    assert len(parsed.text) < 600


def test_per_role_overrides_apply_only_to_that_role(tmp_path):
    (tmp_path / "c.toml").write_text(
        '[llama_server]\nroles = ["small", "large"]\nn_gpu_layers = 50\n'
        '[llama_server.per_role.large]\nmin_temperature = 0.6\ntop_k = 20\nextra_args = ["--flash-attn", "on"]\nbogus = 1\n'
    )
    cfg = load_config(tmp_path / "c.toml").llama_server
    large, small = cfg.for_role("large"), cfg.for_role("small")
    assert (large.min_temperature, large.top_k, large.n_gpu_layers) == (0.6, 20, 50)
    assert large.extra_args == ("--flash-attn", "on")
    assert (small.min_temperature, small.top_k, small.n_gpu_layers) == (0.0, None, 50)


@_REQUIRES_REAL
async def test_real_server_is_relaunched_after_it_dies():
    backend = LlamaServerBackend(LARGE_MODEL, LlamaServerConfig(enable_thinking=False), n_ctx=2048, max_tokens=40)
    await backend.start()
    try:
        first = await backend.complete([{"role": "user", "content": "Reply with one word: the capital of France."}])
        assert "paris" in first.text.lower()
        old_pid = backend._proc.pid
        backend._proc.kill()  # simulate a crash / external kill
        backend._proc.wait()
        second = await backend.complete([{"role": "user", "content": "Reply with one word: the capital of Japan."}])
        assert "tokyo" in second.text.lower()
        assert backend.restarts == 1 and backend._proc.pid != old_pid
    finally:
        await backend.shutdown()
    assert backend._proc is None
