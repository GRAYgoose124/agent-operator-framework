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
