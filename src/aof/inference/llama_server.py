"""Managed llama.cpp `llama-server` backend.

Spawns a local `llama-server` for one GGUF model and talks to it over its OpenAI-compatible API.
Use it for models `llama-cpp-python` cannot load (newer architectures) or to run against an upstream
llama.cpp build. See docs/llama-cpp.md.
"""

from __future__ import annotations

import asyncio
import atexit
import logging
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import aiohttp

from aof.config import LlamaServerConfig, LocalServerConfig
from aof.inference.backend import CompletionResult
from aof.inference.openai_backend import LocalServerBackend

logger = logging.getLogger(__name__)

ENV_BINARY = "AOF_LLAMA_SERVER"


class LlamaServerNotFound(RuntimeError):
    """Raised when no llama-server binary can be located."""


def find_llama_server(configured: str = "") -> str:
    """Locate llama-server: explicit config, then $AOF_LLAMA_SERVER, then PATH."""
    for candidate in (configured, os.environ.get(ENV_BINARY, "")):
        if candidate:
            path = Path(candidate).expanduser()
            if path.is_dir():
                path = path / ("llama-server.exe" if sys.platform == "win32" else "llama-server")
            if path.exists():
                return str(path)
    found = shutil.which("llama-server")
    if found:
        return found
    raise LlamaServerNotFound(
        "llama-server not found. Set [llama_server] binary, the AOF_LLAMA_SERVER env var, "
        "or put llama-server on PATH. See docs/llama-cpp.md."
    )


def _free_port(host: str) -> int:
    with socket.socket() as s:
        s.bind((host, 0))
        return s.getsockname()[1]


class LlamaServerBackend:
    """InferenceBackend that owns a `llama-server` subprocess."""

    def __init__(
        self,
        model_path: str,
        server: LlamaServerConfig,
        *,
        n_ctx: int = 8192,
        max_tokens: int = 512,
    ) -> None:
        self._model_path = model_path
        self._server = server
        self._n_ctx = n_ctx
        self._max_tokens = max_tokens
        self._proc: subprocess.Popen | None = None
        self._log_path: Path | None = None
        self._client: LocalServerBackend | None = None
        self._base_url = ""

    def _command(self, binary: str, port: int) -> list[str]:
        s = self._server
        return [
            binary,
            "-m", self._model_path,
            "-c", str(self._n_ctx * max(1, s.n_parallel)),  # llama-server splits -c across slots
            "-ngl", str(s.n_gpu_layers),
            "-np", str(max(1, s.n_parallel)),
            "--host", s.host,
            "--port", str(port),
            "--jinja",
            *s.extra_args,
        ]

    async def start(self) -> None:
        binary = find_llama_server(self._server.binary)
        port = _free_port(self._server.host)
        fd, log_name = tempfile.mkstemp(prefix="aof-llama-server-", suffix=".log")
        self._log_path = Path(log_name)
        flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
        with os.fdopen(fd, "wb") as log:
            self._proc = subprocess.Popen(
                self._command(binary, port),
                stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, creationflags=flags,
            )
        atexit.register(self._kill)
        self._base_url = f"http://{self._server.host}:{port}"
        try:
            await self._wait_healthy()
        except BaseException:
            await self.shutdown()
            raise
        cfg = LocalServerConfig(
            base_url=f"{self._base_url}/v1",
            api_key="none",
            model=Path(self._model_path).name,
            enable_thinking=self._server.enable_thinking,
            top_p=self._server.top_p,
            top_k=self._server.top_k,
            repeat_penalty=self._server.repeat_penalty,
            min_temperature=self._server.min_temperature,
        )
        self._client = LocalServerBackend(cfg, n_ctx=self._n_ctx, max_tokens=self._max_tokens)
        await self._client.start()
        logger.info("llama-server ready: %s (%s)", self._base_url, Path(self._model_path).name)

    async def _wait_healthy(self) -> None:
        deadline = time.monotonic() + self._server.startup_timeout
        async with aiohttp.ClientSession() as session:
            while time.monotonic() < deadline:
                if self._proc and self._proc.poll() is not None:
                    raise RuntimeError(
                        f"llama-server exited with code {self._proc.returncode}.\n{self._log_tail()}"
                    )
                try:
                    async with session.get(f"{self._base_url}/health") as resp:
                        if resp.status == 200:
                            return
                except aiohttp.ClientError:
                    pass
                await asyncio.sleep(1.0)
        raise TimeoutError(
            f"llama-server not healthy after {self._server.startup_timeout}s.\n{self._log_tail()}"
        )

    def _log_tail(self, lines: int = 15) -> str:
        try:
            text = self._log_path.read_text(encoding="utf-8", errors="replace") if self._log_path else ""
        except OSError:
            return ""
        return "\n".join(text.splitlines()[-lines:])

    def _kill(self) -> None:
        proc, self._proc = self._proc, None
        if proc and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()

    async def shutdown(self) -> None:
        if self._client:
            await self._client.shutdown()
            self._client = None
        self._kill()
        if self._log_path:
            self._log_path.unlink(missing_ok=True)
            self._log_path = None

    async def complete(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        stop: list[str] | None = None,
    ) -> CompletionResult:
        assert self._client is not None, "Call start() first"
        return await self._client.complete(messages, temperature=temperature, max_tokens=max_tokens, stop=stop)

    async def complete_json(self, messages: list[dict[str, str]], schema: dict | None = None) -> dict:
        assert self._client is not None, "Call start() first"
        return await self._client.complete_json(messages, schema)

    def model_info(self) -> dict[str, Any]:
        return {
            "backend": "llama-server",
            "model": Path(self._model_path).name,
            "n_ctx": self._n_ctx,
            "base_url": self._base_url,
        }
