"""Lazy, shared role -> backend pool for role-based specialists."""

from __future__ import annotations

import asyncio
from typing import Any

from aof.config import AppConfig


class RoleBackends:
    """Starts each role's backend on first use and shuts them all down together."""

    def __init__(self, config: AppConfig) -> None:
        self._config = config
        self._backends: dict[str, Any] = {}
        self._lock = asyncio.Lock()

    async def get(self, role: str) -> Any:
        async with self._lock:
            if role not in self._backends:
                from aof.researcher.runner import _make_multi_backends

                self._backends.update(await _make_multi_backends(self._config, {role}))
            return self._backends[role]

    async def close(self) -> None:
        for backend in set(self._backends.values()):
            await backend.shutdown()
        self._backends.clear()
