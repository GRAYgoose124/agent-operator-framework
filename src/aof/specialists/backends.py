"""Lazy, shared role -> backend pool for role-based specialists."""

from __future__ import annotations

import asyncio
from typing import Any

from aof.config import AppConfig


class RoleBackends:
    """Starts each role's backend on first use and shuts them all down together.

    llama-server roles with `residency = "swap"` share one VRAM pool: they are handed out behind a
    `ManagedBackend` that loads a role when it is used and parks the least recently used one to make room.
    """

    def __init__(self, config: AppConfig) -> None:
        from aof.inference.residency import ResidencyManager
        from aof.inference.vram import plan_vram

        self._config = config
        self._backends: dict[str, Any] = {}
        self._lock = asyncio.Lock()
        self.plan = plan_vram(config)
        swap = [p for p in self.plan.roles.values() if p.residency == "swap"]
        self.residency = ResidencyManager(
            pool_mb=max((p.gpu_mb for p in swap), default=0),
            park_mode=config.vram.park,
            idle_park_seconds=config.vram.idle_park_seconds,
        )

    def _swaps(self, role: str) -> bool:
        p = self.plan.roles.get(role)
        return bool(p and p.residency == "swap" and role in self._config.llama_server.roles)

    async def get(self, role: str) -> Any:
        async with self._lock:
            if role not in self._backends:
                from aof.inference.residency import ManagedBackend
                from aof.researcher.runner import _make_multi_backends

                lazy = frozenset({role}) if self._swaps(role) else frozenset()
                made = await _make_multi_backends(self._config, {role}, self.plan, lazy)
                if lazy:
                    plan = self.plan.roles[role]
                    self.residency.register(role, made[role], swappable=True, need_mb=plan.gpu_mb)
                    made[role] = ManagedBackend(role, made[role], self.residency)
                self._backends.update(made)
            return self._backends[role]

    async def stage(self, roles: set[str] | frozenset[str]) -> None:
        """Announce the roles the next phase of work needs, so swap roles load once now instead of per call."""
        for role in roles:
            await self.get(role)
        await self.residency.stage(roles)

    async def close(self) -> None:
        for backend in set(self._backends.values()):
            await backend.shutdown()
        self._backends.clear()
