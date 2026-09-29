"""Stage-aware model residency: keep the models a stage needs on the GPU and park the rest.

`swap` roles (see `aof.inference.vram`) share one VRAM pool. A request goes through `lease(role)`, which loads the
role if needed and parks the least recently used swap role(s) to make room, waiting for their in-flight requests to
drain first. `stage(roles)` does the same up front for a whole phase of work, so the switch happens once, not per call.

Parked weights stay in the OS file cache, so waking a model is a reload from system memory, not from disk. Roles
that are `pinned`, or served by a backend that cannot be parked, are never touched.
"""

from __future__ import annotations

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Iterable

logger = logging.getLogger(__name__)


@dataclass
class _Entry:
    role: str
    backend: Any  # needs park()/wake()/resident when swappable
    swappable: bool
    need_mb: int = 0
    inflight: int = 0
    last_used: float = 0.0
    draining: bool = False  # being parked: new leases wait instead of extending its life
    wakes: int = 0
    parks: int = 0


@dataclass
class ResidencyManager:
    pool_mb: int  # VRAM available to swap roles; 0 = unknown (then only one swap role is resident at a time)
    park_mode: str = "unload"
    idle_park_seconds: int = 0
    _entries: dict[str, _Entry] = field(default_factory=dict)
    _cond: asyncio.Condition = field(default_factory=asyncio.Condition)

    def register(self, role: str, backend: Any, *, swappable: bool, need_mb: int = 0) -> None:
        swappable = swappable and hasattr(backend, "park") and hasattr(backend, "wake")
        self._entries[role] = _Entry(role, backend, swappable, need_mb, last_used=time.monotonic())

    # --- internals (call with the condition held) ---

    def _resident_swap(self, besides: str = "") -> list[_Entry]:
        return [e for e in self._entries.values() if e.swappable and e.role != besides and e.backend.resident]

    def _victims(self, target: _Entry) -> list[_Entry]:
        """Least recently used resident swap roles to park so `target` fits in the pool."""
        others = sorted(self._resident_swap(target.role), key=lambda e: e.last_used)
        used = sum(e.need_mb for e in others)
        out: list[_Entry] = []
        while others and (self.pool_mb <= 0 or used + target.need_mb > self.pool_mb):
            v = others.pop(0)
            out.append(v)
            used -= v.need_mb
        return out

    async def _park(self, entry: _Entry) -> None:
        logger.info("Parking %s (%s)", entry.role, self.park_mode)
        await entry.backend.park(self.park_mode)
        entry.parks += 1

    async def _make_resident(self, entry: _Entry) -> None:
        """Called with the condition held; returns once `entry` is up (parking victims as they drain)."""
        while not entry.backend.resident:
            victims = self._victims(entry)
            busy = [v for v in victims if v.inflight > 0]
            if busy:
                for v in busy:
                    v.draining = True
                await self._cond.wait()
                continue
            for v in victims:
                await self._park(v)
                v.draining = False
            logger.info("Waking %s", entry.role)
            await entry.backend.wake()
            entry.wakes += 1
            self._cond.notify_all()

    async def _park_idle(self) -> None:
        if self.idle_park_seconds <= 0:
            return
        now = time.monotonic()
        for e in self._resident_swap():
            if e.inflight == 0 and now - e.last_used > self.idle_park_seconds:
                await self._park(e)

    # --- API ---

    @asynccontextmanager
    async def lease(self, role: str) -> AsyncIterator[Any]:
        """Hold `role`'s backend resident for the duration of the block."""
        entry = self._entries[role]
        if not entry.swappable:
            yield entry.backend
            return
        async with self._cond:
            while entry.draining:
                await self._cond.wait()
            await self._park_idle()
            if not entry.backend.resident:
                await self._make_resident(entry)
            entry.inflight += 1
            entry.last_used = time.monotonic()
        try:
            yield entry.backend
        finally:
            async with self._cond:
                entry.inflight -= 1
                entry.last_used = time.monotonic()
                self._cond.notify_all()

    async def stage(self, roles: Iterable[str]) -> None:
        """Make exactly these swap roles resident (as far as the pool allows) and park the other swap roles."""
        wanted = [self._entries[r] for r in roles if r in self._entries and self._entries[r].swappable]
        async with self._cond:
            for e in self._resident_swap():
                if e not in wanted and e.inflight == 0:
                    await self._park(e)
            for e in wanted:
                if not e.backend.resident:
                    await self._make_resident(e)

    def stats(self) -> dict[str, dict[str, int | bool]]:
        return {
            r: {"resident": bool(e.backend.resident), "wakes": e.wakes, "parks": e.parks}
            for r, e in self._entries.items() if e.swappable
        }


class ManagedBackend:
    """Backend proxy that leases its role from the manager around every call."""

    def __init__(self, role: str, backend: Any, manager: ResidencyManager) -> None:
        self._role, self._backend, self._manager = role, backend, manager

    async def complete(self, *args, **kwargs):
        async with self._manager.lease(self._role) as b:
            return await b.complete(*args, **kwargs)

    async def complete_json(self, *args, **kwargs):
        async with self._manager.lease(self._role) as b:
            return await b.complete_json(*args, **kwargs)

    def model_info(self):
        return self._backend.model_info()

    async def shutdown(self) -> None:
        await self._backend.shutdown()

    def __getattr__(self, name: str):  # everything else (start, n_ctx, ...) goes straight through
        return getattr(self._backend, name)
