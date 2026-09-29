"""Capability -> provider-chain registry with a cheap-first cascade and per-provider stats."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from aof.config import AppConfig
from aof.specialists.base import CAPABILITIES, Provider, SpecialistExhausted, SpecialistResult

logger = logging.getLogger(__name__)


@dataclass
class ProviderStats:
    """What a provider did for one capability (feeds vault metrics: tier usage and escalation rate)."""

    calls: int = 0
    accepted: int = 0
    abstained: int = 0
    errors: int = 0
    seconds: float = 0.0


@dataclass
class SpecialistRegistry:
    """Resolves a capability to its ordered providers and cascades on abstain/error."""

    chains: dict[str, list[Provider]]
    stats: dict[tuple[str, str], ProviderStats] = field(default_factory=dict)

    def providers_for(self, capability: str) -> list[Provider]:
        """Providers in the chain that exist, are available, and support `capability`."""
        return [
            p for p in self.chains.get(capability, [])
            if callable(getattr(p, capability, None)) and p.available()
        ]

    async def call(self, capability: str, *args: Any, only: str | None = None, **kwargs: Any) -> SpecialistResult:
        """Try each provider in order; return the first non-abstaining result.

        `only` restricts the call to one named provider (e.g. "role:large") to escalate directly.
        """
        providers = self.providers_for(capability)
        if only is not None:
            providers = [p for p in providers if p.name == only]
        if not providers:
            raise SpecialistExhausted(f"no available provider for capability {capability!r}")
        tried: list[str] = []
        for provider in providers:
            st = self.stats.setdefault((capability, provider.name), ProviderStats())
            st.calls += 1
            started = time.perf_counter()
            try:
                result = await getattr(provider, capability)(*args, **kwargs)
            except Exception as e:
                st.errors += 1
                logger.warning("%s.%s raised %s: %s", provider.name, capability, type(e).__name__, str(e)[:120])
                result = None
            else:
                if result is None:
                    st.abstained += 1
            finally:
                st.seconds += time.perf_counter() - started
            tried.append(provider.name)
            if result is not None:
                st.accepted += 1
                return result
        raise SpecialistExhausted(f"all providers abstained or failed for {capability!r}: {', '.join(tried)}")

    def escalation_rate(self, capability: str) -> float:
        """Fraction of `capability` calls the first available provider did not settle itself."""
        providers = self.providers_for(capability)
        if not providers:
            return 0.0
        first = self.stats.get((capability, providers[0].name))
        if first is None or first.calls == 0:
            return 0.0
        return 1.0 - first.accepted / first.calls

    def summary(self) -> dict[str, dict[str, dict[str, float]]]:
        out: dict[str, dict[str, dict[str, float]]] = {}
        for (cap, name), st in sorted(self.stats.items()):
            out.setdefault(cap, {})[name] = {
                "calls": st.calls, "accepted": st.accepted, "abstained": st.abstained,
                "errors": st.errors, "seconds": round(st.seconds, 3),
            }
        return out

    async def close(self) -> None:
        seen: set[int] = set()
        for providers in self.chains.values():
            for p in providers:
                if id(p) not in seen:
                    seen.add(id(p))
                    await p.close()


def build_registry(config: AppConfig, get_backend) -> SpecialistRegistry:
    """Build providers from `config.specialists.chains`. `get_backend(role)` is an async role->backend getter."""
    from aof.specialists.embed_provider import SentenceTransformerProvider
    from aof.specialists.needle_provider import NeedleProvider
    from aof.specialists.role_provider import RoleProvider

    cache: dict[str, Provider] = {}

    def make(spec: str) -> Provider:
        if spec not in cache:
            kind, _, arg = spec.partition(":")
            if kind == "needle":
                cache[spec] = NeedleProvider(int(arg or 3))
            elif kind == "role":
                cache[spec] = RoleProvider(arg, config, get_backend)
            elif kind == "sentence-transformers":
                cache[spec] = SentenceTransformerProvider(arg or "all-MiniLM-L6-v2", config.specialists.embed_device)
            else:
                raise ValueError(f"unknown specialist provider spec {spec!r}")
        return cache[spec]

    chains = {
        cap: [make(spec) for spec in specs]
        for cap, specs in config.specialists.chains.items()
        if cap in CAPABILITIES
    }
    return SpecialistRegistry(chains=chains)
