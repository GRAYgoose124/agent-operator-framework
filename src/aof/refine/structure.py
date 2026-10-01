"""One-call vault structuring: topic hubs, concept notes, source notes and sparse typed claim links.

See `aof.refine.graph` for the layers and why claim links are no longer "every similar claim".
"""

from __future__ import annotations

from pathlib import Path

from aof.memory.store import MemoryStore
from aof.refine.embcache import cached_embedder
from aof.refine.graph import GraphOptions, build_graph
from aof.refine.pipeline import registry_embedder
from aof.specialists import SpecialistRegistry


async def structure_vault(
    store: MemoryStore,
    registry: SpecialistRegistry,
    *,
    options: GraphOptions | None = None,
    workspace_root: str | Path | None = None,
    progress=None,
) -> dict[str, int]:
    """Rebuild the vault's derived structure. Safe to re-run: claim links are replaced, derived notes refreshed."""
    embed = cached_embedder(registry, workspace_root) if workspace_root else registry_embedder(registry)
    if embed is None:
        raise RuntimeError("structuring needs an embedding provider (sentence-transformers)")
    report = await build_graph(store, registry, embed, options, progress=progress)
    return report.as_dict()
