"""One-call vault structuring: link related claims, then build hub notes over them."""

from __future__ import annotations

from aof.memory.store import MemoryStore
from aof.refine.assess import live_claims
from aof.refine.hubs import build_hubs
from aof.refine.link import apply_links, propose_links
from aof.refine.pipeline import registry_embedder
from aof.refine.verify import claim_text
from aof.specialists import SpecialistRegistry


async def structure_vault(
    store: MemoryStore,
    registry: SpecialistRegistry,
    *,
    link_min_sim: float = 0.55,
    hub_threshold: float = 0.5,
    hub_min_size: int = 3,
    max_hub_size: int = 60,
) -> dict[str, int]:
    """Link live claims and (re)build hubs. Safe to re-run: links are merged and hubs are refreshed in place."""
    embed = registry_embedder(registry)
    if embed is None:
        raise RuntimeError("structuring needs an embedding provider (sentence-transformers)")
    notes = [n for n in await live_claims(store) if n.kind == "claim"]
    if not notes:
        return {"claims": 0, "linked": 0, "hubs": 0}
    vectors = dict(zip((n.id for n in notes), await embed([claim_text(n) for n in notes])))
    linked = await apply_links(store, notes, propose_links(notes, vectors, min_sim=link_min_sim))
    hubs = await build_hubs(
        store, notes, vectors, registry=registry, threshold=hub_threshold, min_size=hub_min_size,
        max_hub_size=max_hub_size,
    )
    return {"claims": len(notes), "linked": linked, "hubs": len(hubs)}
