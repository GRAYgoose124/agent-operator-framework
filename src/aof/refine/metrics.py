"""Vault quality metrics: is the vault deduplicated, connected, sourced and verified?"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Sequence

import numpy as np
from urllib.parse import urlparse

from aof.memory.store import MemoryStore
from aof.memory.zettel import ZettelNote
from aof.refine.assess import live_claims
from aof.refine.vectors import cosine_matrix
from aof.refine.pipeline import registry_embedder
from aof.refine.verify import claim_text
from aof.specialists import SpecialistRegistry


@dataclass
class VaultMetrics:
    live_claims: int = 0
    archived: int = 0  # merged duplicates and curated-out claims, all retained
    hubs: int = 0
    duplicate_rate: float = 0.0  # live claim pairs with cosine >= threshold, per live claim
    orphan_rate: float = 0.0  # live claims with no links at all
    avg_links: float = 0.0
    hub_coverage: float = 0.0  # live claims belonging to at least one hub
    sourced_rate: float = 0.0  # live claims with >= 1 source (should be 1.0)
    corroborated_rate: float = 0.0  # live claims backed by >= 2 sources
    verified_rate: float = 0.0  # live claims with a recorded corroboration lookup
    conflicts: int = 0
    source_mix: dict[str, int] = field(default_factory=dict)  # source host -> claim count

    def as_dict(self) -> dict:
        return asdict(self)

    def render(self) -> str:
        rows = [
            ("live claims", self.live_claims), ("archived (retained)", self.archived), ("hubs", self.hubs),
            ("duplicate rate", f"{self.duplicate_rate:.2f}"), ("orphan rate", f"{self.orphan_rate:.0%}"),
            ("avg links / claim", f"{self.avg_links:.1f}"), ("hub coverage", f"{self.hub_coverage:.0%}"),
            ("sourced", f"{self.sourced_rate:.0%}"), ("corroborated (>=2 sources)", f"{self.corroborated_rate:.0%}"),
            ("verified by lookup", f"{self.verified_rate:.0%}"), ("open conflicts", self.conflicts),
        ]
        out = ["| metric | value |", "|---|---|", *[f"| {k} | {v} |" for k, v in rows]]
        if self.source_mix:
            out += ["", "Sources: " + ", ".join(f"{h} ({c})" for h, c in sorted(self.source_mix.items(), key=lambda kv: -kv[1]))]
        return "\n".join(out)


def _host(url: str) -> str:
    host = urlparse(url).netloc.lower()
    return host[4:] if host.startswith("www.") else host or "unknown"


def measure(
    live: Sequence[ZettelNote],
    archived: int,
    hubs: Sequence[ZettelNote],
    vectors: dict[str, Sequence[float]] | None = None,
    *,
    duplicate_threshold: float = 0.9,
) -> VaultMetrics:
    n = len(live)
    m = VaultMetrics(live_claims=n, archived=archived, hubs=len(hubs))
    if n == 0:
        return m
    hub_members = {i for h in hubs for i in h.links}  # includes claims listed by any hub, at any level
    m.orphan_rate = sum(1 for x in live if not x.links) / n
    m.avg_links = sum(len(x.links) for x in live) / n
    m.hub_coverage = sum(1 for x in live if x.id in hub_members) / n
    m.sourced_rate = sum(1 for x in live if x.sources) / n
    m.corroborated_rate = sum(1 for x in live if len(x.sources) >= 2) / n
    m.verified_rate = sum(1 for x in live if "**Corroboration**" in x.content) / n
    m.conflicts = sum(1 for x in live if "conflict" in x.tags)
    for x in live:
        for host in {_host(u) for u in x.sources}:
            m.source_mix[host] = m.source_mix.get(host, 0) + 1
    if vectors and n > 1:
        sims = cosine_matrix([vectors[x.id] for x in live])
        m.duplicate_rate = int(np.triu(sims >= duplicate_threshold, k=1).sum()) / n
    return m


async def compute_metrics(store: MemoryStore, registry: SpecialistRegistry | None = None) -> VaultMetrics:
    live = [n for n in await live_claims(store) if n.kind == "claim"]
    everything = await store.get_notes_by_tags(["claim"], limit=100000)
    hubs = await store.get_notes_by_tags(["hub"], limit=100000, exclude_tags=["archived"])
    vectors = None
    embed = registry_embedder(registry) if registry is not None else None
    if embed and live:
        vectors = dict(zip((n.id for n in live), await embed([claim_text(n) for n in live])))
    return measure(live, len(everything) - len(live), hubs, vectors)
