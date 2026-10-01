"""Graph-aware retrieval over the vault: what a report writer (or `ask`) should read about a topic.

Plain nearest-neighbour search returns the ten sentences that paraphrase the query best, usually from one or two
papers. A review needs breadth: the mechanisms, the evidence for and against, the neighbouring concepts. Retrieval
therefore spreads activation through the structure `aof.refine.graph` builds:

1. **Seeds**: cosine similarity of every claim to the query (max over several query phrasings).
2. **Spread**: each concept and topic hub is activated by the mean similarity of its best members; claims then get
   a bonus from their activated concepts/hubs. A claim that shares the right concept but not the query's wording
   can surface.
3. **Select**: maximal marginal relevance (no near-duplicates), a per-work cap (no single paper dominates), a light
   preference for corroborated / core-graded claims, and a penalty for claims already used elsewhere in the report.

Everything is matrix maths over cached embeddings; the vault is loaded once per `VaultGraph` and refreshed
incrementally after new research.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np

from aof.memory.store import MemoryStore
from aof.memory.zettel import ZettelNote
from aof.refine.assess import live_claims
from aof.refine.claims import Embedder
from aof.refine.graph import parse_evidence
from aof.refine.vectors import unit_rows
from aof.refine.verify import claim_text, url_work_key

logger = logging.getLogger(__name__)


@dataclass
class Evidence:
    """One retrieved claim, with what a writer needs to cite it."""

    note_id: str
    text: str
    source_url: str
    source_label: str  # "Title (Year)"
    year: str
    score: float
    corroborated: bool = False
    concepts: list[str] = field(default_factory=list)  # concept labels
    hub: str = ""  # most specific topic hub title

    def tokens(self) -> int:
        return estimate_tokens(self.text) + estimate_tokens(self.source_label) + 12


def estimate_tokens(text: str) -> int:
    """Conservative token estimate (~3.2 chars per token for English scientific prose)."""
    return int(len(text) / 3.2) + 1


@dataclass
class MapEntry:
    note_id: str
    title: str
    kind: str  # hub | concept
    activation: float
    size: int


class VaultGraph:
    """The live claims, their vectors, and the concept/hub/source layers, loaded for retrieval."""

    def __init__(self, store: MemoryStore, embed: Embedder) -> None:
        self.store, self.embed = store, embed
        self.claims: list[ZettelNote] = []
        self.index: dict[str, int] = {}
        self.x = np.zeros((0, 1), dtype=np.float32)
        self.groups: dict[str, list[int]] = {}  # hub/concept id -> member claim rows
        self.group_title: dict[str, str] = {}
        self.group_kind: dict[str, str] = {}
        self.claim_groups: list[list[str]] = []
        self.work: list[str] = []  # primary work key per claim row
        self.source_label: dict[str, str] = {}  # work key -> label
        self.hub_parent: dict[str, str] = {}

    async def load(self) -> "VaultGraph":
        claims = [n for n in await live_claims(self.store) if n.kind == "claim"]
        vecs = await self.embed([claim_text(n) for n in claims]) if claims else []
        self.claims = claims
        self.index = {n.id: i for i, n in enumerate(claims)}
        self.x = unit_rows(vecs) if claims else np.zeros((0, 1), dtype=np.float32)
        derived = await self.store.get_notes_by_tags(["hub", "concept", "source"], limit=100000, exclude_tags=["archived"])
        self.groups, self.group_title, self.group_kind = {}, {}, {}
        for d in derived:
            if d.status == "archived":
                continue
            if d.kind == "source" and d.sources:
                self.source_label[url_work_key(d.sources[0])] = d.title
                continue
            if d.kind not in ("hub", "concept"):
                continue
            rows = [self.index[i] for i in d.links if i in self.index]
            if d.kind == "hub":
                for i in d.links:
                    if i.startswith("hub-"):
                        self.hub_parent[i] = d.id
            if rows:
                self.groups[d.id] = rows
                self.group_title[d.id] = d.title
                self.group_kind[d.id] = d.kind
        self.claim_groups = [[] for _ in claims]
        for gid, rows in self.groups.items():
            for r in rows:
                self.claim_groups[r].append(gid)
        self.work = [url_work_key(n.sources[0]) if n.sources else n.id for n in claims]
        for n in claims:  # label works that have no source note yet (vault not structured)
            for e in parse_evidence(n):
                key = url_work_key(e["url"])
                if key not in self.source_label:
                    self.source_label[key] = f"{e['cite']} ({e['year']})" if e.get("year") else e["cite"]
        logger.info("vault graph: %d claims, %d groups", len(claims), len(self.groups))
        return self

    # ------------------------------------------------------------ scoring

    async def _query_sims(self, queries: Sequence[str]) -> np.ndarray:
        qs = [q for q in dict.fromkeys(q.strip() for q in queries) if q]
        if not qs or not len(self.claims):
            return np.zeros(len(self.claims), dtype=np.float32)
        q = unit_rows(await self.embed(qs))
        sims = self.x @ q.T  # (n_claims, n_queries)
        # max over phrasings, plus a little of the mean so claims relevant to several phrasings rank higher
        return 0.8 * sims.max(axis=1) + 0.2 * sims.mean(axis=1)

    def _group_activation(self, seed: np.ndarray, top: int = 5) -> dict[str, float]:
        act: dict[str, float] = {}
        for gid, rows in self.groups.items():
            vals = np.sort(seed[rows])[-top:]
            act[gid] = float(vals.mean()) if len(vals) else 0.0
        return act

    def _spread(self, seed: np.ndarray, act: dict[str, float], concept_w: float, hub_w: float) -> np.ndarray:
        bonus = np.zeros_like(seed)
        for r, gids in enumerate(self.claim_groups):
            c = max((act[g] for g in gids if self.group_kind[g] == "concept"), default=0.0)
            h = max((act[g] for g in gids if self.group_kind[g] == "hub"), default=0.0)
            bonus[r] = concept_w * c + hub_w * h
        return seed + bonus

    # ------------------------------------------------------------ public API

    async def retrieve(
        self,
        queries: Sequence[str],
        *,
        k: int = 30,
        token_budget: int = 0,
        per_work: int = 3,
        mmr_lambda: float = 0.75,
        concept_weight: float = 0.15,
        hub_weight: float = 0.1,
        used: Iterable[str] = (),
        used_penalty: float = 0.08,
        min_score: float = 0.2,
        pool: int = 400,
    ) -> list[Evidence]:
        """Up to `k` claims (and within `token_budget` estimated tokens, if set) for the queries, best first."""
        if not self.claims:
            return []
        seed = await self._query_sims(queries)
        score = self._spread(seed, self._group_activation(seed), concept_weight, hub_weight)
        score += 0.03 * np.array([n.status == "canonical" or "grade:core" in n.tags for n in self.claims])
        used_rows = [self.index[i] for i in used if i in self.index]
        if used_rows:
            score[used_rows] -= used_penalty
        cand = np.argsort(-score)[:pool]
        cand = [int(i) for i in cand if score[i] >= min_score]
        chosen: list[int] = []
        per: dict[str, int] = defaultdict(int)
        tokens = 0
        while cand and len(chosen) < k:
            if chosen:
                redundancy = (self.x[cand] @ self.x[chosen].T).max(axis=1)
                mmr = mmr_lambda * score[cand] - (1 - mmr_lambda) * redundancy
                j = int(np.argmax(mmr))
            else:
                j = 0
            r = cand.pop(j)
            if chosen and float((self.x[chosen] @ self.x[r]).max()) >= 0.93:
                continue  # near-duplicate of something already chosen
            if per[self.work[r]] >= per_work:
                continue
            ev = self._evidence(r, float(score[r]))
            if token_budget and tokens + ev.tokens() > token_budget:
                if tokens:
                    break
            tokens += ev.tokens()
            per[self.work[r]] += 1
            chosen.append(r)
        return [self._evidence(r, float(score[r])) for r in chosen]

    async def vault_map(self, queries: Sequence[str], *, hubs: int = 12, concepts: int = 25) -> list[MapEntry]:
        """The topic hubs and concepts most activated by the queries (what the vault knows around a topic)."""
        if not self.claims:
            return []
        seed = await self._query_sims(queries)
        act = self._group_activation(seed)
        out: list[MapEntry] = []
        for kind, limit in (("hub", hubs), ("concept", concepts)):
            ranked = sorted((g for g in act if self.group_kind[g] == kind), key=lambda g: -act[g])[:limit]
            out += [MapEntry(g, self.group_title[g], kind, act[g], len(self.groups[g])) for g in ranked]
        return out

    def _evidence(self, r: int, score: float) -> Evidence:
        n = self.claims[r]
        url = n.sources[0] if n.sources else n.source
        label = self.source_label.get(self.work[r], url)
        year = ""
        if label.endswith(")") and label[-6:-5] == "(" and label[-5:-1].isdigit():
            year = label[-5:-1]
        gids = self.claim_groups[r]
        return Evidence(
            note_id=n.id, text=claim_text(n), source_url=url, source_label=label, year=year, score=score,
            corroborated=len({url_work_key(u) for u in n.sources}) >= 2,
            concepts=[self.group_title[g] for g in gids if self.group_kind[g] == "concept"],
            hub=next((self.group_title[g] for g in gids if self.group_kind[g] == "hub"), ""),
        )

    async def refresh(self) -> None:
        """Reload after research added claims (embeddings are cached, so only new claims are embedded)."""
        await self.load()
