"""Persistent embedding cache: a vault is re-embedded by every structure, retrieval and report pass, so remember vectors.

Keyed by (embedding model, text). Stored in a small SQLite file in the workspace; float32 blobs.
"""

from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path

import numpy as np

from aof.refine.claims import Embedder
from aof.specialists import EMBED, SpecialistRegistry


class EmbeddingCache:
    def __init__(self, path: str | Path, model_key: str) -> None:
        self.path = Path(path)
        self.model_key = model_key
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(self.path))
        self._db.execute("CREATE TABLE IF NOT EXISTS emb (k TEXT PRIMARY KEY, v BLOB NOT NULL)")
        self.hits = 0
        self.misses = 0

    def _key(self, text: str) -> str:
        return hashlib.sha1(f"{self.model_key}\n{text}".encode("utf-8")).hexdigest()

    def get_many(self, texts: list[str]) -> list[list[float] | None]:
        keys = [self._key(t) for t in texts]
        found: dict[str, bytes] = {}
        for i in range(0, len(keys), 500):
            chunk = keys[i:i + 500]
            rows = self._db.execute(
                f"SELECT k, v FROM emb WHERE k IN ({','.join('?' * len(chunk))})", chunk,
            ).fetchall()
            found.update(rows)
        return [np.frombuffer(found[k], dtype=np.float32).tolist() if k in found else None for k in keys]

    def put_many(self, texts: list[str], vectors: list[list[float]]) -> None:
        self._db.executemany(
            "INSERT OR REPLACE INTO emb (k, v) VALUES (?, ?)",
            [(self._key(t), np.asarray(v, dtype=np.float32).tobytes()) for t, v in zip(texts, vectors)],
        )
        self._db.commit()

    def wrap(self, embed: Embedder) -> Embedder:
        async def cached(texts: list[str]) -> list[list[float]]:
            got = self.get_many(texts)
            todo = list(dict.fromkeys(t for t, v in zip(texts, got) if v is None))
            self.hits += len(texts) - sum(v is None for v in got)
            self.misses += len(todo)
            if todo:
                fresh = dict(zip(todo, await embed(todo)))
                self.put_many(list(fresh), list(fresh.values()))
                got = [v if v is not None else fresh[t] for t, v in zip(texts, got)]
            return got  # type: ignore[return-value]

        return cached

    def close(self) -> None:
        self._db.close()


def cached_embedder(registry: SpecialistRegistry, workspace_root: str | Path) -> Embedder | None:
    """The registry's embedder behind a workspace-local cache (None when no embedding provider is available)."""
    providers = registry.providers_for(EMBED)
    if not providers:
        return None
    first = providers[0]
    model_key = f"{first.name}:{getattr(first, 'model_id', '')}"

    async def embed(texts: list[str]) -> list[list[float]]:
        return (await registry.call(EMBED, texts)).value

    return EmbeddingCache(Path(workspace_root) / "embeddings.db", model_key).wrap(embed)
