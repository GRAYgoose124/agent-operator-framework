"""Vector store for semantic search using ChromaDB and sentence-transformers."""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from aof.config import MemoryConfig

logger = logging.getLogger(__name__)

COLLECTION_NAME = "aof_notes"
COLLECTION_NAME_CITATIONS = "aof_citations"


class VectorStore:
    """ChromaDB-backed vector store for embedding and similarity search of notes.

    Lazy init: loads model and ChromaDB only when first used.
    """

    def __init__(
        self,
        persist_directory: str | Path,
        embedding_model: str = "all-MiniLM-L6-v2",
    ) -> None:
        self._persist_dir = Path(persist_directory)
        self._embedding_model = embedding_model
        self._client = None
        self._collection = None
        self._embedding_fn = None

    def _ensure_initialized(self) -> None:
        """Lazy load ChromaDB client and collection."""
        if self._client is not None:
            return
        try:
            import chromadb
            from chromadb.config import Settings
            from chromadb.utils import embedding_functions

            self._persist_dir.mkdir(parents=True, exist_ok=True)
            self._embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=self._embedding_model,
            )
            self._client = chromadb.PersistentClient(
                path=str(self._persist_dir),
                settings=Settings(anonymized_telemetry=False),
            )
            self._collection = self._client.get_or_create_collection(
                name=COLLECTION_NAME,
                embedding_function=self._embedding_fn,
                metadata={"hnsw:space": "cosine"},
            )
            logger.info(
                "Vector store initialized at %s (model=%s)",
                self._persist_dir,
                self._embedding_model,
            )
        except ImportError as e:
            raise ImportError(
                "Vector store requires chromadb and sentence-transformers. "
                "pip install chromadb sentence-transformers"
            ) from e

    def add(self, note_id: str, content: str, metadata: dict | None = None) -> None:
        """Add or update a note embedding."""
        self._ensure_initialized()
        text = (content or "").strip()
        if not text:
            return
        meta = metadata or {}
        meta["note_id"] = note_id
        self._collection.upsert(
            ids=[note_id],
            documents=[text],
            metadatas=[meta],
        )

    def delete(self, note_id: str) -> None:
        """Remove a note from the vector store."""
        self._ensure_initialized()
        try:
            self._collection.delete(ids=[note_id])
        except Exception as e:
            logger.debug("Vector delete %s: %s", note_id, e)

    def search(self, query: str, limit: int = 5) -> list[tuple[str, float]]:
        """Return list of (note_id, score) ordered by similarity. Higher score = more similar."""
        self._ensure_initialized()
        if not (query or "").strip():
            return []
        results = self._collection.query(
            query_texts=[query.strip()],
            n_results=min(limit, 20),
            include=["distances", "metadatas"],
        )
        if not results or not results["ids"] or not results["ids"][0]:
            return []
        ids = results["ids"][0]
        distances = results["distances"][0]
        # Cosine space: distance = 1 - cosine_sim, so similarity = max(0, 1 - distance)
        return [(nid, max(0.0, 1.0 - float(d))) for nid, d in zip(ids, distances)]

    def count(self) -> int:
        """Return number of documents in the collection."""
        self._ensure_initialized()
        return self._collection.count()


class CitationVectorStore:
    """ChromaDB-backed vector store for citation similarity search.

    Stores citations (url, title, snippet) for semantic retrieval.
    Shares persist directory with notes; uses separate collection aof_citations.
    """

    def __init__(
        self,
        persist_directory: str | Path,
        embedding_model: str = "all-MiniLM-L6-v2",
    ) -> None:
        self._persist_dir = Path(persist_directory)
        self._embedding_model = embedding_model
        self._client = None
        self._collection = None
        self._embedding_fn = None

    def _ensure_initialized(self) -> None:
        """Lazy load ChromaDB client and citations collection."""
        if self._client is not None:
            return
        try:
            import chromadb
            from chromadb.config import Settings
            from chromadb.utils import embedding_functions

            self._persist_dir.mkdir(parents=True, exist_ok=True)
            self._embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=self._embedding_model,
            )
            self._client = chromadb.PersistentClient(
                path=str(self._persist_dir),
                settings=Settings(anonymized_telemetry=False),
            )
            self._collection = self._client.get_or_create_collection(
                name=COLLECTION_NAME_CITATIONS,
                embedding_function=self._embedding_fn,
                metadata={"hnsw:space": "cosine"},
            )
            logger.info(
                "Citation vector store initialized at %s (model=%s)",
                self._persist_dir,
                self._embedding_model,
            )
        except ImportError as e:
            raise ImportError(
                "Citation vector store requires chromadb and sentence-transformers. "
                "pip install chromadb sentence-transformers"
            ) from e

    def add(self, citation_id: str, url: str, title: str, snippet: str, source_id: str = "") -> None:
        """Add or update a citation embedding."""
        self._ensure_initialized()
        text = f"{title}\n{snippet}\n{url}".strip()
        if not text:
            return
        self._collection.upsert(
            ids=[citation_id],
            documents=[text],
            metadatas=[{"url": url, "title": title, "snippet": snippet[:500], "source_id": source_id}],
        )

    def search(self, query: str, limit: int = 5) -> list[tuple[dict, float]]:
        """Return list of (citation_metadata, score) ordered by similarity."""
        self._ensure_initialized()
        if not (query or "").strip():
            return []
        results = self._collection.query(
            query_texts=[query.strip()],
            n_results=min(limit, 20),
            include=["distances", "metadatas"],
        )
        if not results or not results["ids"] or not results["ids"][0]:
            return []
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]
        return [
            (dict(meta) if isinstance(meta, dict) else {}, max(0.0, 1.0 - float(d)))
            for meta, d in zip(metadatas, distances)
        ]

    def count(self) -> int:
        """Return number of citations in the collection."""
        self._ensure_initialized()
        return self._collection.count()
