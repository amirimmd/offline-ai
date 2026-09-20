"""Coordinates raw document storage and semantic indexing."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from sqlalchemy import select

from offline_ai.database.models import DocumentChunk
from offline_ai.database.session import Database
from offline_ai.memory.embedding_backends import create_embedding_backend
from offline_ai.memory.embeddings import EmbeddingBackend
from offline_ai.memory.raw import RawMemory
from offline_ai.memory.semantic import FaissVectorStore, VectorStore
from offline_ai.retrieval.hybrid import HybridRetriever, RetrievalWeights
from offline_ai.retrieval.semantic import SemanticRetriever
from offline_ai.utils.logging import get_logger

logger = get_logger(__name__)


class MemoryManager:
    def __init__(
        self,
        db: Database,
        raw: RawMemory,
        *,
        embedder: EmbeddingBackend,
        vector_store: VectorStore,
        index_path: Path,
        id_map_path: Path,
        retrieval_weights: RetrievalWeights | None = None,
    ) -> None:
        self.db = db
        self.raw = raw
        self.embedder = embedder
        self.vector_store = vector_store
        self.index_path = index_path
        self.id_map_path = id_map_path
        self.semantic = SemanticRetriever(embedder, vector_store)
        self.retriever = HybridRetriever(
            db, embedder, vector_store, weights=retrieval_weights
        )
        self._vectors_dirty = False

    @classmethod
    def create(
        cls,
        db: Database,
        raw: RawMemory,
        *,
        index_path: Path,
        id_map_path: Path,
        dimension: int = 768,
        embedding_model_path: str | None = None,
        device: str | None = None,
        weights: RetrievalWeights | None = None,
    ) -> MemoryManager:
        embedder = create_embedding_backend(
            prefer="auto",
            model_path=embedding_model_path,
            dimension=dimension,
            device=device,
        )
        store = FaissVectorStore(dimension=embedder.dimension)
        if index_path.exists() and id_map_path.exists():
            try:
                store.load(index_path, id_map_path)
            except Exception as exc:
                logger.error(
                    "Failed to load vector index; starting empty",
                    extra={"event": "vector_load_fail", "error": str(exc), "component": "memory"},
                )
                store = FaissVectorStore(dimension=embedder.dimension)
        return cls(
            db,
            raw,
            embedder=embedder,
            vector_store=store,
            index_path=index_path,
            id_map_path=id_map_path,
            retrieval_weights=weights,
        )

    def index_document(self, document_id: str, *, persist: bool = True) -> int:
        with self.db.session() as session:
            chunks = list(
                session.execute(
                    select(DocumentChunk)
                    .where(DocumentChunk.document_id == document_id)
                    .order_by(DocumentChunk.chunk_index)
                ).scalars()
            )
            if not chunks:
                return 0
            n = self.semantic.index_chunks(
                [c.chunk_id for c in chunks],
                [c.text for c in chunks],
                [c.document_id for c in chunks],
            )
        if persist:
            self.persist_vectors()
        else:
            self._vectors_dirty = True
        return n

    def persist_vectors(self) -> None:
        self.vector_store.save(self.index_path, self.id_map_path)
        self._vectors_dirty = False

    def flush_vectors_if_dirty(self) -> None:
        if getattr(self, "_vectors_dirty", False):
            self.persist_vectors()

    def search(self, query: str, **kwargs: Any) -> dict[str, Any]:
        return self.retriever.search(query, **kwargs).to_dict()
