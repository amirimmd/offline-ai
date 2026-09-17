"""Semantic retrieval over FAISS + chunk/document mapping."""

from __future__ import annotations

from typing import Any

import numpy as np

from offline_ai.memory.embeddings import EmbeddingBackend
from offline_ai.memory.semantic import VectorStore


class SemanticRetriever:
    def __init__(self, embedder: EmbeddingBackend, store: VectorStore) -> None:
        self.embedder = embedder
        self.store = store

    def search(self, query: str, top_k: int = 20) -> list[dict[str, Any]]:
        vec = self.embedder.embed_query(query)
        hits = self.store.search(vec, top_k=top_k)
        results = []
        for app_id, score, meta in hits:
            results.append(
                {
                    "chunk_id": app_id,
                    "document_id": meta.get("document_id"),
                    "score": float(score),
                    "meta": meta,
                    "channel": "semantic",
                }
            )
        return results

    def index_chunks(
        self,
        chunk_ids: list[str],
        texts: list[str],
        document_ids: list[str],
    ) -> int:
        vectors = self.embedder.embed_documents(texts)
        metas = [
            {"document_id": d, "chunk_id": c} for c, d in zip(chunk_ids, document_ids, strict=True)
        ]
        self.store.add(vectors, chunk_ids, metas)
        return len(chunk_ids)
