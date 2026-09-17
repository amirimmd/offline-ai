"""Vector store interface tests."""

from __future__ import annotations

from offline_ai.memory.embedding_backends import HashingEmbeddingBackend
from offline_ai.memory.semantic import FaissVectorStore


def test_vector_delete_tombstone() -> None:
    emb = HashingEmbeddingBackend(32)
    store = FaissVectorStore(32)
    v = emb.embed_documents(["one", "two"])
    store.add(v, ["a", "b"], [{"document_id": "D1"}, {"document_id": "D2"}])
    assert store.delete(["a"]) == 1
    hits = store.search(emb.embed_query("one"), top_k=5)
    assert all(h[0] != "a" for h in hits)
