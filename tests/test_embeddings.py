"""Embedding and vector store tests — Phase 4."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from offline_ai import LocalAI
from offline_ai.memory.embedding_backends import HashingEmbeddingBackend
from offline_ai.memory.semantic import FaissVectorStore


def test_hashing_embeddings_normalized() -> None:
    emb = HashingEmbeddingBackend(dimension=128)
    v = emb.embed_query("Company X attack")
    assert v.shape == (128,)
    assert abs(float(np.linalg.norm(v)) - 1.0) < 1e-5


def test_faiss_persist(tmp_path: Path) -> None:
    emb = HashingEmbeddingBackend(dimension=64)
    store = FaissVectorStore(64)
    vecs = emb.embed_documents(["alpha malware", "beta phishing", "gamma update"])
    store.add(
        vecs,
        ["CHK-1", "CHK-2", "CHK-3"],
        [{"document_id": "DOC-1"}, {"document_id": "DOC-2"}, {"document_id": "DOC-3"}],
    )
    hits = store.search(emb.embed_query("malware"), top_k=2)
    assert hits
    assert hits[0][0] in {"CHK-1", "CHK-2", "CHK-3"}

    idx = tmp_path / "faiss.index"
    mp = tmp_path / "id_map.json"
    store.save(idx, mp)
    store2 = FaissVectorStore(64)
    store2.load(idx, mp)
    hits2 = store2.search(emb.embed_query("malware"), top_k=2)
    assert hits2[0][0] == hits[0][0]


def test_ingest_embeds_and_search(tmp_path: Path) -> None:
    ai = LocalAI(tmp_path / "ws")
    ai.ingest_text("Company X VPN infrastructure was compromised by malware.", source="report")
    ai.ingest_text("Unrelated weather update for Tehran.", source="news")
    result = ai.search("Company X attack VPN", top_k=5, rerank_top_k=5)
    assert result["results"]
    top = result["results"][0]
    assert "Company X" in (top.get("text") or top.get("snippet") or "")
    # restart preserves vectors
    ai2 = LocalAI(tmp_path / "ws")
    assert ai2.vector_store is not None and ai2.vector_store.size >= 1
    result2 = ai2.search("VPN compromised", top_k=5, rerank_top_k=5)
    assert result2["results"]
