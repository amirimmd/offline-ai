"""Reranking tests."""

from __future__ import annotations

from offline_ai.retrieval.reranker import LexicalFallbackReranker


def test_lexical_rerank() -> None:
    docs = [
        {"document_id": "a", "snippet": "gardening tips", "final_score": 0.5},
        {"document_id": "b", "snippet": "malware attack on company", "final_score": 0.4},
    ]
    out = LexicalFallbackReranker().rerank("malware company", docs, top_k=1)
    assert out[0]["document_id"] == "b"
