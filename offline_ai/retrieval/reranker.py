"""Local reranker abstraction."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class Reranker(ABC):
    @abstractmethod
    def rerank(self, query: str, documents: list[dict[str, Any]], top_k: int | None = None) -> list[dict[str, Any]]:
        ...


class LexicalFallbackReranker(Reranker):
    """Fallback: prefer higher hybrid/final scores; boost query term overlap."""

    def rerank(
        self, query: str, documents: list[dict[str, Any]], top_k: int | None = None
    ) -> list[dict[str, Any]]:
        q_terms = {t.lower() for t in query.split() if len(t) > 2}
        scored = []
        for d in documents:
            text = (d.get("text") or d.get("snippet") or "").lower()
            overlap = sum(1 for t in q_terms if t in text)
            base = float(d.get("final_score") or d.get("score") or 0.0)
            scored.append({**d, "rerank_score": base + 0.2 * overlap, "term_overlap": overlap})
        scored.sort(key=lambda x: (x["rerank_score"], x.get("term_overlap", 0)), reverse=True)
        if top_k is not None:
            scored = scored[:top_k]
        return scored


class CrossEncoderReranker(Reranker):
    def __init__(self, model_path: str, device: str | None = None) -> None:
        from sentence_transformers import CrossEncoder

        self._model = CrossEncoder(model_path, device=device)
        self.model_path = model_path

    def rerank(
        self, query: str, documents: list[dict[str, Any]], top_k: int | None = None
    ) -> list[dict[str, Any]]:
        pairs = []
        for d in documents:
            text = d.get("text") or d.get("snippet") or ""
            pairs.append([query, text])
        if not pairs:
            return []
        scores = self._model.predict(pairs)
        out = []
        for d, s in zip(documents, scores, strict=True):
            out.append({**d, "rerank_score": float(s)})
        out.sort(key=lambda x: x["rerank_score"], reverse=True)
        if top_k is not None:
            out = out[:top_k]
        return out


def create_reranker(model_path: str | None = None, device: str | None = None) -> Reranker:
    from pathlib import Path

    if model_path and Path(model_path).exists():
        try:
            return CrossEncoderReranker(model_path, device=device)
        except Exception:
            pass
    return LexicalFallbackReranker()
