"""
Embedding backends.

``HashingEmbeddingBackend`` is always available (no network, no large weights).
``SentenceTransformersEmbeddingBackend`` is used when a local model directory
exists under the path configured in ``models.yaml``.
"""

from __future__ import annotations

import hashlib
from typing import Sequence

import numpy as np

from offline_ai.memory.embeddings import EmbeddingBackend


class HashingEmbeddingBackend(EmbeddingBackend):
    """
    Deterministic n-gram hashing embedder.

    Suitable for tests and environments where neural embedding weights have not
    been provisioned. Vectors are L2-normalized.
    """

    def __init__(self, dimension: int = 768, ngram_range: tuple[int, int] = (3, 5)) -> None:
        if dimension < 32:
            raise ValueError("dimension must be >= 32")
        self._dim = dimension
        self._ngram_range = ngram_range

    @property
    def dimension(self) -> int:
        return self._dim

    @property
    def model_name(self) -> str:
        return f"hashing-ngram-v1-d{self._dim}"

    def _vectorize(self, text: str) -> np.ndarray:
        vec = np.zeros(self._dim, dtype=np.float32)
        t = text.lower()
        lo, hi = self._ngram_range
        for n in range(lo, hi + 1):
            for i in range(max(0, len(t) - n + 1)):
                gram = t[i : i + n]
                h = hashlib.blake2b(gram.encode("utf-8"), digest_size=8).digest()
                idx = int.from_bytes(h[:4], "little") % self._dim
                sign = 1.0 if (h[4] % 2 == 0) else -1.0
                vec[idx] += sign
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        return np.vstack([self._vectorize(t) for t in texts]).astype(np.float32)

    def embed_queries(self, texts: Sequence[str]) -> np.ndarray:
        return self.embed_documents(texts)


class SentenceTransformersEmbeddingBackend(EmbeddingBackend):
    """Local sentence-transformers model loaded from disk (offline)."""

    def __init__(
        self,
        model_path: str,
        *,
        dimension: int = 768,
        device: str | None = None,
        query_prefix: str = "query: ",
        passage_prefix: str = "passage: ",
    ) -> None:
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_path, device=device)
        self._dim = int(dimension)
        self._name = model_path
        self._query_prefix = query_prefix
        self._passage_prefix = passage_prefix

    @property
    def dimension(self) -> int:
        return self._dim

    @property
    def model_name(self) -> str:
        return self._name

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        prefixed = [self._passage_prefix + t for t in texts]
        emb = self._model.encode(list(prefixed), normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(emb, dtype=np.float32)

    def embed_queries(self, texts: Sequence[str]) -> np.ndarray:
        prefixed = [self._query_prefix + t for t in texts]
        emb = self._model.encode(list(prefixed), normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(emb, dtype=np.float32)


def create_embedding_backend(
    *,
    prefer: str = "auto",
    model_path: str | None = None,
    dimension: int = 768,
    device: str | None = None,
    query_prefix: str = "query: ",
    passage_prefix: str = "passage: ",
) -> EmbeddingBackend:
    """
    Factory for embedding backends.

    ``prefer``:
      - ``sentence_transformers``: require a local model path
      - ``hashing``: always use hashing embedder
      - ``auto``: use sentence-transformers when the path exists, else hashing
    """
    from pathlib import Path

    if prefer == "hashing":
        return HashingEmbeddingBackend(dimension=dimension)
    if prefer == "sentence_transformers":
        if not model_path:
            raise ValueError("model_path required for sentence_transformers")
        return SentenceTransformersEmbeddingBackend(
            model_path,
            dimension=dimension,
            device=device,
            query_prefix=query_prefix,
            passage_prefix=passage_prefix,
        )
    if model_path and Path(model_path).exists():
        try:
            return SentenceTransformersEmbeddingBackend(
                model_path,
                dimension=dimension,
                device=device,
                query_prefix=query_prefix,
                passage_prefix=passage_prefix,
            )
        except Exception:
            pass
    return HashingEmbeddingBackend(dimension=dimension)
