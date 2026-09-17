"""Vector store abstraction and FAISS implementation."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np

from offline_ai.utils.logging import get_logger

logger = get_logger(__name__)


class VectorStore(ABC):
    @abstractmethod
    def add(self, vectors: np.ndarray, ids: list[str], metas: list[dict[str, Any]] | None = None) -> list[int]:
        """Add vectors; return assigned internal faiss ids. Application ids in `ids`."""

    @abstractmethod
    def search(self, query: np.ndarray, top_k: int = 10) -> list[tuple[str, float, dict[str, Any]]]:
        """Return list of (application_id, score, meta)."""

    @abstractmethod
    def delete(self, ids: list[str]) -> int: ...

    @abstractmethod
    def save(self, index_path: Path, id_map_path: Path) -> None: ...

    @abstractmethod
    def load(self, index_path: Path, id_map_path: Path) -> None: ...

    @property
    @abstractmethod
    def size(self) -> int: ...


class FaissVectorStore(VectorStore):
    """
    FAISS IndexFlatIP (cosine via normalized vectors).
    Application IDs are authoritative; FAISS ids are ephemeral row indices mapped in id_map.
    """

    def __init__(self, dimension: int) -> None:
        import faiss

        self.dimension = dimension
        self._faiss = faiss
        self.index = faiss.IndexFlatIP(dimension)
        self._id_map: dict[str, dict[str, Any]] = {}  # app_id -> {faiss_id, meta}
        self._faiss_to_app: dict[int, str] = {}
        self._next_faiss_id = 0
        self._tombstones: set[str] = set()

    def add(
        self,
        vectors: np.ndarray,
        ids: list[str],
        metas: list[dict[str, Any]] | None = None,
    ) -> list[int]:
        if vectors.dtype != np.float32:
            vectors = vectors.astype(np.float32)
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        if vectors.shape[1] != self.dimension:
            raise ValueError(f"Expected dim {self.dimension}, got {vectors.shape[1]}")
        if len(ids) != vectors.shape[0]:
            raise ValueError("ids length must match vectors")
        metas = metas or [{} for _ in ids]
        # Normalize for cosine via inner product
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        vectors = vectors / norms

        start = self.index.ntotal
        self.index.add(vectors)
        assigned: list[int] = []
        for i, app_id in enumerate(ids):
            faiss_id = start + i
            self._id_map[app_id] = {"faiss_id": faiss_id, "meta": metas[i]}
            self._faiss_to_app[faiss_id] = app_id
            self._tombstones.discard(app_id)
            assigned.append(faiss_id)
        self._next_faiss_id = self.index.ntotal
        return assigned

    def search(self, query: np.ndarray, top_k: int = 10) -> list[tuple[str, float, dict[str, Any]]]:
        if self.index.ntotal == 0:
            return []
        q = query.astype(np.float32)
        if q.ndim == 1:
            q = q.reshape(1, -1)
        n = np.linalg.norm(q, axis=1, keepdims=True)
        n[n == 0] = 1.0
        q = q / n
        k = min(top_k * 3, self.index.ntotal)  # overfetch for tombstones
        scores, indices = self.index.search(q, k)
        results: list[tuple[str, float, dict[str, Any]]] = []
        for score, idx in zip(scores[0], indices[0], strict=False):
            if idx < 0:
                continue
            app_id = self._faiss_to_app.get(int(idx))
            if app_id is None or app_id in self._tombstones:
                continue
            meta = self._id_map.get(app_id, {}).get("meta", {})
            results.append((app_id, float(score), meta))
            if len(results) >= top_k:
                break
        return results

    def delete(self, ids: list[str]) -> int:
        n = 0
        for app_id in ids:
            if app_id in self._id_map:
                self._tombstones.add(app_id)
                n += 1
        return n

    def save(self, index_path: Path, id_map_path: Path) -> None:
        index_path.parent.mkdir(parents=True, exist_ok=True)
        id_map_path.parent.mkdir(parents=True, exist_ok=True)
        self._faiss.write_index(self.index, str(index_path))
        payload = {
            "dimension": self.dimension,
            "id_map": self._id_map,
            "faiss_to_app": {str(k): v for k, v in self._faiss_to_app.items()},
            "tombstones": list(self._tombstones),
            "next_faiss_id": self._next_faiss_id,
        }
        id_map_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info(
            "FAISS index saved",
            extra={"event": "vector_save", "component": "vector_store"},
        )

    def load(self, index_path: Path, id_map_path: Path) -> None:
        if not index_path.exists() or not id_map_path.exists():
            raise FileNotFoundError("FAISS index or id map missing")
        self.index = self._faiss.read_index(str(index_path))
        payload = json.loads(id_map_path.read_text(encoding="utf-8"))
        self.dimension = int(payload["dimension"])
        self._id_map = payload["id_map"]
        self._faiss_to_app = {int(k): v for k, v in payload["faiss_to_app"].items()}
        self._tombstones = set(payload.get("tombstones") or [])
        self._next_faiss_id = int(payload.get("next_faiss_id", self.index.ntotal))

    @property
    def size(self) -> int:
        return len([i for i in self._id_map if i not in self._tombstones])
