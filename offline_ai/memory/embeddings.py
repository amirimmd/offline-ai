"""Embedding backend abstraction."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

import numpy as np


class EmbeddingBackend(ABC):
    @property
    @abstractmethod
    def dimension(self) -> int: ...

    @property
    @abstractmethod
    def model_name(self) -> str: ...

    @abstractmethod
    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        """Return float32 array shape (n, dim)."""

    @abstractmethod
    def embed_queries(self, texts: Sequence[str]) -> np.ndarray:
        """Return float32 array shape (n, dim)."""

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed_queries([text])[0]
