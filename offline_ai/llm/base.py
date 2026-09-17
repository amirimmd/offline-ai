"""LLM backend abstraction."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Iterator


@dataclass
class LLMMessage:
    role: str
    content: str


@dataclass
class LLMResult:
    text: str
    model: str
    usage: dict[str, Any] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)


class LLMBackend(ABC):
    @abstractmethod
    def load(self) -> None: ...

    @abstractmethod
    def unload(self) -> None: ...

    @property
    @abstractmethod
    def is_loaded(self) -> bool: ...

    @property
    @abstractmethod
    def model_name(self) -> str: ...

    @abstractmethod
    def generate(
        self,
        prompt: str,
        *,
        max_tokens: int = 512,
        temperature: float = 0.1,
        stop: list[str] | None = None,
    ) -> LLMResult: ...

    @abstractmethod
    def chat(
        self,
        messages: list[LLMMessage],
        *,
        max_tokens: int = 512,
        temperature: float = 0.1,
        stop: list[str] | None = None,
    ) -> LLMResult: ...

    @abstractmethod
    def stream(
        self,
        prompt: str,
        *,
        max_tokens: int = 512,
        temperature: float = 0.1,
    ) -> Iterator[str]: ...
