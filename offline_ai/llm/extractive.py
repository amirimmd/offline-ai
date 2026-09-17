"""
Extractive LLM backend.

Used when no GGUF or Transformers weights are present under ``models.yaml``
paths. Builds answers only from evidence blocks supplied in the prompt and
cites the provided document IDs. Does not invent sources.
"""

from __future__ import annotations

import re
from typing import Iterator

from offline_ai.llm.base import LLMBackend, LLMMessage, LLMResult


class ExtractiveLLMBackend(LLMBackend):
    """Grounded answerer that summarizes supplied evidence spans."""

    def __init__(self) -> None:
        self._loaded = False

    @property
    def model_name(self) -> str:
        return "extractive-grounded-v1"

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def load(self) -> None:
        self._loaded = True

    def unload(self) -> None:
        self._loaded = False

    def generate(
        self,
        prompt: str,
        *,
        max_tokens: int = 512,
        temperature: float = 0.1,
        stop: list[str] | None = None,
    ) -> LLMResult:
        self.load()
        text = self._answer_from_prompt(prompt)
        return LLMResult(text=text, model=self.model_name)

    def chat(
        self,
        messages: list[LLMMessage],
        *,
        max_tokens: int = 512,
        temperature: float = 0.1,
        stop: list[str] | None = None,
    ) -> LLMResult:
        prompt = "\n\n".join(f"{m.role.upper()}:\n{m.content}" for m in messages)
        return self.generate(prompt, max_tokens=max_tokens, temperature=temperature, stop=stop)

    def stream(
        self,
        prompt: str,
        *,
        max_tokens: int = 512,
        temperature: float = 0.1,
    ) -> Iterator[str]:
        yield self.generate(prompt, max_tokens=max_tokens, temperature=temperature).text

    def _answer_from_prompt(self, prompt: str) -> str:
        # Expected evidence format in the prompt: [DOC-000000001] <span>
        evidence = re.findall(
            r"\[(DOC-\d+)\]\s*(.+?)(?=\n\[DOC-|\n\nQUESTION:|\Z)",
            prompt,
            flags=re.DOTALL,
        )
        if not evidence:
            return "Insufficient evidence in stored memory."

        lines = ["Based on stored evidence:"]
        for doc_id, span in evidence[:8]:
            span = " ".join(span.split())
            if len(span) > 240:
                span = span[:237] + "..."
            lines.append(f"- {span} [{doc_id}]")
        lines.append("")
        lines.append("Citations: " + ", ".join(f"[{d}]" for d, _ in evidence[:8]))
        return "\n".join(lines)
