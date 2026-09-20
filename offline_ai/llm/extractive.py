"""
Extractive LLM backend.

Used when no GGUF or Transformers weights are present under ``models.yaml``
paths. Builds a question-specific answer only from evidence blocks supplied
in the prompt and cites the provided document IDs. Does not invent sources.
"""

from __future__ import annotations

import re
from typing import Iterator

from offline_ai.llm.base import LLMBackend, LLMMessage, LLMResult
from offline_ai.llm.grounded_qa import compose_grounded_answer
from offline_ai.utils.persian import (
    insufficient_message,
    role_focus_person,
    token_overlap,
)


class ExtractiveLLMBackend(LLMBackend):
    """Grounded answerer that answers the asked question from evidence spans."""

    def __init__(self) -> None:
        self._loaded = False

    @property
    def model_name(self) -> str:
        return "extractive-grounded-v2"

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
        q_match = re.search(r"QUESTION:\s*(.+?)\s*$", prompt, flags=re.DOTALL)
        query = (q_match.group(1) if q_match else prompt).strip()

        evidence = re.findall(
            r"\[(DOC-\d+)\]\s*(.+?)(?=\n\[DOC-|\n\nQUESTION:|\Z)",
            prompt,
            flags=re.DOTALL,
        )
        if query:
            kept = [
                (doc_id, span)
                for doc_id, span in evidence
                if token_overlap(query, span) > 0
            ]
            if not kept and role_focus_person(query):
                victim = role_focus_person(query)
                kept = [
                    (doc_id, span)
                    for doc_id, span in evidence
                    if victim and victim in span
                ]
            # Always retain murder / missing spans for role questions.
            if role_focus_person(query) or "قاتل" in query:
                for doc_id, span in evidence:
                    if any(k in span for k in ("قتل", "پیدا نکرد", "پیدا نشد", "به بعد")):
                        if (doc_id, span) not in kept:
                            kept.append((doc_id, span))
            evidence = kept
        if not evidence:
            return insufficient_message(query)
        return compose_grounded_answer(query, evidence)
