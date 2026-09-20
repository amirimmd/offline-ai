"""Reject unknown citations and weakly grounded answers."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from offline_ai.utils.persian import contains_persian


@dataclass
class GroundingResult:
    ok: bool
    answer: str
    warnings: list[str] = field(default_factory=list)
    unsupported_spans: list[str] = field(default_factory=list)


class GroundingValidator:
    """
    Validates that answer citations exist and that answer content is supported
    by evidence text (token overlap heuristic + citation presence).
    """

    def __init__(
        self,
        *,
        citation_required: bool = True,
        allow_unsupported_claims: bool = False,
        insufficient_message: str = "Insufficient evidence in stored memory.",
        min_overlap: float = 0.08,
    ) -> None:
        self.citation_required = citation_required
        self.allow_unsupported_claims = allow_unsupported_claims
        self.insufficient_message = insufficient_message
        self.min_overlap = min_overlap

    def validate(
        self,
        answer: str,
        *,
        evidence_texts: list[str],
        valid_citation_ids: list[str],
        cited_ids: list[str],
        bad_citations: list[str],
    ) -> GroundingResult:
        warnings: list[str] = []
        if bad_citations:
            return GroundingResult(
                ok=False,
                answer=self.insufficient_message,
                warnings=[f"Unknown citations rejected: {bad_citations}"],
            )
        if not evidence_texts:
            return GroundingResult(ok=False, answer=self.insufficient_message, warnings=["No evidence"])

        if self.citation_required and not cited_ids:
            # Try to attach first evidence citation if answer looks substantive
            if valid_citation_ids and answer.strip() and answer.strip() != self.insufficient_message:
                answer = answer.rstrip() + f" [{valid_citation_ids[0]}]"
                cited_ids = [valid_citation_ids[0]]
                warnings.append("Auto-attached missing citation to evidence document")
            else:
                return GroundingResult(
                    ok=False,
                    answer=self.insufficient_message,
                    warnings=["citation_required but none present"],
                )

        # Ensure all cited ids are in valid set
        invalid = [c for c in cited_ids if c not in valid_citation_ids]
        if invalid:
            return GroundingResult(
                ok=False,
                answer=self.insufficient_message,
                warnings=[f"Citations not in retrieved evidence: {invalid}"],
            )

        overlap = self._token_overlap(answer, " ".join(evidence_texts))
        persian_insufficient = "شواهد کافی در حافظه"
        if overlap < self.min_overlap and not self.allow_unsupported_claims:
            if (
                self.insufficient_message.lower() in answer.lower()
                or persian_insufficient in answer
            ):
                return GroundingResult(ok=True, answer=answer, warnings=warnings)
            warnings.append(f"Low lexical overlap with evidence ({overlap:.3f})")
            if not self.allow_unsupported_claims:
                # Soft fail only when answer barely touches evidence nouns.
                bullets = []
                for i, t in enumerate(evidence_texts[:5]):
                    cid = valid_citation_ids[i] if i < len(valid_citation_ids) else valid_citation_ids[0]
                    span = " ".join(t.split())[:200]
                    bullets.append(f"- {span}")
                    bullets.append(f"  {cid}")
                header = (
                    "بر اساس شواهد ذخیره شده:"
                    if contains_persian(" ".join(evidence_texts))
                    else "Based on stored evidence:"
                )
                answer = header + "\n" + "\n".join(bullets)
                warnings.append("Regenerated answer from evidence due to weak grounding")

        return GroundingResult(ok=True, answer=answer, warnings=warnings)

    @staticmethod
    def _token_overlap(answer: str, evidence: str) -> float:
        def toks(s: str) -> set[str]:
            return {t.lower() for t in re.findall(r"[\w\u0600-\u06FF]{3,}", s)}

        a, e = toks(answer), toks(evidence)
        # Drop citation tokens
        a = {t for t in a if not t.startswith("doc")}
        if not a:
            return 1.0
        return len(a & e) / len(a)
