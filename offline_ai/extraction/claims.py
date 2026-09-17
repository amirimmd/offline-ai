"""Claim extraction from document text (local patterns)."""

from __future__ import annotations

import re
from dataclasses import dataclass


CLAIM_PATTERNS = [
    # Subject announced/compromised/attacked ...
    re.compile(
        r"(?P<subject>[A-Z][\w\s]{1,40}?)\s+(?:announced that|reported that|said that)\s+(?P<object>.+?)(?:\.|$)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?P<subject>[A-Z][\w\s]{1,40}?)\s+(?P<predicate>was|were|is|are)\s+(?P<object>compromised|breached|attacked|targeted|infected)(?:\s+by\s+(?P<actor>.+?))?(?:\.|$)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?P<subject>[A-Z][\w\s]{1,40}?)\s+(?P<predicate>attacked|targeted|compromised)\s+(?P<object>.+?)(?:\.|$)",
        re.IGNORECASE,
    ),
]


@dataclass
class ExtractedClaim:
    subject: str
    predicate: str
    object: str
    confidence: float
    source_span: str
    start_offset: int | None = None
    end_offset: int | None = None


class ClaimExtractor:
    def extract(self, text: str) -> list[ExtractedClaim]:
        claims: list[ExtractedClaim] = []
        for pat in CLAIM_PATTERNS:
            for m in pat.finditer(text):
                gd = m.groupdict()
                subject = (gd.get("subject") or "").strip()
                predicate = (gd.get("predicate") or "related_to").strip()
                obj = (gd.get("object") or gd.get("actor") or "").strip()
                if not subject or not obj:
                    continue
                claims.append(
                    ExtractedClaim(
                        subject=subject,
                        predicate=predicate.lower(),
                        object=obj,
                        confidence=0.6,
                        source_span=m.group(0).strip(),
                        start_offset=m.start(),
                        end_offset=m.end(),
                    )
                )
        # Dedup by span
        uniq: dict[str, ExtractedClaim] = {}
        for c in claims:
            uniq[c.source_span] = c
        return list(uniq.values())
