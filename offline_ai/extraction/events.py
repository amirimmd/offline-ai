"""Event extraction (lightweight)."""

from __future__ import annotations

import re
from dataclasses import dataclass

from offline_ai.extraction.entities import DATE_RE
from offline_ai.utils.persian import JALALI_DATE_RE


EVENT_VERBS = re.compile(
    r"(attack|breach|compromise|leak|ransomware|outage|exploit|intrusion|"
    r"جلسه|دیدار|سفر|حمله|نفوذ|نشت|باج)",
    re.IGNORECASE,
)


@dataclass
class ExtractedEvent:
    name: str
    description: str
    confidence: float
    date_hint: str | None = None


class EventExtractor:
    def extract(self, text: str) -> list[ExtractedEvent]:
        events: list[ExtractedEvent] = []
        if not EVENT_VERBS.search(text):
            return events
        date = None
        m = DATE_RE.search(text) or JALALI_DATE_RE.search(text)
        if m:
            date = m.group(0)
        # Use first sentence as description
        sentence = re.split(r"(?<=[.!?])\s+", text.strip())[0][:300]
        events.append(
            ExtractedEvent(
                name="security_event",
                description=sentence,
                confidence=0.5,
                date_hint=date,
            )
        )
        return events
