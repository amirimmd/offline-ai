"""Local entity extraction for Persian and English text."""

from __future__ import annotations

import re
from dataclasses import dataclass

from offline_ai.utils.hashing import normalize_text
from offline_ai.utils.persian import (
    CITIES,
    COUNTRIES,
    JALALI_DATE_RE,
    JALALI_MONTHS,
    STOPWORDS,
    extract_jalali_dates,
    tokenize,
)

CVE_RE = re.compile(r"\bCVE-\d{4}-\d{4,7}\b", re.IGNORECASE)
DATE_RE = re.compile(
    r"\b(?:\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{2,4}|"
    r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4})\b",
    re.IGNORECASE,
)
PROPER_RE = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})\b")
TECH_KEYWORDS = {
    "vpn": "TECHNOLOGY",
    "malware": "MALWARE",
    "ransomware": "MALWARE",
    "phishing": "TECHNOLOGY",
    "firewall": "TECHNOLOGY",
    "zero-day": "VULNERABILITY",
    "0day": "VULNERABILITY",
    "بدافزار": "MALWARE",
    "باج‌افزار": "MALWARE",
    "باج افزار": "MALWARE",
    "فیشینگ": "TECHNOLOGY",
}
ORG_HINTS = ("inc", "corp", "ltd", "llc", "company", "bank", "university", "ministry", "شرکت", "بانک", "وزارت")
COUNTRY_HINTS = {
    "iran": "COUNTRY",
    "usa": "COUNTRY",
    "united states": "COUNTRY",
    "china": "COUNTRY",
    "russia": "COUNTRY",
    "israel": "COUNTRY",
    "germany": "COUNTRY",
    "france": "COUNTRY",
    "uk": "COUNTRY",
    "spain": "COUNTRY",
}
NAME_SKIP = STOPWORDS | set(JALALI_MONTHS) | {
    "رفت",
    "رفته",
    "داشت",
    "داشته",
    "جلسه",
    "دیدار",
    "سفر",
    "اعلام",
    "گزارش",
    "قتل",
    "رسوند",
    "رساند",
    "رسانده",
    "کشت",
    "کشته",
    "پیدا",
    "نکردیم",
    "نشد",
    "دیگر",
    "تاریخ",
    "بعد",
}


@dataclass
class ExtractedEntity:
    value: str
    normalized_value: str
    entity_type: str
    confidence: float
    start_offset: int | None = None
    end_offset: int | None = None


class EntityExtractor:
    def extract(self, text: str) -> list[ExtractedEntity]:
        found: list[ExtractedEntity] = []
        seen: set[tuple[str, str]] = set()

        def add(value: str, etype: str, conf: float, start: int | None = None, end: int | None = None) -> None:
            norm = normalize_text(value)
            key = (etype, norm)
            if not norm or key in seen:
                return
            seen.add(key)
            found.append(
                ExtractedEntity(
                    value=value.strip(),
                    normalized_value=norm,
                    entity_type=etype,
                    confidence=conf,
                    start_offset=start,
                    end_offset=end,
                )
            )

        for m in CVE_RE.finditer(text):
            add(m.group(0).upper(), "CVE", 0.99, m.start(), m.end())

        for m in DATE_RE.finditer(text):
            add(m.group(0), "DATE", 0.8, m.start(), m.end())

        for phrase in extract_jalali_dates(text):
            idx = text.find(phrase)
            add(phrase, "DATE", 0.9, idx if idx >= 0 else None, (idx + len(phrase)) if idx >= 0 else None)

        lower = text.lower()
        for kw, etype in TECH_KEYWORDS.items():
            idx = lower.find(kw)
            if idx >= 0:
                add(kw, etype, 0.7, idx, idx + len(kw))

        for name, etype in {**COUNTRY_HINTS, **COUNTRIES, **CITIES}.items():
            idx = text.find(name) if not name.isascii() else lower.find(name)
            if idx >= 0:
                display = name if not name.isascii() else (name.title() if name != "usa" else "USA")
                add(display, etype, 0.8, idx, idx + len(name))

        for m in PROPER_RE.finditer(text):
            val = m.group(1)
            if val.lower() in {"the", "a", "an", "and", "or", "for", "with"}:
                continue
            etype = "ORGANIZATION" if any(h in val.lower() for h in ORG_HINTS) else "PERSON"
            if val.lower().startswith("company "):
                etype = "COMPANY"
            add(val, etype, 0.55, m.start(1), m.end(1))

        tokens = tokenize(text)
        claimed = {e.normalized_value for e in found}
        i = 0
        while i < len(tokens):
            tok = tokens[i]
            if tok.isdigit() or tok in NAME_SKIP or tok in COUNTRIES or tok in CITIES:
                i += 1
                continue
            pair = f"{tok} {tokens[i + 1]}" if i + 1 < len(tokens) else ""
            if pair and tokens[i + 1] not in NAME_SKIP and pair not in COUNTRIES:
                if normalize_text(pair) not in claimed:
                    add(pair, "PERSON", 0.62)
                    claimed.add(normalize_text(pair))
                    i += 2
                    continue
            if len(tok) >= 2 and normalize_text(tok) not in claimed:
                add(tok, "PERSON", 0.58)
                claimed.add(normalize_text(tok))
            i += 1

        return found
