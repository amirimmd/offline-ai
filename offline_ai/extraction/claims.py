"""Claim extraction from Persian and English — normalized relation triples."""

from __future__ import annotations

import re
from dataclasses import dataclass

from offline_ai.extraction.facts import extract_fact_sheet, names_match
from offline_ai.utils.persian import extract_jalali_dates, normalize_persian


# Canonical predicates stored in SQLite / graph
PRED_KILLED = "killed"
PRED_MISSING_SINCE = "missing_since"
PRED_HAD_MEETING = "had_meeting"
PRED_TRAVELED_TO = "traveled_to"
PRED_OCCURRED_ON = "occurred_on"
PRED_ASSOCIATED_WITH = "associated_with"
PRED_MET_WITH = "met_with"
PRED_LOCATED_IN = "located_in"

CLAIM_PATTERNS = [
    re.compile(
        r"(?P<subject>[A-Z][\w\s]{1,40}?)\s+(?:announced that|reported that|said that)\s+(?P<object>.+?)(?:\.|$)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?P<subject>[A-Z][\w\s]{1,40}?)\s+(?P<predicate>was|were|is|are)\s+(?P<object>compromised|breached|attacked|targeted|infected)(?:\s+by\s+(?P<actor>.+?))?(?:\.|$)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?P<subject>[A-Z][\w\s]{1,40}?)\s+(?P<predicate>attacked|targeted|compromised|killed|murdered)\s+(?P<object>.+?)(?:\.|$)",
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


def _add(
    claims: list[ExtractedClaim],
    *,
    subject: str,
    predicate: str,
    obj: str,
    span: str,
    confidence: float = 0.85,
) -> None:
    subject = " ".join((subject or "").split())
    obj = " ".join((obj or "").split())
    if not subject or not obj:
        return
    claims.append(
        ExtractedClaim(
            subject=subject,
            predicate=predicate,
            object=obj,
            confidence=confidence,
            source_span=span[:400],
        )
    )


class ClaimExtractor:
    """Extract durable subject–predicate–object triples from text."""

    def extract(self, text: str) -> list[ExtractedClaim]:
        text = " ".join((text or "").split())
        claims: list[ExtractedClaim] = []
        if not text:
            return claims

        sheet = extract_fact_sheet("TMP", text)

        for m in sheet.murders:
            _add(
                claims,
                subject=m.killer,
                predicate=PRED_KILLED,
                obj=m.victim,
                span=m.span,
                confidence=0.92,
            )
            _add(
                claims,
                subject=m.victim,
                predicate=PRED_ASSOCIATED_WITH,
                obj=m.killer,
                span=m.span,
                confidence=0.7,
            )

        for miss in sheet.missing:
            _add(
                claims,
                subject=miss.person,
                predicate=PRED_MISSING_SINCE,
                obj=miss.since,
                span=miss.span,
                confidence=0.9,
            )

        for trip in sheet.trips:
            _add(
                claims,
                subject=trip.person,
                predicate=PRED_TRAVELED_TO,
                obj=trip.destination,
                span=trip.span,
                confidence=0.88,
            )
            if trip.date:
                _add(
                    claims,
                    subject=trip.person,
                    predicate=PRED_OCCURRED_ON,
                    obj=trip.date,
                    span=trip.span,
                    confidence=0.8,
                )

        for meeting in sheet.meetings:
            people = meeting.people or []
            when = meeting.date
            if len(people) >= 2:
                a, b = people[0], people[1]
                _add(
                    claims,
                    subject=a,
                    predicate=PRED_MET_WITH,
                    obj=b,
                    span=meeting.span,
                    confidence=0.86,
                )
                _add(
                    claims,
                    subject=b,
                    predicate=PRED_MET_WITH,
                    obj=a,
                    span=meeting.span,
                    confidence=0.86,
                )
            elif len(people) == 1:
                _add(
                    claims,
                    subject=people[0],
                    predicate=PRED_HAD_MEETING,
                    obj=when or "جلسه",
                    span=meeting.span,
                    confidence=0.84,
                )
            if when and people:
                for p in people:
                    _add(
                        claims,
                        subject=p,
                        predicate=PRED_OCCURRED_ON,
                        obj=when,
                        span=meeting.span,
                        confidence=0.82,
                    )

        for place in sheet.places:
            for person in sheet.people:
                if place in text and person in text:
                    # Only if travel already captured or co-mention with country
                    if any(names_match(t.person, person) and names_match(t.destination, place) for t in sheet.trips):
                        _add(
                            claims,
                            subject=person,
                            predicate=PRED_LOCATED_IN,
                            obj=place,
                            span=text,
                            confidence=0.75,
                        )

        # Legacy English / simple Persian patterns
        for pat in CLAIM_PATTERNS:
            for m in pat.finditer(text):
                gd = m.groupdict()
                subject = (gd.get("subject") or "").strip()
                predicate = (gd.get("predicate") or "related_to").strip().lower()
                obj = (gd.get("object") or gd.get("actor") or "").strip()
                if predicate in {"killed", "murdered"}:
                    predicate = PRED_KILLED
                elif predicate in {"was", "were", "is", "are"}:
                    predicate = "was"
                _add(
                    claims,
                    subject=subject,
                    predicate=predicate,
                    obj=obj,
                    span=m.group(0),
                    confidence=0.6,
                )

        # Date anchors only when no stronger temporal claim exists
        dates = extract_jalali_dates(text)
        if dates and sheet.people and not any(
            c.predicate in {PRED_OCCURRED_ON, PRED_MISSING_SINCE, PRED_HAD_MEETING} for c in claims
        ):
            for p in sheet.people[:2]:
                if any(bad in p for bad in ("قتل", "رسوند", "رو", "را")):
                    continue
                _add(
                    claims,
                    subject=p,
                    predicate=PRED_OCCURRED_ON,
                    obj=dates[0],
                    span=text,
                    confidence=0.55,
                )

        uniq: dict[tuple[str, str, str], ExtractedClaim] = {}
        for c in claims:
            key = (
                normalize_persian(c.subject),
                c.predicate,
                normalize_persian(c.object),
            )
            prev = uniq.get(key)
            if prev is None or c.confidence > prev.confidence:
                uniq[key] = c
        return list(uniq.values())
