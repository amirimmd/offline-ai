"""Structured facts from a single evidence span (Persian and English)."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from offline_ai.extraction.entities import EntityExtractor
from offline_ai.utils.hashing import normalize_text
from offline_ai.utils.persian import (
    CITIES,
    COUNTRIES,
    contains_persian,
    extract_jalali_dates,
    normalize_persian,
    tokenize,
)

_ENTITIES = EntityExtractor()

_PERSON = r"[\u0600-\u06FF]+(?:\s+[\u0600-\u06FF]+){0,1}"

_TRIP_RE = re.compile(
    rf"(?P<person>{_PERSON})\s+"
    r"(?:رفت(?:ه)?|سفر کرد(?:ه)?)\s+به\s+"
    r"(?P<dest>[\u0600-\u06FF]+)"
)
_EN_TRIP_RE = re.compile(
    r"\b(?P<person>[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+"
    r"(?:went|travelled|traveled)\s+to\s+"
    r"(?P<dest>[A-Z][a-z]+)"
)
_MEETING_RE = re.compile(
    r"(?:یک\s+)?جلسه(?P<qual>\s+مهم)?\s+با\s+"
    rf"(?P<other>{_PERSON})"
)
_MEETING_HAD_RE = re.compile(
    rf"(?P<person>{_PERSON})\s+(?:در\s+تاریخ\s+)?(?P<when>.{{0,40}}?)\s*"
    r"(?:یک\s+)?جلسه(?P<qual>\s+مهم)?\s+(?:داشته(?:\s+است)?|داشت)"
)
_EN_MEETING_RE = re.compile(
    r"\b(?:meeting|met)\s+with\s+(?P<other>[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)",
    re.IGNORECASE,
)
_EN_EVENT_RE = re.compile(
    r"(?P<subject>.+?)\s+(?:was|were|is|are)\s+"
    r"(?P<pred>compromised|breached|attacked|targeted|infected)"
    r"(?:\s+by\s+(?P<obj>.+?))?(?:\.|$)",
    re.IGNORECASE,
)

# محمد امیری علی رو به قتل رسوند / رساند / کشت
_MURDER_RE = re.compile(
    rf"(?P<killer>{_PERSON})\s+"
    rf"(?P<victim>{_PERSON})\s+"
    r"(?:را|رو)\s+"
    r"(?:به\s+قتل\s+(?:رساند|رسوند|رسانده(?:\s+است)?)|کشت(?:ه(?:\s+است)?)?)"
)
_MURDER_RE2 = re.compile(
    rf"(?P<killer>{_PERSON})\s+"
    r"(?:به\s+قتل\s+(?:رساند|رسوند)|کشت)\s+"
    rf"(?P<victim>{_PERSON})"
)
# محمد امیری قاتل علی است
_MURDER_RE3 = re.compile(
    rf"(?P<killer>{_PERSON})\s+قاتل\s+(?P<victim>{_PERSON})\s+(?:است|بود|شده(?:\s+است)?)"
)
# قاتل علی محمد امیری است
_MURDER_RE4 = re.compile(
    rf"قاتل\s+(?P<victim>{_PERSON})\s+(?P<killer>{_PERSON})\s+(?:است|بود)"
)

# در/از تاریخ 13 فروردین به بعد ... محمد امیری را پیدا نکردیم
_MISSING_RE = re.compile(
    r"(?:(?:از|در)\s+(?:تاریخ\s+)?)?(?P<since>.{0,40}?)\s+به\s+بعد\s+"
    r"(?:دیگر\s+)?"
    rf"(?P<person>{_PERSON})\s+"
    r"(?:را\s+)?(?:پیدا\s+نکردیم|پیدا\s+نشد|نیافتیم|مفقود\s+شد)"
)
_MISSING_RE2 = re.compile(
    rf"(?P<person>{_PERSON})\s+"
    r"(?:را\s+)?(?:(?:از|در)\s+(?:تاریخ\s+)?)?(?P<since>.{0,40}?)\s+به\s+بعد\s+"
    r"(?:دیگر\s+)?(?:پیدا\s+نکردیم|پیدا\s+نشد|نیافتیم|مفقود\s+شد|پیدا\s+نشده)"
)
# در تاریخ 13 فروردین 1405 دیگر محمد امیری را پیدا نکردیم
_MISSING_RE3 = re.compile(
    r"(?:(?:از|در)\s+(?:تاریخ\s+)?)?(?P<since>\d{1,2}(?:\s+و\s+\d{1,2})?\s+"
    r"(?:فروردین|اردیبهشت|خرداد|تیر|مرداد|شهریور|مهر|آبان|آذر|دی|بهمن|اسفند)"
    r"(?:\s+\d{3,4})?)\s+"
    r"(?:به\s+بعد\s+)?"
    r"(?:دیگر\s+)?"
    rf"(?P<person>{_PERSON})\s+"
    r"(?:را\s+)?(?:پیدا\s+نکردیم|پیدا\s+نشد|نیافتیم|پیدا\s+نشده)"
)


@dataclass
class TripFact:
    person: str
    destination: str
    date: str | None
    doc_id: str
    span: str


@dataclass
class MeetingFact:
    people: list[str]
    qualifier: str | None
    place: str | None
    date: str | None
    doc_id: str
    span: str


@dataclass
class ActionFact:
    subject: str
    predicate: str
    obj: str
    doc_id: str
    span: str


@dataclass
class MurderFact:
    killer: str
    victim: str
    doc_id: str
    span: str
    date: str | None = None


@dataclass
class MissingFact:
    person: str
    since: str
    doc_id: str
    span: str


@dataclass
class FactSheet:
    doc_id: str
    text: str
    people: list[str] = field(default_factory=list)
    places: list[str] = field(default_factory=list)
    dates: list[str] = field(default_factory=list)
    trips: list[TripFact] = field(default_factory=list)
    meetings: list[MeetingFact] = field(default_factory=list)
    actions: list[ActionFact] = field(default_factory=list)
    murders: list[MurderFact] = field(default_factory=list)
    missing: list[MissingFact] = field(default_factory=list)


def names_match(a: str, b: str) -> bool:
    na, nb = normalize_text(a), normalize_text(b)
    if not na or not nb:
        return False
    if na == nb or na in nb or nb in na:
        return True
    ta, tb = set(tokenize(a)), set(tokenize(b))
    return bool(ta) and (ta <= tb or tb <= ta)


def _clean_person(name: str) -> str:
    name = " ".join((name or "").split())
    drop = {
        "در",
        "تاریخ",
        "از",
        "به",
        "دیگر",
        "و",
        "که",
        "این",
        "آن",
        "را",
        "رو",
        "یک",
        "قتل",
        "رسوند",
        "رساند",
        "کشت",
        "جلسه",
        "مهم",
        "داشته",
        "داشت",
        "رفت",
        "پیدا",
        "نکردیم",
    }
    parts = [p for p in name.split() if p not in drop and p not in drop]
    # Drop trailing object markers if glued
    while parts and parts[-1] in {"را", "رو"}:
        parts.pop()
    return " ".join(parts).strip()


def _date_near(text: str, hint: str | None = None) -> str | None:
    dates = extract_jalali_dates(hint or text)
    if dates:
        return dates[0]
    return None


def extract_fact_sheet(doc_id: str, text: str) -> FactSheet:
    # Unify Arabic/Persian letter variants before pattern matching.
    text = normalize_persian(text or "")
    sheet = FactSheet(doc_id=doc_id, text=text)
    entities = _ENTITIES.extract(text)

    people: list[str] = []
    seen_people: set[str] = set()
    for ent in entities:
        if ent.entity_type != "PERSON":
            continue
        key = ent.normalized_value
        if key in seen_people:
            continue
        seen_people.add(key)
        people.append(ent.value)
    sheet.people = people

    places: list[str] = []
    for ent in entities:
        if ent.entity_type in {"COUNTRY", "LOCATION"}:
            if ent.value not in places:
                places.append(ent.value)
    for name in list(COUNTRIES) + list(CITIES):
        if name in text and name not in places:
            places.append(name)
    sheet.places = places

    sheet.dates = extract_jalali_dates(text)
    date = sheet.dates[0] if sheet.dates else None
    place = sheet.places[0] if sheet.places else None

    for m in _TRIP_RE.finditer(text):
        sheet.trips.append(
            TripFact(
                person=_clean_person(m.group("person")),
                destination=m.group("dest").strip(),
                date=date,
                doc_id=doc_id,
                span=text,
            )
        )
    for m in _EN_TRIP_RE.finditer(text):
        sheet.trips.append(
            TripFact(
                person=m.group("person").strip(),
                destination=m.group("dest").strip(),
                date=date,
                doc_id=doc_id,
                span=text,
            )
        )

    for m in list(_MEETING_RE.finditer(text)) + list(_EN_MEETING_RE.finditer(text)):
        other = _clean_person(m.group("other"))
        qual = (m.groupdict().get("qual") or "").strip() or None
        primary = next((p for p in people if not names_match(p, other)), None)
        participants: list[str] = []
        for p in (primary, other):
            if p and not any(names_match(p, x) for x in participants):
                participants.append(p)
        if len(participants) < 2 and other:
            for p in people:
                if not names_match(p, other) and p not in participants:
                    participants.insert(0, p)
                    break
            if other not in participants:
                participants.append(other)
        sheet.meetings.append(
            MeetingFact(
                people=participants,
                qualifier=qual,
                place=place,
                date=date,
                doc_id=doc_id,
                span=text,
            )
        )

    for m in _MEETING_HAD_RE.finditer(text):
        person = _clean_person(m.group("person"))
        when = _date_near(text, m.group("when")) or date
        qual = (m.group("qual") or "").strip() or None
        if not any(
            names_match(person, (mf.people[0] if mf.people else "")) and mf.date == when
            for mf in sheet.meetings
        ):
            sheet.meetings.append(
                MeetingFact(
                    people=[person] if person else people[:1],
                    qualifier=qual,
                    place=place,
                    date=when,
                    doc_id=doc_id,
                    span=text,
                )
            )

    for m in (
        list(_MURDER_RE.finditer(text))
        + list(_MURDER_RE2.finditer(text))
        + list(_MURDER_RE3.finditer(text))
        + list(_MURDER_RE4.finditer(text))
    ):
        killer = _clean_person(m.group("killer"))
        victim = _clean_person(m.group("victim"))
        if killer and victim and not names_match(killer, victim):
            sheet.murders.append(
                MurderFact(
                    killer=killer,
                    victim=victim,
                    doc_id=doc_id,
                    span=text,
                    date=date,
                )
            )
            for name in (killer, victim):
                if name and not any(names_match(name, p) for p in sheet.people):
                    sheet.people.append(name)

    for m in list(_MISSING_RE.finditer(text)) + list(_MISSING_RE2.finditer(text)) + list(
        _MISSING_RE3.finditer(text)
    ):
        person = _clean_person(m.group("person"))
        since = _date_near(text, m.group("since")) or _date_near(text)
        if person and since:
            if not any(
                names_match(person, x.person) and since == x.since for x in sheet.missing
            ):
                sheet.missing.append(
                    MissingFact(
                        person=person,
                        since=since,
                        doc_id=doc_id,
                        span=text,
                    )
                )
            if not any(names_match(person, p) for p in sheet.people):
                sheet.people.append(person)

    for m in _EN_EVENT_RE.finditer(text):
        sheet.actions.append(
            ActionFact(
                subject=m.group("subject").strip(),
                predicate=m.group("pred").strip().lower(),
                obj=(m.group("obj") or "").strip().rstrip("."),
                doc_id=doc_id,
                span=text,
            )
        )

    if (
        not sheet.trips
        and not sheet.meetings
        and not sheet.actions
        and not sheet.murders
        and not sheet.missing
    ):
        if contains_persian(text) or text:
            sheet.actions.append(
                ActionFact(
                    subject=people[0] if people else "",
                    predicate="mentioned",
                    obj="",
                    doc_id=doc_id,
                    span=text,
                )
            )
    return sheet
