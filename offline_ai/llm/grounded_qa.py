"""Question-aware answers built only from retrieved evidence."""

from __future__ import annotations

import re
from dataclasses import dataclass

from offline_ai.extraction.facts import (
    FactSheet,
    MeetingFact,
    MissingFact,
    MurderFact,
    TripFact,
    extract_fact_sheet,
    names_match,
)
from offline_ai.utils.persian import (
    contains_persian,
    content_tokens,
    insufficient_message,
    normalize_persian,
    role_focus_person,
    tokenize,
    token_overlap,
)

INTENT_WHO = "who"
INTENT_WHERE = "where"
INTENT_WHEN = "when"
INTENT_WITH = "with_whom"
INTENT_MEETING = "meeting"
INTENT_WHY = "why"
INTENT_WHAT = "what"
INTENT_YESNO = "yes_no"
INTENT_ABOUT = "about"
INTENT_KILLER = "killer"
INTENT_MISSING_SINCE = "missing_since"


@dataclass
class Question:
    raw: str
    intent: str
    persian: bool
    focus_tokens: set[str]
    role_person: str | None = None


def analyze_question(query: str) -> Question:
    raw = (query or "").strip()
    n = normalize_persian(raw)
    persian = contains_persian(raw)
    intent = INTENT_ABOUT
    role_person = role_focus_person(raw)

    if re.search(r"قاتل|کشنده", n) and re.search(r"پیدا\s*نشده|پیدا\s*نکرد|نیافت|مفقود|از چه زمان", n):
        intent = INTENT_MISSING_SINCE
    elif re.search(r"قاتل\s+|کشنده\s+|چه کسی\s+(?:کشت|به قتل)", n):
        intent = INTENT_KILLER
    elif re.search(r"کیست|کیه|\bwho\s+is\b|چه کسی است", n):
        intent = INTENT_WHO
    elif re.search(r"با چه کسی|با کی\b|با چه کس|\bwith whom\b", n):
        intent = INTENT_WITH
    elif re.search(r"کجا|where", n):
        intent = INTENT_WHERE
    elif re.search(r"چه زمان|چه روز|چه تاریخ|\bwhen\b|تاریخ\b|از چه زمان", n):
        intent = INTENT_WHEN
    elif re.search(r"\bکی\s+(رفت|رفته|داشت|داشته|کرد|سفر)", n):
        intent = INTENT_WHEN
    elif re.search(r"جلسه|دیدار|ملاقات|\bmeeting\b", n):
        intent = INTENT_MEETING
    elif re.search(r"چرا|\bwhy\b", n):
        intent = INTENT_WHY
    elif re.search(r"^آیا|\b(did|does|was|is)\b.+\?", n):
        intent = INTENT_YESNO
    elif re.search(r"چه کرد|چه شد|چطور|چگونه|گزارش|خلاصه|what happened|what did", n):
        intent = INTENT_WHAT
    elif re.search(r"^کی\s+", n):
        intent = INTENT_WHO

    return Question(
        raw=raw,
        intent=intent,
        persian=persian,
        focus_tokens=content_tokens(raw),
        role_person=role_person,
    )


def _score_name(name: str, tokens: set[str]) -> int:
    return len(set(tokenize(name)) & tokens)


def _pick_person(question: Question, sheets: list[FactSheet]) -> str | None:
    if question.role_person:
        # Prefer the role's target (e.g. علی in قاتل علی) only when asking about that person
        # For killer/missing intents we resolve the role separately.
        if question.intent not in {INTENT_KILLER, INTENT_MISSING_SINCE}:
            return question.role_person
    people: list[str] = []
    for s in sheets:
        for p in s.people:
            if not any(names_match(p, x) for x in people):
                people.append(p)
    if not people:
        return None
    ranked = sorted(
        people,
        key=lambda p: (_score_name(p, question.focus_tokens), len(p)),
        reverse=True,
    )
    if _score_name(ranked[0], question.focus_tokens) == 0:
        if question.intent in {
            INTENT_WHO,
            INTENT_WHERE,
            INTENT_WHEN,
            INTENT_WITH,
        } and len(question.focus_tokens) >= 1:
            return None
        return ranked[0]
    return ranked[0]


def _pick_place(question: Question, sheets: list[FactSheet]) -> str | None:
    places: list[str] = []
    for s in sheets:
        for p in s.places:
            if p not in places:
                places.append(p)
    if not places:
        return None
    ranked = sorted(
        places,
        key=lambda p: (_score_name(p, question.focus_tokens), len(p)),
        reverse=True,
    )
    if _score_name(ranked[0], question.focus_tokens) == 0:
        return None
    return ranked[0]


def _others(meeting: MeetingFact, person: str) -> list[str]:
    return [p for p in meeting.people if not names_match(p, person)]


def _quote(text: str, limit: int = 220) -> str:
    span = " ".join(text.replace("\u200c", " ").split())
    if len(span) > limit:
        span = span[: limit - 3] + "..."
    return span


def _cite_lines(doc_ids: list[str]) -> list[str]:
    return [f"[{did}]" for did in dict.fromkeys(doc_ids)]


def _relevant_sheets(query: str, sheets: list[FactSheet]) -> list[FactSheet]:
    overlapped = [s for s in sheets if token_overlap(query, s.text) > 0]
    # Keep murder/missing sheets even if lexical overlap is weak (role bridging).
    linked = [
        s
        for s in sheets
        if s.murders or s.missing or s in overlapped
    ]
    return linked or sheets


def _all_murders(sheets: list[FactSheet]) -> list[MurderFact]:
    return [m for s in sheets for m in s.murders]


def _all_missing(sheets: list[FactSheet]) -> list[MissingFact]:
    return [m for s in sheets for m in s.missing]


def resolve_killer(victim: str | None, sheets: list[FactSheet]) -> MurderFact | None:
    murders = _all_murders(sheets)
    if not murders:
        return None
    if victim:
        for m in murders:
            if names_match(m.victim, victim):
                return m
    return murders[0]


def resolve_missing_for_role(question: Question, sheets: list[FactSheet]) -> tuple[MissingFact | None, MurderFact | None]:
    """قاتل علی پیدا نشده → killer via murder fact → missing-since of killer."""
    victim = question.role_person
    murder = resolve_killer(victim, sheets)
    missing_list = _all_missing(sheets)
    if not missing_list:
        return None, murder
    if murder:
        for miss in missing_list:
            if names_match(miss.person, murder.killer):
                return miss, murder
    if victim:
        # Sometimes text says killer missing without clear name match order
        for miss in missing_list:
            if not names_match(miss.person, victim):
                return miss, murder
    return missing_list[0], murder


def compose_grounded_answer(query: str, evidence: list[tuple[str, str]]) -> str:
    """Build a question-specific answer. Invents nothing beyond the spans."""
    query = (query or "").strip()
    question = analyze_question(query)
    sheets = _relevant_sheets(
        query,
        [extract_fact_sheet(did, text) for did, text in evidence],
    )
    if not sheets:
        return insufficient_message(query)

    # Prefer multi-hop role answers before generic when/who.
    if question.persian and question.intent in {INTENT_MISSING_SINCE, INTENT_KILLER}:
        hop = _compose_role_hop(question, sheets)
        if hop:
            return hop
    if question.persian and (
        "قاتل" in question.focus_tokens
        or "کشنده" in normalize_persian(query)
        or _all_murders(sheets)
    ):
        hop = _compose_role_hop(question, sheets)
        if hop:
            return hop

    person = _pick_person(question, sheets)
    place = _pick_place(question, sheets)
    trips: list[TripFact] = [t for s in sheets for t in s.trips]
    meetings: list[MeetingFact] = [m for s in sheets for m in s.meetings]

    if question.persian:
        return _compose_fa(question, person, place, sheets, trips, meetings)
    return _compose_en(question, person, place, sheets, trips, meetings)


def _compose_role_hop(question: Question, sheets: list[FactSheet]) -> str | None:
    miss, murder = resolve_missing_for_role(question, sheets)
    used: list[str] = []
    details: list[str] = []

    if question.intent == INTENT_MISSING_SINCE or (
        miss and ("پیدا" in question.focus_tokens or "زمانی" in question.focus_tokens or "زمان" in normalize_persian(question.raw))
    ):
        if miss:
            killer = miss.person
            victim = murder.victim if murder else (question.role_person or "")
            if murder:
                lead = (
                    f"بر اساس شواهد، قاتل {murder.victim} یعنی {murder.killer} "
                    f"از {miss.since} به بعد پیدا نشده است."
                )
                details.append(f"قاتل: {murder.killer}")
                details.append(f"قربانی: {murder.victim}")
                used.append(murder.doc_id)
            else:
                lead = f"{killer} از {miss.since} به بعد پیدا نشده است."
                if victim:
                    details.append(f"اشاره به نقش مرتبط با: {victim}")
            details.append(f"زمان ناپدید شدن / پیدا نشدن: {miss.since}")
            used.append(miss.doc_id)
            return _format_fa(lead, details, sheets, used)

    if murder and (
        question.intent in {INTENT_KILLER, INTENT_WHO}
        or "قاتل" in question.focus_tokens
    ):
        lead = f"قاتل {murder.victim} در شواهد ذخیره‌شده {murder.killer} است."
        details.append(f"قاتل: {murder.killer}")
        details.append(f"قربانی: {murder.victim}")
        used.append(murder.doc_id)
        if miss and names_match(miss.person, murder.killer):
            details.append(f"از {miss.since} به بعد پیدا نشده است")
            used.append(miss.doc_id)
        return _format_fa(lead, details, sheets, used)

    return None


def _format_fa(
    lead: str,
    details: list[str],
    sheets: list[FactSheet],
    used: list[str],
) -> str:
    quotes = []
    for did in dict.fromkeys(used):
        span = next((s.text for s in sheets if s.doc_id == did), None)
        if span:
            quotes.append(_quote(span))
    lines = [lead.strip(), ""]
    if details:
        lines.append("جزئیات مستند")
        lines.extend(details)
        lines.append("")
    lines.append("شاهد")
    if quotes:
        lines.extend(quotes)
    else:
        lines.append(_quote(sheets[0].text))
    lines.extend(_cite_lines(used or [sheets[0].doc_id]))
    return "\n".join(lines)


def _compose_fa(
    question: Question,
    person: str | None,
    place: str | None,
    sheets: list[FactSheet],
    trips: list[TripFact],
    meetings: list[MeetingFact],
) -> str:
    intent = question.intent
    p_trips = [t for t in trips if person and names_match(t.person, person)]
    p_meetings = [
        m for m in meetings if person and any(names_match(x, person) for x in m.people)
    ]
    place_trips = [t for t in trips if place and names_match(t.destination, place)]
    used: list[str] = []
    details: list[str] = []
    lead = ""

    if intent == INTENT_WHO:
        if not person:
            return insufficient_message(question.raw)
        is_traveler = bool(p_trips)
        if p_meetings and is_traveler:
            m0 = p_meetings[0]
            t0 = p_trips[0]
            others = " و ".join(_others(m0, person)) or "طرف دیگر جلسه"
            when = t0.date or m0.date
            time_bit = f" در {when}" if when else ""
            lead = (
                f"{person} در شواهد ذخیره‌شده فردی است که{time_bit} "
                f"به {t0.destination} رفت و جلسه"
                f"{' مهم' if m0.qualifier else ''} با {others} داشته است."
            )
            details.append(f"طرف‌های جلسه: {'، '.join(m0.people)}")
            details.append(f"مکان: {t0.destination}")
            if when:
                details.append(f"زمان: {when}")
            used.extend([t0.doc_id, m0.doc_id])
        elif p_meetings:
            m0 = p_meetings[0]
            others = " و ".join(_others(m0, person)) or ""
            when = m0.date
            time_bit = f" در {when}" if when else ""
            if others:
                lead = (
                    f"{person} در شواهد ذخیره‌شده طرف جلسه"
                    f"{' مهم' if m0.qualifier else ''} با {others} است{time_bit}."
                )
                details.append(f"طرف‌های جلسه: {'، '.join(m0.people)}")
            else:
                lead = (
                    f"{person} در شواهد ذخیره‌شده جلسه"
                    f"{' مهم' if m0.qualifier else ''} داشته است{time_bit}."
                )
            if when:
                details.append(f"زمان: {when}")
            used.append(m0.doc_id)
        elif p_trips:
            t0 = p_trips[0]
            time_bit = f" در {t0.date}" if t0.date else ""
            lead = f"{person} در شواهد ذخیره‌شده به {t0.destination} رفت{time_bit}."
            details.append(f"مقصد: {t0.destination}")
            if t0.date:
                details.append(f"زمان: {t0.date}")
            used.append(t0.doc_id)
        else:
            # Murder victim / killer mention
            murder = resolve_killer(person, sheets)
            if murder and names_match(murder.victim, person):
                lead = f"{person} در شواهد ذخیره‌شده قربانی قتل توسط {murder.killer} است."
                details.append(f"قاتل: {murder.killer}")
                used.append(murder.doc_id)
            elif murder and names_match(murder.killer, person):
                lead = f"{person} در شواهد ذخیره‌شده قاتل {murder.victim} است."
                details.append(f"قربانی: {murder.victim}")
                used.append(murder.doc_id)
            else:
                span = next((s for s in sheets if person and person in s.text), sheets[0])
                lead = f"{person} در شواهد ذخیره‌شده آمده است، بدون شرح هویتی جداگانه."
                details.append(_quote(span.text, 180))
                used.append(span.doc_id)

    elif intent == INTENT_WHERE:
        chosen = p_trips if person else place_trips
        if not chosen and trips:
            chosen = trips
        if not chosen:
            return insufficient_message(question.raw)
        t0 = chosen[0]
        lead = f"{t0.person} به {t0.destination} رفت" + (f" ({t0.date})." if t0.date else ".")
        details.append(f"مقصد: {t0.destination}")
        if t0.date:
            details.append(f"زمان: {t0.date}")
        used.append(t0.doc_id)

    elif intent == INTENT_WHEN:
        # Prefer missing-since / murder dates when role words present
        hop = _compose_role_hop(question, sheets)
        if hop:
            return hop
        dated = [x for x in (p_trips + p_meetings) if getattr(x, "date", None)]
        if not dated:
            dated = [x for x in (trips + meetings) if getattr(x, "date", None)]
        if not dated:
            # Fall back to any sheet dates mentioning the person
            for s in sheets:
                if person and any(names_match(person, p) for p in s.people) and s.dates:
                    lead = f"زمان مرتبط با {person} در شواهد: {s.dates[0]}."
                    details.append(f"تاریخ: {s.dates[0]}")
                    used.append(s.doc_id)
                    return _format_fa(lead, details, sheets, used)
            return insufficient_message(question.raw)
        item = dated[0]
        when = item.date
        if isinstance(item, TripFact):
            lead = f"زمان این رویداد {when} است؛ {item.person} به {item.destination} رفت."
        else:
            lead = f"زمان جلسه {when} است."
            if item.place:
                details.append(f"مکان: {item.place}")
        details.append(f"تاریخ: {when}")
        used.append(item.doc_id)

    elif intent in {INTENT_WITH, INTENT_MEETING}:
        chosen = p_meetings or meetings
        if not chosen:
            return insufficient_message(question.raw)
        m0 = chosen[0]
        who = person or (m0.people[0] if m0.people else "افراد")
        others = _others(m0, who) if person else m0.people[1:]
        other_s = " و ".join(others) if others else "طرف دیگر"
        lead = (
            f"{who} جلسه{' مهم' if m0.qualifier else ''} با {other_s} داشته است"
            + (f" در {m0.place}" if m0.place else "")
            + (f" در {m0.date}" if m0.date else "")
            + "."
        )
        details.append(f"طرف‌های جلسه: {'، '.join(m0.people)}")
        if m0.place:
            details.append(f"مکان: {m0.place}")
        if m0.date:
            details.append(f"زمان: {m0.date}")
        used.append(m0.doc_id)

    elif intent == INTENT_WHY:
        related = p_trips or p_meetings or trips or meetings or _all_murders(sheets)
        if not related:
            return insufficient_message(question.raw)
        lead = "شواهد ذخیره‌شده علت این رویداد را بیان نکرده‌اند؛ فقط خود رویداد ثبت شده است."
        item = related[0]
        details.append(_quote(getattr(item, "span", sheets[0].text), 180))
        used.append(getattr(item, "doc_id", sheets[0].doc_id))

    elif intent == INTENT_YESNO:
        if p_trips or p_meetings or (place and place_trips) or _all_murders(sheets):
            lead = "بله. این موضوع در شواهد ذخیره‌شده آمده است."
            item = (p_trips or p_meetings or place_trips or _all_murders(sheets))[0]
            details.append(_quote(getattr(item, "span", sheets[0].text), 180))
            used.append(getattr(item, "doc_id", sheets[0].doc_id))
        else:
            return insufficient_message(question.raw)

    else:
        hop = _compose_role_hop(question, sheets)
        if hop:
            return hop
        if p_meetings or p_trips:
            return _compose_fa(
                Question(question.raw, INTENT_WHO, True, question.focus_tokens, question.role_person),
                person or (p_trips[0].person if p_trips else p_meetings[0].people[0]),
                place,
                sheets,
                trips,
                meetings,
            )
        span = max(sheets, key=lambda s: token_overlap(question.raw, s.text))
        if token_overlap(question.raw, span.text) == 0 and not span.murders and not span.missing:
            return insufficient_message(question.raw)
        lead = "بر اساس شواهد ذخیره‌شده، این موارد به سؤال مربوط است."
        details.append(_quote(span.text, 200))
        used.append(span.doc_id)

    return _format_fa(lead, details, sheets, used)


def _compose_en(
    question: Question,
    person: str | None,
    place: str | None,
    sheets: list[FactSheet],
    trips: list[TripFact],
    meetings: list[MeetingFact],
) -> str:
    actions = [a for s in sheets for a in s.actions if a.predicate != "mentioned"]
    used: list[str] = []
    details: list[str] = []

    if actions:
        a0 = max(actions, key=lambda a: token_overlap(question.raw, a.span))
        pred = a0.predicate
        obj = f" by {a0.obj}" if a0.obj else ""
        lead = f"{a0.subject} was {pred}{obj}."
        details.append(f"Event: {pred}{obj}".strip())
        if a0.subject:
            details.append(f"Subject: {a0.subject}")
        used.append(a0.doc_id)
        quote_src = a0.span
    elif trips or meetings:
        return _compose_fa(
            Question(question.raw, question.intent, True, question.focus_tokens, question.role_person),
            person,
            place,
            sheets,
            trips,
            meetings,
        )
    else:
        span = max(sheets, key=lambda s: token_overlap(question.raw, s.text))
        lead = "Based on stored evidence:"
        details.append(_quote(span.text, 200))
        used.append(span.doc_id)
        quote_src = span.text

    lines = [lead.strip(), "", "Documented details:"]
    for d in details:
        lines.append(f"- {d}")
    lines.append("")
    lines.append("Source:")
    lines.append(_quote(quote_src))
    lines.extend(_cite_lines(used or [sheets[0].doc_id]))
    return "\n".join(lines)
