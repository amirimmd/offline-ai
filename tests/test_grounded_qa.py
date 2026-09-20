"""Question-aware grounded answers from one evidence span."""

from __future__ import annotations

from offline_ai.llm.grounded_qa import analyze_question, compose_grounded_answer
from offline_ai.utils.console import visual_line

SPAN = "علی رفت به اسپانیا در روز 18 دی 1404 و یک جلسه مهم با رضا پهلوی داشته است."
EV = [("DOC-000000001", SPAN)]


def test_intents() -> None:
    assert analyze_question("علی کیست").intent == "who"
    assert analyze_question("رضا پهلوی کیست ؟").intent == "who"
    assert analyze_question("علی کجا رفت").intent == "where"
    assert analyze_question("علی کی رفت").intent == "when"
    assert analyze_question("علی با چه کسی جلسه داشت").intent == "with_whom"


def test_who_ali_vs_reza() -> None:
    ali = compose_grounded_answer("علی کیست", EV)
    reza = compose_grounded_answer("رضا پهلوی کیست ؟", EV)
    assert ali != reza
    assert ali.startswith("علی")
    assert reza.startswith("رضا")
    assert "اسپانیا" in ali
    assert "جلسه" in reza
    assert "DOC-" in ali and "DOC-" in reza


def test_where_and_when_differ() -> None:
    where = compose_grounded_answer("علی کجا رفت", EV)
    when = compose_grounded_answer("علی کی رفت", EV)
    meeting = compose_grounded_answer("علی با چه کسی جلسه داشت", EV)
    assert "اسپانیا" in where
    assert "1404" in when or "دی" in when
    assert "رضا" in meeting or "پهلوی" in meeting
    assert where.splitlines()[0] != when.splitlines()[0]
    assert meeting.splitlines()[0] != where.splitlines()[0]


def test_console_keeps_logical_persian() -> None:
    text = "علی رفت به اسپانیا"
    assert visual_line(text) == text
    assert visual_line("[DOC-000000006]") == "[DOC-000000006]"
