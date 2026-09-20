"""Multi-hop Persian linking: killer role → missing-since date."""

from __future__ import annotations

from pathlib import Path

from offline_ai import LocalAI
from offline_ai.extraction.facts import extract_fact_sheet
from offline_ai.llm.grounded_qa import analyze_question, compose_grounded_answer
from offline_ai.utils.persian import extract_jalali_dates, role_focus_person


DOC_A = "علی در تاریخ 18 و 19 دی جلسه مهم داشته"
DOC_B = (
    "محمد امیری علی رو به قتل رسوند و در تاریخ 13 فروردین به بعد "
    "دیگر محمد امیری را پیدا نکردیم"
)
Q = "قاتل علی از چه زمانی پیدا نشده است ؟"


def test_jalali_dates_without_year() -> None:
    assert "18 و 19 دی" in extract_jalali_dates(DOC_A)
    assert "13 فروردین" in extract_jalali_dates(DOC_B)


def test_role_focus_person() -> None:
    assert role_focus_person(Q) == "علی"


def test_murder_and_missing_facts() -> None:
    sheet = extract_fact_sheet("DOC-1", DOC_B)
    assert sheet.murders
    assert sheet.murders[0].killer.startswith("محمد")
    assert "علی" in sheet.murders[0].victim
    assert sheet.missing
    assert "13 فروردین" in sheet.missing[0].since


def test_intent_missing_since() -> None:
    assert analyze_question(Q).intent == "missing_since"


def test_compose_links_killer_to_date() -> None:
    answer = compose_grounded_answer(
        Q,
        [("DOC-1", DOC_A), ("DOC-2", DOC_B)],
    )
    assert "13 فروردین" in answer
    assert "محمد" in answer
    assert "18 و 19 دی" not in answer.split("\n")[0]


def test_ask_multi_hop_killer_missing(tmp_path: Path) -> None:
    ai = LocalAI(tmp_path / "ws")
    ai.add(DOC_A)
    ai.add(DOC_B)
    result = ai.ask(Q)
    answer = result["answer"]
    assert "13 فروردین" in answer
    assert "محمد" in answer
    assert "شواهد کافی" not in answer
