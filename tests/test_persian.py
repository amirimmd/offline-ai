"""Persian retrieval and grounded ask."""

from __future__ import annotations

from pathlib import Path

from sqlalchemy import select

from offline_ai import LocalAI
from offline_ai.database.models import Entity
from offline_ai.utils.persian import search_tokens


def test_persian_who_is_ali(tmp_path: Path) -> None:
    ai = LocalAI(tmp_path / "ws")
    ai.add("علی رفت به اسپانیا در روز 18 دی 1404 و یک جلسه مهم با رضا پهلوی داشته است.")
    ai.add("هوا امروز آفتابی است.")
    ai.add("Unrelated championship football match postponed due to weather.")

    result = ai.ask("علی کیست")
    assert isinstance(result, dict)
    answer = result["answer"]
    assert "علی" in answer
    assert "اسپانیا" in answer
    assert "رضا" in answer or "پهلوی" in answer
    assert "football" not in answer.lower()
    assert "DOC-" in answer
    assert result["documents"]
    for doc in result["documents"]:
        assert "علی" in (doc.get("original_text") or "")


def test_persian_search_tokens() -> None:
    assert "علی" in search_tokens("علی کیست")
    assert "کیست" not in search_tokens("علی کیست")


def test_persian_answers_follow_the_question(tmp_path: Path) -> None:
    ai = LocalAI(tmp_path / "ws")
    ai.add("علی رفت به اسپانیا در روز 18 دی 1404 و یک جلسه مهم با رضا پهلوی داشته است.")
    ali = ai.ask("علی کیست", text_only=True)
    reza = ai.ask("رضا پهلوی کیست", text_only=True)
    where = ai.ask("علی کجا رفت", text_only=True)
    assert ali != reza
    assert str(ali).startswith("علی")
    assert str(reza).startswith("رضا")
    assert "اسپانیا" in str(where)


def test_persian_entities(tmp_path: Path) -> None:
    ai = LocalAI(tmp_path / "ws")
    ai.add("علی رفت به اسپانیا در روز 18 دی 1404 و یک جلسه مهم با رضا پهلوی داشته است.")
    with ai.db.session() as session:
        ents = list(session.execute(select(Entity)).scalars())
    types = {(e.entity_type, e.normalized_value) for e in ents}
    assert any(t == "COUNTRY" and "اسپانیا" in v for t, v in types)
    assert any(t == "DATE" and "دی" in v for t, v in types)
    assert any(t == "PERSON" and "علی" in v for t, v in types)
