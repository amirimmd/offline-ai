"""Bulk line + Excel ingestion tests."""

from __future__ import annotations

from pathlib import Path

from offline_ai import LocalAI
from offline_ai.ingestion.bulk import should_split_multiline, split_text_to_lines


def test_multiline_add_splits_into_documents(tmp_path: Path) -> None:
    ai = LocalAI(tmp_path / "ws")
    text = "علی رفت تهران\nرضا رفت اصفهان\nسارا جلسه داشت"
    assert should_split_multiline(text)
    assert len(split_text_to_lines(text)) == 3
    stages: list[str] = []
    stats = ai.add(text, on_progress=stages.append)
    assert stats["documents_added"] == 3
    assert ai.stats()["documents"] == 3
    assert any("دسته‌ای" in s or "/" in s or "تمام" in s for s in stages)


def test_single_line_stays_one_document(tmp_path: Path) -> None:
    ai = LocalAI(tmp_path / "ws")
    stats = ai.add("فقط یک جمله دانش")
    assert stats["documents_added"] == 1


def test_ingest_text_does_not_split(tmp_path: Path) -> None:
    ai = LocalAI(tmp_path / "ws")
    stats = ai.ingest_text("خط ۱\nخط ۲\nخط ۳", source="note")
    assert stats["documents_added"] == 1


def test_excel_ingest(tmp_path: Path) -> None:
    from openpyxl import Workbook

    path = tmp_path / "knowledge.xlsx"
    wb = Workbook()
    ws = wb.active
    ws.append(["متن", "منبع"])
    ws.append(["محمد امیری علی را به قتل رساند", "گزارش ۱"])
    ws.append(["محمد امیری از ۱۳ فروردین پیدا نشد", "گزارش ۲"])
    ws.append(["علی در دی جلسه داشت", "گزارش ۳"])
    wb.save(path)

    ai = LocalAI(tmp_path / "ws")
    stages: list[str] = []
    stats = ai.ingest_file(path, on_progress=stages.append)
    assert stats["documents_added"] == 3
    assert ai.stats()["documents"] == 3
    assert stages


def test_csv_ingest_joined_columns(tmp_path: Path) -> None:
    path = tmp_path / "rows.csv"
    path.write_text(
        "col_a,col_b\nعلی,تهران\nرضا,اصفهان\n",
        encoding="utf-8",
    )
    ai = LocalAI(tmp_path / "ws")
    stats = ai.ingest_file(path)
    assert stats["documents_added"] == 2
