"""Temporal search tests."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

from offline_ai import LocalAI
from offline_ai.retrieval.filters import parse_relative_date


def test_parse_relative() -> None:
    start, end = parse_relative_date("reports from the last 30 days")
    assert start is not None and end is not None
    assert (end - start).days == 30


def test_temporal_ingest_search(tmp_path: Path) -> None:
    ai = LocalAI(tmp_path / "ws")
    now = datetime.now(timezone.utc)
    ai.ingest_text("fresh incident Company X", source="r", timestamp=now - timedelta(days=1))
    ai.ingest_text("ancient incident Company X", source="r", timestamp=now - timedelta(days=90))
    r = ai.search("Company X last 30 days")
    assert r["filters_applied"]["after"] is not None
