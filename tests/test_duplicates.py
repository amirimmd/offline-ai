"""Duplicate detection tests."""

from __future__ import annotations

from pathlib import Path

from offline_ai import LocalAI


def test_same_content_different_sources_kept(tmp_path: Path) -> None:
    ai = LocalAI(tmp_path / "ws")
    # Different source_url => different source_hash => both kept
    s1 = ai.ingest_text("Same body", source="twitter", source_url="http://a/1")
    s2 = ai.ingest_text("Same body", source="rss", source_url="http://b/2")
    assert s1["documents_added"] == 1
    assert s2["documents_added"] == 1
    assert ai.stats()["documents"] == 2
