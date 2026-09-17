"""Hybrid retrieval tests — Phase 5."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

from offline_ai import LocalAI


def test_lexical_and_hybrid(tmp_path: Path) -> None:
    ai = LocalAI(tmp_path / "ws")
    ai.ingest_text("Exact keyword: neonumbra malware family targets banks.", source="intel")
    ai.ingest_text("Completely different gardening tips.", source="blog")
    r = ai.search("neonumbra malware", top_k=5)
    assert r["lexical_hits"] >= 1 or r["semantic_hits"] >= 1
    ids = [x["document_id"] for x in r["results"]]
    assert ids
    doc = ai.get_document(ids[0])
    assert doc and "neonumbra" in doc["original_text"].lower()


def test_temporal_filter_last_30_days(tmp_path: Path) -> None:
    ai = LocalAI(tmp_path / "ws")
    old = datetime.now(timezone.utc) - timedelta(days=60)
    new = datetime.now(timezone.utc) - timedelta(days=2)
    ai.ingest_text("Old report about Company X", source="r", timestamp=old)
    ai.ingest_text("Recent report about Company X attack", source="r", timestamp=new)
    r = ai.search("Find reports from the last 30 days Company X", top_k=10)
    assert r["filters_applied"]["after"] is not None
    after = datetime.fromisoformat(r["filters_applied"]["after"])
    for hit in r["results"]:
        if hit.get("timestamp"):
            ts = datetime.fromisoformat(hit["timestamp"])
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            if after.tzinfo is None:
                after = after.replace(tzinfo=timezone.utc)
            assert ts >= after
