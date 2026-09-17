"""Ingestion and raw memory tests — Phase 3."""

from __future__ import annotations

import json
from pathlib import Path

from offline_ai import LocalAI


def test_ingest_text_and_restart(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    ai = LocalAI(ws)
    stats = ai.ingest_text(
        "Company X VPN was compromised.",
        source="report",
        metadata={"severity": "blue"},
        author="analyst1",
    )
    assert stats["documents_added"] == 1
    doc_id = stats["document_ids"][0]
    doc = ai.get_document(doc_id)
    assert doc is not None
    assert doc["original_text"] == "Company X VPN was compromised."

    # Restart
    ai2 = LocalAI(ws)
    doc2 = ai2.get_document(doc_id)
    assert doc2 is not None
    assert doc2["original_text"] == "Company X VPN was compromised."
    assert ai2.stats()["documents"] == 1


def test_duplicate_detection(tmp_path: Path) -> None:
    ai = LocalAI(tmp_path / "ws")
    s1 = ai.ingest_text("same text", source="a", source_url="http://u")
    s2 = ai.ingest_text("same text", source="a", source_url="http://u")
    assert s1["documents_added"] == 1
    assert s2["duplicates"] == 1
    assert ai.stats()["documents"] == 1


def test_ingest_tweets(tmp_path: Path) -> None:
    ai = LocalAI(tmp_path / "ws")
    tweets = [
        {"text": f"Tweet about attack {i} on Company X", "author": "u1", "created_at": "2024-01-01"}
        for i in range(10)
    ]
    stats = ai.ingest_tweets(tweets)
    assert stats["documents_added"] == 10
    assert ai.stats()["documents"] == 10


def test_ingest_json_file(tmp_path: Path) -> None:
    ai = LocalAI(tmp_path / "ws")
    path = tmp_path / "docs.json"
    path.write_text(
        json.dumps(
            [
                {"text": "Report A about malware", "source": "intel"},
                {"text": "Report B about phishing", "author": "bob"},
            ]
        ),
        encoding="utf-8",
    )
    stats = ai.ingest_file(path)
    assert stats["documents_added"] == 2
