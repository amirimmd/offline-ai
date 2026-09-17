"""Memory manager smoke tests."""

from __future__ import annotations

from pathlib import Path

from offline_ai import LocalAI


def test_memory_stats_and_vectors(tmp_path: Path) -> None:
    ai = LocalAI(tmp_path / "ws")
    ai.ingest_text("vectorized document about phishing", source="x")
    st = ai.stats()
    assert st["documents"] == 1
    assert st["vectors"] >= 1
    assert st["components_ready"]["retriever"] is True
