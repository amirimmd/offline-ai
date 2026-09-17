"""Citation and grounding tests — Phase 8."""

from __future__ import annotations

from pathlib import Path

from offline_ai import LocalAI
from offline_ai.evidence.citation import CitationManager


def test_citation_resolution(tmp_path: Path) -> None:
    ai = LocalAI(tmp_path / "ws")
    stats = ai.ingest_text(
        "Company X announced that its VPN infrastructure was compromised.",
        source="report",
        author="alice",
        source_url="https://example.test/r1",
    )
    doc_id = stats["document_ids"][0]
    cm = CitationManager(ai.db)
    assert cm.validate_citation(doc_id)
    resolved = cm.resolve_citation(doc_id)
    assert resolved is not None
    assert resolved["original_text"].startswith("Company X")
    assert resolved["url"] == "https://example.test/r1"
    assert not cm.validate_citation("DOC-999999999")


def test_ask_grounded_no_invented_sources(tmp_path: Path) -> None:
    ai = LocalAI(tmp_path / "ws")
    ai.ingest_text(
        "Company X announced that its VPN infrastructure was compromised.",
        source="report",
    )
    ai.ingest_text("Weather is sunny in Tehran today.", source="news")
    result = ai.ask("Find all reports related to attacks against Company X.")
    assert "answer" in result
    assert result["documents"]
    for doc in result["documents"]:
        assert doc["id"].startswith("DOC-")
        assert ai.get_document(doc["id"]) is not None
    for cite in result["citations"]:
        assert cite["resolved"] is not None
    # No fabricated DOC ids
    for cite in result.get("citations") or []:
        ref = cite["ref"].strip("[]")
        assert CitationManager(ai.db).validate_citation(ref)


def test_insufficient_evidence(tmp_path: Path) -> None:
    ai = LocalAI(tmp_path / "ws")
    ai.ingest_text("Unrelated gardening advice about tomatoes.", source="blog")
    result = ai.ask("What malware hit Company ZuluOmegaNeverSeen?")
    # May return insufficient or weakly related; must not invent DOC ids
    for cite in result.get("citations") or []:
        assert CitationManager(ai.db).validate_citation(cite["ref"].strip("[]"))
