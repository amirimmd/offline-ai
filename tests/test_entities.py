"""Entity and claim extraction tests — Phase 7."""

from __future__ import annotations

from pathlib import Path

from offline_ai import LocalAI
from sqlalchemy import select

from offline_ai.database.models import Claim, Entity


def test_entity_and_claim_extraction(tmp_path: Path) -> None:
    ai = LocalAI(tmp_path / "ws")
    ai.ingest_text(
        "Company X announced that its VPN infrastructure was compromised. CVE-2024-12345 was used.",
        source="report",
    )
    with ai.db.session() as session:
        ents = list(session.execute(select(Entity)).scalars())
        claims = list(session.execute(select(Claim)).scalars())
    assert any(e.entity_type == "CVE" for e in ents)
    assert any("company x" in e.normalized_value for e in ents) or any(
        "vpn" in e.normalized_value for e in ents
    )
    assert claims
    claim = ai.get_claim(claims[0].claim_id)
    assert claim is not None
    assert claim["source_document_id"].startswith("DOC-")
