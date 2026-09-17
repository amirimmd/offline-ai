"""Claim lookup tests."""

from __future__ import annotations

from pathlib import Path

from sqlalchemy import select

from offline_ai import LocalAI
from offline_ai.database.models import Claim


def test_claims(tmp_path: Path) -> None:
    ai = LocalAI(tmp_path / "ws")
    ai.ingest_text("Acme Corp was compromised by ransomware.", source="intel")
    with ai.db.session() as session:
        claims = list(session.execute(select(Claim)).scalars())
    assert claims
    c = ai.get_claim(claims[0].claim_id)
    assert c["subject"]
    assert c["source_document_id"]
