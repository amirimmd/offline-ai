"""Strong relation storage and multi-hop ask over the claim graph."""

from __future__ import annotations

from pathlib import Path

from sqlalchemy import select

from offline_ai import LocalAI
from offline_ai.database.models import Claim, Relationship
from offline_ai.extraction.claims import PRED_KILLED, PRED_MISSING_SINCE, ClaimExtractor


DOC_A = "علی در تاریخ 18 و 19 دی جلسه مهم داشته"
DOC_B = (
    "محمد امیری علی رو به قتل رسوند و در تاریخ 13 فروردین به بعد "
    "دیگر محمد امیری را پیدا نکردیم"
)
Q = "قاتل علی از چه زمانی پیدا نشده است ؟"


def test_claim_extractor_murder_and_missing() -> None:
    claims = ClaimExtractor().extract(DOC_B)
    preds = {(c.predicate, c.subject, c.object) for c in claims}
    assert any(p[0] == PRED_KILLED and "محمد" in p[1] and "علی" in p[2] for p in preds)
    assert any(p[0] == PRED_MISSING_SINCE and "13 فروردین" in p[2] for p in preds)


def test_relations_persisted_and_answer(tmp_path: Path) -> None:
    ai = LocalAI(tmp_path / "ws")
    ai.add(DOC_A)
    ai.add(DOC_B)

    with ai.db.session() as session:
        claims = list(session.execute(select(Claim)).scalars())
        rels = list(session.execute(select(Relationship)).scalars())
    assert any(c.predicate == PRED_KILLED for c in claims)
    assert any(c.predicate == PRED_MISSING_SINCE for c in claims)
    assert any(r.relation_type == PRED_KILLED for r in rels)

    answer = ai.ask(Q, text_only=True)
    assert "13 فروردین" in str(answer)
    assert "محمد" in str(answer)


def test_rebuild_knowledge_upgrades_workspace(tmp_path: Path) -> None:
    ai = LocalAI(tmp_path / "ws")
    ai.add(DOC_B)
    stats = ai.rebuild_knowledge()
    assert stats["documents"] >= 1
    assert stats["claims"] >= 1
    hop = ai.ask("قاتل علی کیست", text_only=True)
    assert "محمد" in str(hop)
