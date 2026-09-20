"""Query stored claims and entity relationships for multi-hop answers."""

from __future__ import annotations

from typing import Any

from sqlalchemy import or_, select

from offline_ai.database.models import Claim, Entity, Relationship
from offline_ai.database.session import Database
from offline_ai.extraction.claims import (
    PRED_KILLED,
    PRED_MISSING_SINCE,
)
from offline_ai.utils.hashing import normalize_text
from offline_ai.utils.persian import (
    content_tokens,
    normalize_persian,
    role_focus_person,
    tokenize,
)


class RelationStore:
    """Read/write helpers over claims + relationships tables."""

    def __init__(self, db: Database) -> None:
        self.db = db

    def claims_for_query(self, query: str, *, limit: int = 40) -> list[dict[str, Any]]:
        tokens = content_tokens(query)
        role = role_focus_person(query)
        with self.db.session() as session:
            rows = list(session.execute(select(Claim).order_by(Claim.id.desc()).limit(500)).scalars())
        scored: list[tuple[int, Claim]] = []
        for c in rows:
            blob = f"{c.subject} {c.predicate} {c.object} {c.source_span or ''}"
            score = sum(1 for t in tokens if t in normalize_persian(blob))
            if role and (
                role in normalize_persian(c.subject)
                or role in normalize_persian(c.object)
            ):
                score += 3
            if c.predicate in {PRED_KILLED, PRED_MISSING_SINCE} and (
                "قاتل" in tokens or "قتل" in tokens or "پیدا" in tokens or role
            ):
                score += 2
            if score > 0:
                scored.append((score, c))
        scored.sort(key=lambda x: x[0], reverse=True)
        out = []
        for score, c in scored[:limit]:
            out.append(
                {
                    "claim_id": c.claim_id,
                    "subject": c.subject,
                    "predicate": c.predicate,
                    "object": c.object,
                    "confidence": c.confidence,
                    "document_id": c.source_document_id,
                    "span": c.source_span or "",
                    "score": score,
                    "triple": f"{c.subject} —[{c.predicate}]→ {c.object}",
                }
            )
        return out

    def resolve_killer(self, victim: str) -> dict[str, Any] | None:
        victim_n = normalize_persian(victim)
        with self.db.session() as session:
            rows = list(
                session.execute(select(Claim).where(Claim.predicate == PRED_KILLED)).scalars()
            )
        for c in rows:
            if victim_n in normalize_persian(c.object) or normalize_persian(c.object) in victim_n:
                return {
                    "killer": c.subject,
                    "victim": c.object,
                    "document_id": c.source_document_id,
                    "span": c.source_span or "",
                    "claim_id": c.claim_id,
                }
        return None

    def resolve_missing_since(self, person: str) -> dict[str, Any] | None:
        person_n = normalize_persian(person)
        with self.db.session() as session:
            rows = list(
                session.execute(
                    select(Claim).where(Claim.predicate == PRED_MISSING_SINCE)
                ).scalars()
            )
        for c in rows:
            if person_n in normalize_persian(c.subject) or normalize_persian(c.subject) in person_n:
                return {
                    "person": c.subject,
                    "since": c.object,
                    "document_id": c.source_document_id,
                    "span": c.source_span or "",
                    "claim_id": c.claim_id,
                }
        return None

    def answer_role_question(self, query: str) -> dict[str, Any] | None:
        """
        Multi-hop over stored claims:
        قاتل علی از چه زمانی پیدا نشده → killed(?, علی) then missing_since(killer, date)
        """
        role = role_focus_person(query)
        n = normalize_persian(query)
        wants_missing = bool(
            re_search_missing(n) or ("چه زمان" in n and ("قاتل" in n or "پیدا" in n))
        )
        wants_killer = "قاتل" in n or "کشنده" in n

        if not role and not wants_killer:
            return None

        victim = role or ""
        murder = self.resolve_killer(victim) if victim else None
        if murder is None and wants_killer:
            # pick any kill claim mentioning tokens
            claims = self.claims_for_query(query, limit=10)
            for c in claims:
                if c["predicate"] == PRED_KILLED:
                    murder = {
                        "killer": c["subject"],
                        "victim": c["object"],
                        "document_id": c["document_id"],
                        "span": c["span"],
                        "claim_id": c["claim_id"],
                    }
                    break

        if murder is None:
            return None

        missing = self.resolve_missing_since(murder["killer"])
        if wants_missing and missing:
            return {
                "kind": "killer_missing_since",
                "killer": murder["killer"],
                "victim": murder["victim"],
                "since": missing["since"],
                "document_ids": list(
                    dict.fromkeys([murder["document_id"], missing["document_id"]])
                ),
                "spans": [murder["span"], missing["span"]],
                "claims": [murder, missing],
            }
        if wants_killer or wants_missing:
            return {
                "kind": "killer_of",
                "killer": murder["killer"],
                "victim": murder["victim"],
                "since": missing["since"] if missing else None,
                "document_ids": list(
                    dict.fromkeys(
                        [murder["document_id"]]
                        + ([missing["document_id"]] if missing else [])
                    )
                ),
                "spans": [murder["span"]] + ([missing["span"]] if missing else []),
                "claims": [murder] + ([missing] if missing else []),
            }
        return None

    def related_document_ids(self, query: str, *, limit: int = 20) -> list[str]:
        docs: list[str] = []
        for c in self.claims_for_query(query, limit=limit):
            docs.append(c["document_id"])
        hop = self.answer_role_question(query)
        if hop:
            docs.extend(hop["document_ids"])
        # Graph expansion via entity name tokens
        names = [t for t in tokenize(query) if len(t) >= 2]
        role = role_focus_person(query)
        if role:
            names.append(role)
        with self.db.session() as session:
            for name in names:
                norm = normalize_text(name)
                ents = list(
                    session.execute(
                        select(Entity).where(Entity.normalized_value.contains(norm))
                    ).scalars()
                )[:5]
                for ent in ents:
                    rels = list(
                        session.execute(
                            select(Relationship).where(
                                or_(
                                    Relationship.subject_entity_id == ent.entity_id,
                                    Relationship.object_entity_id == ent.entity_id,
                                )
                            )
                        ).scalars()
                    )
                    for r in rels:
                        docs.append(r.source_document_id)
        return list(dict.fromkeys(docs))[:limit]


def re_search_missing(n: str) -> bool:
    return bool(
        ("پیدا" in n and ("نشده" in n or "نکرد" in n))
        or "مفقود" in n
        or "نیافت" in n
    )
