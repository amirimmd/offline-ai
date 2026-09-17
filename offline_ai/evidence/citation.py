"""Citation create / validate / resolve against stored records."""

from __future__ import annotations

import re
from typing import Any

from sqlalchemy import select

from offline_ai.database.ids import next_id
from offline_ai.database.models import Citation, Claim, Document, Entity, Event
from offline_ai.database.session import Database

CITATION_RE = re.compile(r"\[((?:DOC|CLAIM|ENTITY|EVENT)-\d+)\]")


class CitationManager:
    def __init__(self, db: Database) -> None:
        self.db = db

    def create_citation(
        self,
        *,
        target_type: str,
        target_id: str,
        answer_id: str | None = None,
        evidence_span: str | None = None,
    ) -> str:
        with self.db.session() as session:
            if not self._exists(session, target_type, target_id):
                raise ValueError(f"Cannot cite missing {target_type}:{target_id}")
            cid = next_id(session, "CIT")
            session.add(
                Citation(
                    citation_id=cid,
                    answer_id=answer_id,
                    target_type=target_type,
                    target_id=target_id,
                    evidence_span=evidence_span,
                )
            )
            session.commit()
            return cid

    def validate_citation(self, citation_ref: str) -> bool:
        target_id = citation_ref.strip("[]")
        target_type = self._type_from_id(target_id)
        if target_type is None:
            return False
        with self.db.session() as session:
            return self._exists(session, target_type, target_id)

    def resolve_citation(self, citation_ref: str) -> dict[str, Any] | None:
        target_id = citation_ref.strip("[]")
        target_type = self._type_from_id(target_id)
        if target_type is None:
            return None
        with self.db.session() as session:
            if not self._exists(session, target_type, target_id):
                return None
            return self.get_source(target_type, target_id, session=session)

    def get_source(
        self, target_type: str, target_id: str, session=None
    ) -> dict[str, Any] | None:
        owns = session is None
        if owns:
            session = self.db.session()
        try:
            if target_type == "document":
                doc = session.execute(
                    select(Document).where(Document.document_id == target_id)
                ).scalar_one_or_none()
                if not doc:
                    return None
                return {
                    "type": "document",
                    "id": doc.document_id,
                    "original_text": doc.original_text,
                    "source": doc.source,
                    "author": doc.author,
                    "date": doc.timestamp.isoformat() if doc.timestamp else None,
                    "url": doc.source_url,
                    "metadata": doc.metadata_json,
                }
            if target_type == "claim":
                claim = session.execute(
                    select(Claim).where(Claim.claim_id == target_id)
                ).scalar_one_or_none()
                if not claim:
                    return None
                return {
                    "type": "claim",
                    "id": claim.claim_id,
                    "subject": claim.subject,
                    "predicate": claim.predicate,
                    "object": claim.object,
                    "source_document_id": claim.source_document_id,
                    "source_span": claim.source_span,
                }
            if target_type == "entity":
                ent = session.execute(
                    select(Entity).where(Entity.entity_id == target_id)
                ).scalar_one_or_none()
                if not ent:
                    return None
                return {
                    "type": "entity",
                    "id": ent.entity_id,
                    "value": ent.value,
                    "entity_type": ent.entity_type,
                }
            if target_type == "event":
                ev = session.execute(
                    select(Event).where(Event.event_id == target_id)
                ).scalar_one_or_none()
                if not ev:
                    return None
                return {
                    "type": "event",
                    "id": ev.event_id,
                    "name": ev.name,
                    "description": ev.description,
                    "source_document_id": ev.source_document_id,
                }
            return None
        finally:
            if owns:
                session.close()

    def get_evidence(self, document_id: str) -> dict[str, Any] | None:
        return self.resolve_citation(document_id)

    def extract_citation_ids(self, text: str) -> list[str]:
        return CITATION_RE.findall(text)

    def validate_answer_citations(self, text: str) -> tuple[bool, list[str]]:
        ids = self.extract_citation_ids(text)
        bad = [i for i in ids if not self.validate_citation(i)]
        return len(bad) == 0, bad

    @staticmethod
    def _type_from_id(target_id: str) -> str | None:
        if target_id.startswith("DOC-"):
            return "document"
        if target_id.startswith("CLAIM-"):
            return "claim"
        if target_id.startswith("ENTITY-"):
            return "entity"
        if target_id.startswith("EVENT-"):
            return "event"
        return None

    @staticmethod
    def _exists(session, target_type: str, target_id: str) -> bool:
        mapping = {
            "document": (Document, Document.document_id),
            "claim": (Claim, Claim.claim_id),
            "entity": (Entity, Entity.entity_id),
            "event": (Event, Event.event_id),
        }
        model, col = mapping[target_type]
        return (
            session.execute(select(model).where(col == target_id)).scalar_one_or_none()
            is not None
        )
