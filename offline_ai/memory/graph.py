"""Knowledge graph backed by SQLite relationships."""

from __future__ import annotations

from typing import Any

from sqlalchemy import or_, select

from offline_ai.database.models import Relationship
from offline_ai.database.repositories import RelationshipRepository
from offline_ai.database.session import Database


RELATION_TYPES = {
    "works_for",
    "located_in",
    "attacked",
    "mentioned",
    "uses",
    "targets",
    "associated_with",
    "caused",
    "reported_by",
    "related_to",
    "occurred_on",
    "similar_to",
    "duplicate_of",
    "reports_same_event_as",
    "killed",
    "missing_since",
    "had_meeting",
    "traveled_to",
    "met_with",
}


class KnowledgeGraph:
    def __init__(self, db: Database) -> None:
        self.db = db

    def add_relationship(
        self,
        *,
        subject_entity_id: str,
        relation_type: str,
        object_entity_id: str,
        source_document_id: str,
        confidence: float = 0.5,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        if relation_type not in RELATION_TYPES:
            # Allow extension but keep provenance
            pass
        with self.db.session() as session:
            rel = RelationshipRepository(session).create(
                subject_entity_id=subject_entity_id,
                relation_type=relation_type,
                object_entity_id=object_entity_id,
                source_document_id=source_document_id,
                confidence=confidence,
                metadata=metadata,
            )
            session.commit()
            return rel.relationship_id

    def neighbors(self, entity_id: str, limit: int = 50) -> list[dict[str, Any]]:
        with self.db.session() as session:
            rows = session.execute(
                select(Relationship)
                .where(
                    or_(
                        Relationship.subject_entity_id == entity_id,
                        Relationship.object_entity_id == entity_id,
                    )
                )
                .limit(limit)
            ).scalars()
            return [
                {
                    "relationship_id": r.relationship_id,
                    "subject_entity_id": r.subject_entity_id,
                    "relation_type": r.relation_type,
                    "object_entity_id": r.object_entity_id,
                    "source_document_id": r.source_document_id,
                    "confidence": r.confidence,
                }
                for r in rows
            ]

    def documents_for_entity(self, entity_id: str) -> list[str]:
        neigh = self.neighbors(entity_id)
        return sorted({n["source_document_id"] for n in neigh})
