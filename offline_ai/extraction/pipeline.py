"""Extraction pipeline: entities, claims, topics, events, and graph edges."""

from __future__ import annotations

from typing import Any

from offline_ai.database.ids import next_id
from offline_ai.database.models import DocumentTopic, Event, Topic
from offline_ai.database.repositories import ClaimRepository, EntityRepository, RelationshipRepository
from offline_ai.database.session import Database
from offline_ai.extraction.claims import (
    PRED_ASSOCIATED_WITH,
    PRED_HAD_MEETING,
    PRED_KILLED,
    PRED_LOCATED_IN,
    PRED_MET_WITH,
    PRED_MISSING_SINCE,
    PRED_OCCURRED_ON,
    PRED_TRAVELED_TO,
    ClaimExtractor,
)
from offline_ai.extraction.entities import EntityExtractor
from offline_ai.extraction.events import EventExtractor
from offline_ai.extraction.topics import TopicExtractor
from offline_ai.utils.hashing import normalize_text
from offline_ai.utils.logging import get_logger
from offline_ai.utils.persian import extract_jalali_dates

logger = get_logger(__name__)

_PERSON_PREDICATES = {
    PRED_KILLED,
    PRED_MET_WITH,
    PRED_ASSOCIATED_WITH,
    PRED_HAD_MEETING,
    PRED_MISSING_SINCE,
    PRED_TRAVELED_TO,
    PRED_OCCURRED_ON,
    PRED_LOCATED_IN,
}

_DATE_PREDICATES = {PRED_MISSING_SINCE, PRED_OCCURRED_ON, PRED_HAD_MEETING}
_PLACE_PREDICATES = {PRED_TRAVELED_TO, PRED_LOCATED_IN}


class ExtractionPipeline:
    def __init__(self, db: Database) -> None:
        self.db = db
        self.entities = EntityExtractor()
        self.claims = ClaimExtractor()
        self.topics = TopicExtractor()
        self.events = EventExtractor()

    def process_document(self, document_id: str, text: str) -> dict[str, Any]:
        ents = self.entities.extract(text)
        claims = self.claims.extract(text)
        topics = self.topics.extract(text)
        events = self.events.extract(text)

        entity_ids: list[str] = []
        claim_ids: list[str] = []
        topic_ids: list[str] = []
        event_ids: list[str] = []
        relationship_ids: list[str] = []

        with self.db.session() as session:
            erepo = EntityRepository(session)
            crepo = ClaimRepository(session)
            rrepo = RelationshipRepository(session)

            entity_by_norm: dict[str, str] = {}

            def ensure_entity(value: str, etype: str) -> str | None:
                value = " ".join((value or "").split())
                if not value or len(value) < 2:
                    return None
                ent = erepo.upsert(value=value, entity_type=etype)
                entity_by_norm[ent.normalized_value] = ent.entity_id
                if ent.entity_id not in entity_ids:
                    entity_ids.append(ent.entity_id)
                return ent.entity_id

            for e in ents:
                ent = erepo.upsert(
                    value=e.value,
                    entity_type=e.entity_type,
                    normalized_value=e.normalized_value,
                )
                erepo.add_mention(
                    entity_id=ent.entity_id,
                    document_id=document_id,
                    confidence=e.confidence,
                    start_offset=e.start_offset,
                    end_offset=e.end_offset,
                    surface_form=e.value,
                )
                entity_by_norm[ent.normalized_value] = ent.entity_id
                entity_ids.append(ent.entity_id)

            for c in claims:
                claim = crepo.create(
                    subject=c.subject,
                    predicate=c.predicate,
                    object_=c.object,
                    source_document_id=document_id,
                    confidence=c.confidence,
                    source_span=c.source_span,
                    start_offset=c.start_offset,
                    end_offset=c.end_offset,
                )
                claim_ids.append(claim.claim_id)

                # Materialize graph edges for durable multi-hop recall
                if c.predicate not in _PERSON_PREDICATES:
                    continue
                sub_id = ensure_entity(c.subject, "PERSON")
                if c.predicate in _DATE_PREDICATES and extract_jalali_dates(c.object):
                    obj_id = ensure_entity(c.object, "DATE")
                elif c.predicate in _PLACE_PREDICATES:
                    obj_id = ensure_entity(c.object, "LOCATION")
                else:
                    obj_id = ensure_entity(c.object, "PERSON")
                if not sub_id or not obj_id or sub_id == obj_id:
                    continue
                rel = rrepo.create(
                    subject_entity_id=sub_id,
                    relation_type=c.predicate,
                    object_entity_id=obj_id,
                    source_document_id=document_id,
                    confidence=c.confidence,
                    metadata={"claim_id": claim.claim_id, "span": c.source_span},
                )
                relationship_ids.append(rel.relationship_id)

            for name, score in topics:
                norm = normalize_text(name)
                from sqlalchemy import select

                topic = session.execute(
                    select(Topic).where(Topic.normalized_name == norm)
                ).scalar_one_or_none()
                if topic is None:
                    topic = Topic(
                        topic_id=next_id(session, "TOPIC"),
                        name=name,
                        normalized_name=norm,
                    )
                    session.add(topic)
                    session.flush()
                existing_link = session.execute(
                    select(DocumentTopic).where(
                        DocumentTopic.document_id == document_id,
                        DocumentTopic.topic_id == topic.topic_id,
                    )
                ).scalar_one_or_none()
                if existing_link is None:
                    session.add(
                        DocumentTopic(document_id=document_id, topic_id=topic.topic_id)
                    )
                topic_ids.append(topic.topic_id)

            for ev in events:
                event = Event(
                    event_id=next_id(session, "EVENT"),
                    name=ev.name,
                    description=ev.description,
                    source_document_id=document_id,
                    metadata_json={"date_hint": ev.date_hint, "confidence": ev.confidence},
                )
                session.add(event)
                session.flush()
                event_ids.append(event.event_id)

            session.commit()

        logger.info(
            "Extraction complete",
            extra={"event": "extract", "document_id": document_id, "component": "extraction"},
        )
        return {
            "document_id": document_id,
            "entities": entity_ids,
            "claims": claim_ids,
            "topics": topic_ids,
            "events": event_ids,
            "relationships": relationship_ids,
        }

    def reprocess_all(self) -> dict[str, int]:
        """Rebuild claims/relationships from all stored documents (keeps originals)."""
        from sqlalchemy import delete, select

        from offline_ai.database.models import Claim, Document, Relationship

        with self.db.session() as session:
            docs = list(session.execute(select(Document)).scalars())
            session.execute(delete(Relationship))
            session.execute(delete(Claim))
            session.commit()

        counts = {"documents": 0, "claims": 0, "relationships": 0}
        for doc in docs:
            result = self.process_document(doc.document_id, doc.original_text or "")
            counts["documents"] += 1
            counts["claims"] += len(result.get("claims") or [])
            counts["relationships"] += len(result.get("relationships") or [])
        return counts
