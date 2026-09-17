"""Extraction pipeline orchestrator."""

from __future__ import annotations

from typing import Any

from offline_ai.database.ids import next_id
from offline_ai.database.models import DocumentTopic, Event, Topic
from offline_ai.database.repositories import ClaimRepository, EntityRepository
from offline_ai.database.session import Database
from offline_ai.extraction.claims import ClaimExtractor
from offline_ai.extraction.entities import EntityExtractor
from offline_ai.extraction.events import EventExtractor
from offline_ai.extraction.topics import TopicExtractor
from offline_ai.utils.hashing import normalize_text
from offline_ai.utils.logging import get_logger

logger = get_logger(__name__)


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

        with self.db.session() as session:
            erepo = EntityRepository(session)
            crepo = ClaimRepository(session)
            for e in ents:
                ent = erepo.upsert(
                    value=e.value, entity_type=e.entity_type, normalized_value=e.normalized_value
                )
                erepo.add_mention(
                    entity_id=ent.entity_id,
                    document_id=document_id,
                    confidence=e.confidence,
                    start_offset=e.start_offset,
                    end_offset=e.end_offset,
                    surface_form=e.value,
                )
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
        }
