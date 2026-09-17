"""Repository layer for structured memory."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from sqlalchemy import select
from sqlalchemy.orm import Session

from offline_ai.database.ids import next_id
from offline_ai.database.models import (
    Claim,
    Document,
    DocumentChunk,
    Entity,
    EntityMention,
    IngestionRun,
    Relationship,
    utcnow,
)
from offline_ai.utils.hashing import content_hash, normalized_content_hash, source_hash


class DocumentRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def get_by_document_id(self, document_id: str) -> Document | None:
        return self.session.execute(
            select(Document).where(Document.document_id == document_id)
        ).scalar_one_or_none()

    def get_by_source_hash(self, sh: str) -> Document | None:
        return self.session.execute(
            select(Document).where(Document.source_hash == sh)
        ).scalar_one_or_none()

    def get_by_content_hash(self, ch: str) -> list[Document]:
        return list(
            self.session.execute(select(Document).where(Document.content_hash == ch)).scalars()
        )

    def create(
        self,
        *,
        text: str,
        source: str,
        source_url: str | None = None,
        author: str | None = None,
        timestamp: datetime | None = None,
        language: str | None = None,
        metadata: dict[str, Any] | None = None,
        source_type: str | None = None,
        source_name: str | None = None,
        collection_method: str | None = None,
        ingestion_run_id: str | None = None,
        document_id: str | None = None,
    ) -> tuple[Document, bool]:
        """
        Insert immutable document. Returns (document, created).
        If exact source_hash exists, returns existing without modifying original_text.
        """
        sh = source_hash(source, source_url, text)
        existing = self.get_by_source_hash(sh)
        if existing is not None:
            return existing, False

        doc_id = document_id or next_id(self.session, "DOC")
        doc = Document(
            document_id=doc_id,
            source=source,
            source_url=source_url,
            author=author,
            timestamp=timestamp,
            ingestion_timestamp=utcnow(),
            original_text=text,
            language=language,
            content_hash=content_hash(text),
            normalized_hash=normalized_content_hash(text),
            source_hash=sh,
            metadata_json=metadata or {},
            source_type=source_type,
            source_name=source_name or source,
            collection_method=collection_method,
            ingestion_run_id=ingestion_run_id,
        )
        self.session.add(doc)
        self.session.flush()
        return doc, True

    def list_all(self, limit: int = 100, offset: int = 0) -> list[Document]:
        return list(
            self.session.execute(
                select(Document).order_by(Document.id).offset(offset).limit(limit)
            ).scalars()
        )

    def count(self) -> int:
        from sqlalchemy import func

        return int(self.session.execute(select(func.count()).select_from(Document)).scalar_one())


class ChunkRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def add_chunks(
        self,
        document_id: str,
        chunks: list[tuple[int, int, int, str]],
    ) -> list[DocumentChunk]:
        """chunks: list of (chunk_index, start, end, text)."""
        created: list[DocumentChunk] = []
        for idx, start, end, text in chunks:
            ch = DocumentChunk(
                chunk_id=next_id(self.session, "CHK"),
                document_id=document_id,
                chunk_index=idx,
                start_offset=start,
                end_offset=end,
                text=text,
                content_hash=content_hash(text),
            )
            self.session.add(ch)
            created.append(ch)
        self.session.flush()
        return created

    def get_by_document(self, document_id: str) -> list[DocumentChunk]:
        return list(
            self.session.execute(
                select(DocumentChunk)
                .where(DocumentChunk.document_id == document_id)
                .order_by(DocumentChunk.chunk_index)
            ).scalars()
        )


class IngestionRunRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def start(self, source_description: str | None = None) -> IngestionRun:
        run = IngestionRun(
            run_id=next_id(self.session, "RUN"),
            source_description=source_description,
            status="running",
        )
        self.session.add(run)
        self.session.flush()
        return run

    def finish(
        self,
        run: IngestionRun,
        *,
        documents_received: int,
        documents_added: int,
        duplicates: int,
        errors: int,
        entities: int = 0,
        claims: int = 0,
        embeddings: int = 0,
        duration_ms: int | None = None,
        status: str = "completed",
        error_log: str | None = None,
    ) -> IngestionRun:
        run.documents_received = documents_received
        run.documents_added = documents_added
        run.duplicates = duplicates
        run.errors = errors
        run.entities = entities
        run.claims = claims
        run.embeddings = embeddings
        run.duration_ms = duration_ms
        run.status = status
        run.error_log = error_log
        run.finished_at = utcnow()
        self.session.flush()
        return run


class EntityRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def upsert(
        self,
        *,
        value: str,
        entity_type: str,
        normalized_value: str | None = None,
    ) -> Entity:
        from offline_ai.utils.hashing import normalize_text

        norm = normalized_value or normalize_text(value)
        existing = self.session.execute(
            select(Entity).where(
                Entity.entity_type == entity_type, Entity.normalized_value == norm
            )
        ).scalar_one_or_none()
        if existing:
            return existing
        ent = Entity(
            entity_id=next_id(self.session, "ENTITY"),
            value=value,
            normalized_value=norm,
            entity_type=entity_type,
        )
        self.session.add(ent)
        self.session.flush()
        return ent

    def get(self, entity_id: str) -> Entity | None:
        return self.session.execute(
            select(Entity).where(Entity.entity_id == entity_id)
        ).scalar_one_or_none()

    def add_mention(
        self,
        *,
        entity_id: str,
        document_id: str,
        confidence: float = 0.0,
        start_offset: int | None = None,
        end_offset: int | None = None,
        surface_form: str | None = None,
    ) -> EntityMention:
        m = EntityMention(
            entity_id=entity_id,
            document_id=document_id,
            confidence=confidence,
            start_offset=start_offset,
            end_offset=end_offset,
            surface_form=surface_form,
        )
        self.session.add(m)
        self.session.flush()
        return m


class ClaimRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def create(
        self,
        *,
        subject: str,
        predicate: str,
        object_: str,
        source_document_id: str,
        confidence: float = 0.0,
        source_span: str | None = None,
        start_offset: int | None = None,
        end_offset: int | None = None,
    ) -> Claim:
        claim = Claim(
            claim_id=next_id(self.session, "CLAIM"),
            subject=subject,
            predicate=predicate,
            object=object_,
            confidence=confidence,
            source_document_id=source_document_id,
            source_span=source_span,
            start_offset=start_offset,
            end_offset=end_offset,
        )
        self.session.add(claim)
        self.session.flush()
        return claim

    def get(self, claim_id: str) -> Claim | None:
        return self.session.execute(
            select(Claim).where(Claim.claim_id == claim_id)
        ).scalar_one_or_none()


class RelationshipRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def create(
        self,
        *,
        subject_entity_id: str,
        relation_type: str,
        object_entity_id: str,
        source_document_id: str,
        confidence: float = 0.0,
        metadata: dict[str, Any] | None = None,
    ) -> Relationship:
        rel = Relationship(
            relationship_id=next_id(self.session, "REL"),
            subject_entity_id=subject_entity_id,
            relation_type=relation_type,
            object_entity_id=object_entity_id,
            source_document_id=source_document_id,
            confidence=confidence,
            metadata_json=metadata or {},
        )
        self.session.add(rel)
        self.session.flush()
        return rel
