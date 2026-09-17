"""SQLAlchemy ORM models for structured memory."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    event,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class Document(Base):
    __tablename__ = "documents"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    document_id: Mapped[str] = mapped_column(String(32), unique=True, nullable=False, index=True)
    source: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    source_url: Mapped[Optional[str]] = mapped_column(String(2048), nullable=True)
    author: Mapped[Optional[str]] = mapped_column(String(512), nullable=True, index=True)
    timestamp: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True, index=True)
    ingestion_timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)
    original_text: Mapped[str] = mapped_column(Text, nullable=False)
    language: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    normalized_hash: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    source_hash: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    metadata_json: Mapped[dict[str, Any]] = mapped_column("metadata", JSON, default=dict)
    source_type: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    source_name: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    collection_method: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    verification_status: Mapped[str] = mapped_column(String(32), default="unverified")
    ingestion_run_id: Mapped[Optional[str]] = mapped_column(
        String(32), ForeignKey("ingestion_runs.run_id"), nullable=True
    )

    chunks: Mapped[list[DocumentChunk]] = relationship(back_populates="document")
    claims: Mapped[list[Claim]] = relationship(back_populates="document")

    __table_args__ = (
        UniqueConstraint("source_hash", name="uq_documents_source_hash"),
    )


class DocumentChunk(Base):
    __tablename__ = "document_chunks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    chunk_id: Mapped[str] = mapped_column(String(32), unique=True, nullable=False, index=True)
    document_id: Mapped[str] = mapped_column(
        String(32), ForeignKey("documents.document_id"), nullable=False, index=True
    )
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    start_offset: Mapped[int] = mapped_column(Integer, nullable=False)
    end_offset: Mapped[int] = mapped_column(Integer, nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)

    document: Mapped[Document] = relationship(back_populates="chunks")

    __table_args__ = (
        UniqueConstraint("document_id", "chunk_index", name="uq_chunk_doc_index"),
    )


class Source(Base):
    __tablename__ = "sources"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    source_id: Mapped[str] = mapped_column(String(32), unique=True, nullable=False)
    source_type: Mapped[Optional[str]] = mapped_column(String(64))
    source_name: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    source_url: Mapped[Optional[str]] = mapped_column(String(2048))
    reliability_tier: Mapped[Optional[str]] = mapped_column(String(64))
    metadata_json: Mapped[dict[str, Any]] = mapped_column("metadata", JSON, default=dict)


class Author(Base):
    __tablename__ = "authors"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    author_id: Mapped[str] = mapped_column(String(32), unique=True, nullable=False)
    name: Mapped[str] = mapped_column(String(512), nullable=False, index=True)
    normalized_name: Mapped[str] = mapped_column(String(512), nullable=False, index=True)
    metadata_json: Mapped[dict[str, Any]] = mapped_column("metadata", JSON, default=dict)


class Entity(Base):
    __tablename__ = "entities"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    entity_id: Mapped[str] = mapped_column(String(32), unique=True, nullable=False, index=True)
    value: Mapped[str] = mapped_column(String(512), nullable=False)
    normalized_value: Mapped[str] = mapped_column(String(512), nullable=False, index=True)
    entity_type: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    metadata_json: Mapped[dict[str, Any]] = mapped_column("metadata", JSON, default=dict)

    __table_args__ = (
        UniqueConstraint("entity_type", "normalized_value", name="uq_entity_type_norm"),
        Index("ix_entities_type_value", "entity_type", "value"),
    )


class EntityMention(Base):
    __tablename__ = "entity_mentions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    entity_id: Mapped[str] = mapped_column(
        String(32), ForeignKey("entities.entity_id"), nullable=False, index=True
    )
    document_id: Mapped[str] = mapped_column(
        String(32), ForeignKey("documents.document_id"), nullable=False, index=True
    )
    confidence: Mapped[float] = mapped_column(Float, default=0.0)
    start_offset: Mapped[Optional[int]] = mapped_column(Integer)
    end_offset: Mapped[Optional[int]] = mapped_column(Integer)
    surface_form: Mapped[Optional[str]] = mapped_column(String(512))


class DocumentEntity(Base):
    __tablename__ = "document_entities"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    document_id: Mapped[str] = mapped_column(
        String(32), ForeignKey("documents.document_id"), nullable=False, index=True
    )
    entity_id: Mapped[str] = mapped_column(
        String(32), ForeignKey("entities.entity_id"), nullable=False, index=True
    )

    __table_args__ = (
        UniqueConstraint("document_id", "entity_id", name="uq_doc_entity"),
    )


class Claim(Base):
    __tablename__ = "claims"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    claim_id: Mapped[str] = mapped_column(String(32), unique=True, nullable=False, index=True)
    subject: Mapped[str] = mapped_column(String(512), nullable=False)
    predicate: Mapped[str] = mapped_column(String(512), nullable=False)
    object: Mapped[str] = mapped_column(String(1024), nullable=False)
    confidence: Mapped[float] = mapped_column(Float, default=0.0)
    source_document_id: Mapped[str] = mapped_column(
        String(32), ForeignKey("documents.document_id"), nullable=False, index=True
    )
    source_span: Mapped[Optional[str]] = mapped_column(Text)
    start_offset: Mapped[Optional[int]] = mapped_column(Integer)
    end_offset: Mapped[Optional[int]] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    document: Mapped[Document] = relationship(back_populates="claims")


class DocumentClaim(Base):
    __tablename__ = "document_claims"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    document_id: Mapped[str] = mapped_column(
        String(32), ForeignKey("documents.document_id"), nullable=False, index=True
    )
    claim_id: Mapped[str] = mapped_column(
        String(32), ForeignKey("claims.claim_id"), nullable=False, index=True
    )

    __table_args__ = (UniqueConstraint("document_id", "claim_id", name="uq_doc_claim"),)


class Topic(Base):
    __tablename__ = "topics"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    topic_id: Mapped[str] = mapped_column(String(32), unique=True, nullable=False)
    name: Mapped[str] = mapped_column(String(255), nullable=False, unique=True, index=True)
    normalized_name: Mapped[str] = mapped_column(String(255), nullable=False, index=True)


class DocumentTopic(Base):
    __tablename__ = "document_topics"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    document_id: Mapped[str] = mapped_column(
        String(32), ForeignKey("documents.document_id"), nullable=False, index=True
    )
    topic_id: Mapped[str] = mapped_column(
        String(32), ForeignKey("topics.topic_id"), nullable=False, index=True
    )

    __table_args__ = (UniqueConstraint("document_id", "topic_id", name="uq_doc_topic"),)


class Event(Base):
    __tablename__ = "events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    event_id: Mapped[str] = mapped_column(String(32), unique=True, nullable=False, index=True)
    name: Mapped[str] = mapped_column(String(512), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text)
    event_time: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), index=True)
    source_document_id: Mapped[Optional[str]] = mapped_column(
        String(32), ForeignKey("documents.document_id"), index=True
    )
    metadata_json: Mapped[dict[str, Any]] = mapped_column("metadata", JSON, default=dict)


class Relationship(Base):
    __tablename__ = "relationships"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    relationship_id: Mapped[str] = mapped_column(String(32), unique=True, nullable=False)
    subject_entity_id: Mapped[str] = mapped_column(String(32), ForeignKey("entities.entity_id"), index=True)
    relation_type: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    object_entity_id: Mapped[str] = mapped_column(String(32), ForeignKey("entities.entity_id"), index=True)
    confidence: Mapped[float] = mapped_column(Float, default=0.0)
    source_document_id: Mapped[str] = mapped_column(
        String(32), ForeignKey("documents.document_id"), nullable=False, index=True
    )
    metadata_json: Mapped[dict[str, Any]] = mapped_column("metadata", JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class IngestionRun(Base):
    __tablename__ = "ingestion_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(String(32), unique=True, nullable=False, index=True)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    finished_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    source_description: Mapped[Optional[str]] = mapped_column(String(512))
    documents_received: Mapped[int] = mapped_column(Integer, default=0)
    documents_added: Mapped[int] = mapped_column(Integer, default=0)
    duplicates: Mapped[int] = mapped_column(Integer, default=0)
    errors: Mapped[int] = mapped_column(Integer, default=0)
    entities: Mapped[int] = mapped_column(Integer, default=0)
    claims: Mapped[int] = mapped_column(Integer, default=0)
    embeddings: Mapped[int] = mapped_column(Integer, default=0)
    duration_ms: Mapped[Optional[int]] = mapped_column(Integer)
    status: Mapped[str] = mapped_column(String(32), default="running")
    error_log: Mapped[Optional[str]] = mapped_column(Text)


class Citation(Base):
    __tablename__ = "citations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    citation_id: Mapped[str] = mapped_column(String(32), unique=True, nullable=False, index=True)
    answer_id: Mapped[Optional[str]] = mapped_column(String(32), index=True)
    target_type: Mapped[str] = mapped_column(String(32), nullable=False)  # document|claim|entity|event
    target_id: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    evidence_span: Mapped[Optional[str]] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class ModelVersion(Base):
    __tablename__ = "model_versions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    model_key: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    role: Mapped[str] = mapped_column(String(64), nullable=False)
    local_path: Mapped[str] = mapped_column(String(1024), nullable=False)
    version_label: Mapped[str] = mapped_column(String(128), nullable=False)
    activated_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    metadata_json: Mapped[dict[str, Any]] = mapped_column("metadata", JSON, default=dict)


class AdapterVersion(Base):
    __tablename__ = "adapter_versions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    adapter_id: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    name: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    path: Mapped[str] = mapped_column(String(1024), nullable=False)
    base_model_key: Mapped[str] = mapped_column(String(128), nullable=False)
    status: Mapped[str] = mapped_column(String(32), default="created")
    metrics_json: Mapped[dict[str, Any]] = mapped_column("metrics", JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    activated_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    is_active: Mapped[bool] = mapped_column(Boolean, default=False)


class Feedback(Base):
    __tablename__ = "feedback"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    feedback_id: Mapped[str] = mapped_column(String(32), unique=True, nullable=False)
    answer_id: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    rating: Mapped[Optional[int]] = mapped_column(Integer)
    correction: Mapped[Optional[str]] = mapped_column(Text)
    original_answer: Mapped[Optional[str]] = mapped_column(Text)
    retrieved_document_ids: Mapped[list[Any]] = mapped_column(JSON, default=list)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class QueryRecord(Base):
    __tablename__ = "queries"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    query_id: Mapped[str] = mapped_column(String(32), unique=True, nullable=False)
    conversation_id: Mapped[Optional[str]] = mapped_column(String(32), index=True)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    metadata_json: Mapped[dict[str, Any]] = mapped_column("metadata", JSON, default=dict)


class AnswerRecord(Base):
    __tablename__ = "answers"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    answer_id: Mapped[str] = mapped_column(String(32), unique=True, nullable=False, index=True)
    query_id: Mapped[Optional[str]] = mapped_column(String(32), ForeignKey("queries.query_id"))
    answer_text: Mapped[str] = mapped_column(Text, nullable=False)
    structured_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    grounded: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class AuditLog(Base):
    __tablename__ = "audit_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    event: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    component: Mapped[Optional[str]] = mapped_column(String(64))
    details_json: Mapped[dict[str, Any]] = mapped_column("details", JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class Conversation(Base):
    __tablename__ = "conversations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    conversation_id: Mapped[str] = mapped_column(String(32), unique=True, nullable=False, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    metadata_json: Mapped[dict[str, Any]] = mapped_column("metadata", JSON, default=dict)


class ConversationMessage(Base):
    __tablename__ = "conversation_messages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    message_id: Mapped[str] = mapped_column(String(32), unique=True, nullable=False)
    conversation_id: Mapped[str] = mapped_column(
        String(32), ForeignKey("conversations.conversation_id"), nullable=False, index=True
    )
    role: Mapped[str] = mapped_column(String(32), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    promoted: Mapped[bool] = mapped_column(Boolean, default=False)


class IdSequence(Base):
    """Application-level ID counters (DOC-, CLAIM-, etc.)."""

    __tablename__ = "id_sequences"

    prefix: Mapped[str] = mapped_column(String(32), primary_key=True)
    next_value: Mapped[int] = mapped_column(Integer, nullable=False, default=1)


class VectorMapping(Base):
    """Maps FAISS row ids to application chunk/document ids."""

    __tablename__ = "vector_mappings"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    faiss_id: Mapped[int] = mapped_column(Integer, unique=True, nullable=False, index=True)
    chunk_id: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    document_id: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    embedding_model: Mapped[Optional[str]] = mapped_column(String(128))


def enable_sqlite_foreign_keys(engine) -> None:
    @event.listens_for(engine, "connect")
    def _set_sqlite_pragma(dbapi_conn, _connection_record):  # type: ignore[no-untyped-def]
        cursor = dbapi_conn.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()
