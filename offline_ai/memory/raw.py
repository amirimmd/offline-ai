"""Raw immutable document memory."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from offline_ai.database.models import Document
from offline_ai.database.repositories import ChunkRepository, DocumentRepository
from offline_ai.database.session import Database
from offline_ai.utils.hashing import content_hash, normalize_text
from offline_ai.utils.logging import get_logger

logger = get_logger(__name__)


def chunk_text(
    text: str,
    *,
    min_chars_for_chunking: int = 1200,
    chunk_size: int = 800,
    chunk_overlap: int = 120,
) -> list[tuple[int, int, int, str]]:
    """
    Return list of (index, start_offset, end_offset, chunk_text).
    Short texts (e.g. tweets) are stored as a single chunk.
    """
    if len(text) <= min_chars_for_chunking:
        return [(0, 0, len(text), text)]

    chunks: list[tuple[int, int, int, str]] = []
    start = 0
    idx = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        if end < n:
            # try to break on whitespace
            window = text[start:end]
            br = window.rfind(" ")
            if br > chunk_size // 2:
                end = start + br
        chunk = text[start:end]
        chunks.append((idx, start, end, chunk))
        idx += 1
        if end >= n:
            break
        start = max(end - chunk_overlap, start + 1)
    return chunks


class RawMemory:
    """Immutable source document store."""

    def __init__(self, db: Database, chunking: dict[str, Any] | None = None) -> None:
        self.db = db
        self.chunking = chunking or {}

    def ingest_document(
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
    ) -> tuple[Document, bool, list]:
        """
        Persist original text immutably.
        Returns (document, created, chunks).
        """
        if not text or not str(text).strip():
            raise ValueError("Document text must be non-empty")

        with self.db.session() as session:
            repo = DocumentRepository(session)
            doc, created = repo.create(
                text=text,
                source=source,
                source_url=source_url,
                author=author,
                timestamp=timestamp,
                language=language,
                metadata=metadata,
                source_type=source_type,
                source_name=source_name,
                collection_method=collection_method,
                ingestion_run_id=ingestion_run_id,
            )
            chunks = []
            if created:
                parts = chunk_text(
                    text,
                    min_chars_for_chunking=int(
                        self.chunking.get("min_chars_for_chunking", 1200)
                    ),
                    chunk_size=int(self.chunking.get("chunk_size", 800)),
                    chunk_overlap=int(self.chunking.get("chunk_overlap", 120)),
                )
                chunks = ChunkRepository(session).add_chunks(doc.document_id, parts)
                logger.info(
                    "Document ingested",
                    extra={
                        "event": "ingest_document",
                        "document_id": doc.document_id,
                        "component": "raw_memory",
                    },
                )
            session.commit()
            session.refresh(doc)
            return doc, created, chunks

    def get_document(self, document_id: str) -> Document | None:
        with self.db.session() as session:
            return DocumentRepository(session).get_by_document_id(document_id)

    def find_exact_duplicates(self, text: str) -> list[Document]:
        ch = content_hash(text)
        with self.db.session() as session:
            return DocumentRepository(session).get_by_content_hash(ch)

    def find_normalized_duplicates(self, text: str) -> list[Document]:
        from sqlalchemy import select

        from offline_ai.database.models import Document as Doc
        from offline_ai.utils.hashing import normalized_content_hash

        nh = normalized_content_hash(text)
        with self.db.session() as session:
            return list(
                session.execute(select(Doc).where(Doc.normalized_hash == nh)).scalars()
            )

    @staticmethod
    def document_to_dict(doc: Document) -> dict[str, Any]:
        return {
            "document_id": doc.document_id,
            "source": doc.source,
            "source_url": doc.source_url,
            "author": doc.author,
            "timestamp": doc.timestamp.isoformat() if doc.timestamp else None,
            "ingestion_timestamp": doc.ingestion_timestamp.isoformat()
            if doc.ingestion_timestamp
            else None,
            "original_text": doc.original_text,
            "language": doc.language,
            "content_hash": doc.content_hash,
            "metadata": doc.metadata_json,
            "verification_status": doc.verification_status,
        }
