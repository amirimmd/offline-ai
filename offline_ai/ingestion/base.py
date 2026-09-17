"""Ingestion base types and orchestration."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from offline_ai.database.repositories import IngestionRunRepository
from offline_ai.database.session import Database
from offline_ai.memory.raw import RawMemory
from offline_ai.utils.logging import get_logger
from offline_ai.utils.paths import safe_join

logger = get_logger(__name__)


@dataclass
class IngestItem:
    text: str
    source: str = "unknown"
    source_url: str | None = None
    author: str | None = None
    timestamp: datetime | None = None
    language: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    source_type: str | None = None
    source_name: str | None = None
    collection_method: str | None = None


@dataclass
class IngestionStats:
    run_id: str
    documents_received: int = 0
    documents_added: int = 0
    duplicates: int = 0
    errors: int = 0
    entities: int = 0
    claims: int = 0
    embeddings: int = 0
    duration_ms: int = 0
    document_ids: list[str] = field(default_factory=list)
    error_messages: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "documents_received": self.documents_received,
            "documents_added": self.documents_added,
            "duplicates": self.duplicates,
            "errors": self.errors,
            "entities": self.entities,
            "claims": self.claims,
            "embeddings": self.embeddings,
            "duration_ms": self.duration_ms,
            "document_ids": self.document_ids,
            "error_messages": self.error_messages,
        }


class IngestionPipeline:
    """Batch ingest items into raw memory; optional post-hooks for embed/extract."""

    def __init__(
        self,
        db: Database,
        raw_memory: RawMemory,
        *,
        after_document: Callable[[Any, bool], None] | None = None,
    ) -> None:
        self.db = db
        self.raw = raw_memory
        self.after_document = after_document

    def ingest_items(
        self,
        items: list[IngestItem],
        *,
        source_description: str | None = None,
    ) -> IngestionStats:
        started = time.perf_counter()
        with self.db.session() as session:
            run = IngestionRunRepository(session).start(source_description)
            session.commit()
            run_id = run.run_id

        stats = IngestionStats(run_id=run_id)
        for item in items:
            stats.documents_received += 1
            try:
                doc, created, _chunks = self.raw.ingest_document(
                    text=item.text,
                    source=item.source,
                    source_url=item.source_url,
                    author=item.author,
                    timestamp=item.timestamp,
                    language=item.language,
                    metadata=item.metadata,
                    source_type=item.source_type,
                    source_name=item.source_name,
                    collection_method=item.collection_method,
                    ingestion_run_id=run_id,
                )
                if created:
                    stats.documents_added += 1
                    stats.document_ids.append(doc.document_id)
                else:
                    stats.duplicates += 1
                    stats.document_ids.append(doc.document_id)
                if self.after_document:
                    self.after_document(doc, created)
            except Exception as exc:  # noqa: BLE001 - collect per-item errors
                stats.errors += 1
                stats.error_messages.append(str(exc))
                logger.error(
                    "Ingestion item failed",
                    extra={"event": "ingest_error", "error": str(exc), "component": "ingestion"},
                )

        stats.duration_ms = int((time.perf_counter() - started) * 1000)
        with self.db.session() as session:
            from sqlalchemy import select

            from offline_ai.database.models import IngestionRun

            run_obj = session.execute(
                select(IngestionRun).where(IngestionRun.run_id == run_id)
            ).scalar_one()
            IngestionRunRepository(session).finish(
                run_obj,
                documents_received=stats.documents_received,
                documents_added=stats.documents_added,
                duplicates=stats.duplicates,
                errors=stats.errors,
                entities=stats.entities,
                claims=stats.claims,
                embeddings=stats.embeddings,
                duration_ms=stats.duration_ms,
                status="completed" if stats.errors == 0 else "completed_with_errors",
                error_log="\n".join(stats.error_messages) if stats.error_messages else None,
            )
            session.commit()
        return stats

    def ingest_file(self, path: Path, workspace_root: Path | None = None) -> IngestionStats:
        path = Path(path)
        if workspace_root is not None:
            # Allow absolute paths; reject traversal when joining under workspace.
            if not path.is_absolute():
                path = safe_join(workspace_root, path.as_posix())
        if not path.exists():
            raise FileNotFoundError(path)
        suffix = path.suffix.lower()
        if suffix == ".json":
            from offline_ai.ingestion.json_ingest import load_json_items
            items = load_json_items(path)
        elif suffix == ".jsonl":
            from offline_ai.ingestion.json_ingest import load_jsonl_items
            items = load_jsonl_items(path)
        elif suffix == ".csv":
            from offline_ai.ingestion.csv_ingest import load_csv_items
            items = load_csv_items(path)
        elif suffix in {".txt", ".md", ".markdown"}:
            from offline_ai.ingestion.text_ingest import load_text_file
            items = [load_text_file(path)]
        else:
            raise ValueError(f"Unsupported ingestion format: {suffix}")
        return self.ingest_items(items, source_description=str(path))
