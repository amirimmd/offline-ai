"""Ingestion base types and orchestration."""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from offline_ai.database.repositories import IngestionRunRepository
from offline_ai.database.session import Database
from offline_ai.ingestion.bulk import MAX_STATS_DOCUMENT_IDS, format_progress
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
    total_hint: int | None = None

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
            "error_messages": self.error_messages[:50],
            "total_hint": self.total_hint,
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
        items: Iterable[IngestItem],
        *,
        source_description: str | None = None,
        on_progress: Callable[[dict[str, Any]], None] | None = None,
        progress_every: int = 25,
        total_hint: int | None = None,
    ) -> IngestionStats:
        started = time.perf_counter()
        with self.db.session() as session:
            run = IngestionRunRepository(session).start(source_description)
            session.commit()
            run_id = run.run_id

        stats = IngestionStats(run_id=run_id, total_hint=total_hint)

        def _emit(force: bool = False) -> None:
            if on_progress is None:
                return
            if not force and stats.documents_received % max(1, progress_every) != 0:
                return
            payload = {
                "phase": "ingest",
                "done": stats.documents_received,
                "total": total_hint,
                "added": stats.documents_added,
                "duplicates": stats.duplicates,
                "errors": stats.errors,
                "message": format_progress(
                    stats.documents_received,
                    total_hint,
                    added=stats.documents_added,
                    duplicates=stats.duplicates,
                    errors=stats.errors,
                ),
            }
            try:
                on_progress(payload)
            except Exception:  # noqa: BLE001
                logger.debug("ingest progress callback failed", exc_info=True)

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
                    if len(stats.document_ids) < MAX_STATS_DOCUMENT_IDS:
                        stats.document_ids.append(doc.document_id)
                else:
                    stats.duplicates += 1
                    if len(stats.document_ids) < MAX_STATS_DOCUMENT_IDS:
                        stats.document_ids.append(doc.document_id)
                if self.after_document:
                    self.after_document(doc, created)
            except Exception as exc:  # noqa: BLE001 - collect per-item errors
                stats.errors += 1
                if len(stats.error_messages) < 50:
                    stats.error_messages.append(str(exc))
                logger.error(
                    "Ingestion item failed",
                    extra={"event": "ingest_error", "error": str(exc), "component": "ingestion"},
                )
            _emit(force=False)

        _emit(force=True)
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

    def ingest_file(
        self,
        path: Path,
        workspace_root: Path | None = None,
        *,
        on_progress: Callable[[dict[str, Any]], None] | None = None,
        progress_every: int = 25,
        text_column: str | None = None,
        line_mode: bool | None = None,
    ) -> IngestionStats:
        path = Path(path)
        if workspace_root is not None:
            if not path.is_absolute():
                path = safe_join(workspace_root, path.as_posix())
        if not path.exists():
            raise FileNotFoundError(path)
        suffix = path.suffix.lower()
        total_hint: int | None = None
        items: Iterable[IngestItem]

        if suffix == ".json":
            from offline_ai.ingestion.json_ingest import load_json_items

            materialized = load_json_items(path)
            total_hint = len(materialized)
            items = materialized
        elif suffix == ".jsonl":
            from offline_ai.ingestion.json_ingest import load_jsonl_items

            # Stream-friendly alternative: still materialize for now (JSONL usually smaller);
            # for huge JSONL use line iterator below if needed.
            materialized = load_jsonl_items(path)
            total_hint = len(materialized)
            items = materialized
        elif suffix == ".csv":
            from offline_ai.ingestion.csv_ingest import count_csv_data_rows, iter_csv_items

            try:
                total_hint = count_csv_data_rows(path)
            except Exception:
                total_hint = None
            items = iter_csv_items(path)
        elif suffix in {".xlsx", ".xlsm"}:
            from offline_ai.ingestion.excel_ingest import count_excel_data_rows, iter_excel_items

            try:
                total_hint = count_excel_data_rows(path)
            except Exception:
                total_hint = None
            items = iter_excel_items(path, text_column=text_column)
        elif suffix in {".xls"}:
            raise ValueError(
                "Legacy .xls is not supported. Save as .xlsx or CSV and try again."
            )
        elif suffix in {".txt", ".md", ".markdown"}:
            # Large corpora: one line per document. Small notes: single document.
            from offline_ai.ingestion.bulk import iter_text_file_lines
            from offline_ai.ingestion.text_ingest import load_text_file

            if line_mode is None:
                # Heuristic: > 50 lines → line mode for bulk knowledge dumps
                with path.open("r", encoding="utf-8", errors="replace") as fh:
                    sample = sum(1 for _ in zip(fh, range(51)))
                line_mode = sample > 50
            if line_mode:
                # Count lines for progress (second pass); OK for large files on SSD
                with path.open("r", encoding="utf-8", errors="replace") as fh:
                    total_hint = sum(1 for raw in fh if raw.strip())
                items = iter_text_file_lines(path)
            else:
                items = [load_text_file(path)]
                total_hint = 1
        else:
            raise ValueError(
                f"Unsupported ingestion format: {suffix}. "
                "Use .xlsx, .csv, .json, .jsonl, .txt, or .md"
            )

        if on_progress is not None:
            try:
                on_progress(
                    {
                        "phase": "start",
                        "done": 0,
                        "total": total_hint,
                        "added": 0,
                        "duplicates": 0,
                        "errors": 0,
                        "message": f"شروع خواندن فایل: {path.name}"
                        + (f" (~{total_hint:,} ردیف)" if total_hint else ""),
                    }
                )
            except Exception:  # noqa: BLE001
                pass

        return self.ingest_items(
            items,
            source_description=str(path),
            on_progress=on_progress,
            progress_every=progress_every,
            total_hint=total_hint,
        )
