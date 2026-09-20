"""Bulk / streaming helpers for large knowledge corpora."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from offline_ai.ingestion.base import IngestItem

# When pasted text has this many non-empty lines, treat each line as one document.
BULK_LINE_THRESHOLD = 2
# Soft cap for how many document_ids we keep in the stats payload (memory).
MAX_STATS_DOCUMENT_IDS = 200


def split_text_to_lines(text: str) -> list[str]:
    """Split pasted / file text into non-empty knowledge lines."""
    lines: list[str] = []
    for raw in (text or "").replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        line = raw.strip()
        if line:
            lines.append(line)
    return lines


def should_split_multiline(text: str, *, threshold: int = BULK_LINE_THRESHOLD) -> bool:
    return len(split_text_to_lines(text)) >= threshold


def iter_line_items(
    lines: Iterable[str],
    *,
    source: str = "bulk_lines",
    source_type: str = "bulk",
) -> Iterator[IngestItem]:
    from offline_ai.ingestion.base import IngestItem as _IngestItem

    for i, line in enumerate(lines, start=1):
        text = (line or "").strip()
        if not text:
            continue
        yield _IngestItem(
            text=text,
            source=source,
            source_type=source_type,
            source_name=source,
            collection_method="bulk",
            metadata={"line_no": i},
        )


def iter_text_file_lines(path: Path, *, source: str | None = None) -> Iterator[IngestItem]:
    """Stream a .txt/.md file: one non-empty line → one document (for huge corpora)."""
    from offline_ai.ingestion.base import IngestItem as _IngestItem

    src = source or path.name
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for i, raw in enumerate(fh, start=1):
            line = raw.strip()
            if not line:
                continue
            yield _IngestItem(
                text=line,
                source=src,
                source_type="text_lines",
                source_name=path.name,
                collection_method="file",
                metadata={"line_no": i, "path": str(path)},
            )


def format_progress(
    done: int, total: int | None, *, added: int, duplicates: int, errors: int
) -> str:
    if total and total > 0:
        pct = min(100, int(100 * done / total))
        return (
            f"ذخیره دسته‌ای: {done:,} / {total:,} ({pct}%) "
            f"· جدید {added:,} · تکراری {duplicates:,}"
        )
    return (
        f"ذخیره دسته‌ای: {done:,} ردیف · جدید {added:,} "
        f"· تکراری {duplicates:,} · خطا {errors:,}"
    )


def progress_message(payload: dict[str, Any]) -> str:
    return format_progress(
        int(payload.get("done", 0) or 0),
        payload.get("total"),
        added=int(payload.get("added", 0) or 0),
        duplicates=int(payload.get("duplicates", 0) or 0),
        errors=int(payload.get("errors", 0) or 0),
    )
