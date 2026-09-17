"""Plain text / markdown ingestion."""

from __future__ import annotations

from pathlib import Path

from offline_ai.ingestion.base import IngestItem


def load_text_file(path: Path) -> IngestItem:
    text = path.read_text(encoding="utf-8")
    return IngestItem(
        text=text,
        source=path.name,
        source_type="file",
        source_name=path.name,
        collection_method="file",
        metadata={"filename": path.name, "suffix": path.suffix},
    )
