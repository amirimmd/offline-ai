"""CSV ingestion (streaming-friendly)."""

from __future__ import annotations

import csv
from collections.abc import Iterator
from pathlib import Path

from offline_ai.ingestion.base import IngestItem
from offline_ai.ingestion.json_ingest import item_from_mapping


def iter_csv_items(path: Path) -> Iterator[IngestItem]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader, start=2):
            mapping = {k: v for k, v in dict(row).items() if v is not None and str(v).strip()}
            if not mapping:
                continue
            try:
                item = item_from_mapping(mapping, default_source=path.name)
            except ValueError:
                # Join all cells when no dedicated text column
                text = " | ".join(str(v).strip() for v in mapping.values() if str(v).strip())
                if not text:
                    continue
                item = IngestItem(
                    text=text,
                    source=path.name,
                    source_type="csv",
                    source_name=path.name,
                    collection_method="file",
                    metadata={"row": i, **mapping},
                )
            item.metadata = {**(item.metadata or {}), "row": i}
            item.source_type = item.source_type or "csv"
            item.collection_method = "file"
            yield item


def load_csv_items(path: Path) -> list[IngestItem]:
    return list(iter_csv_items(path))


def count_csv_data_rows(path: Path) -> int:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        try:
            next(reader)  # header
        except StopIteration:
            return 0
        return sum(1 for row in reader if any(c.strip() for c in row if c))
