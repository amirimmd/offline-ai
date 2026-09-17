"""CSV ingestion."""

from __future__ import annotations

import csv
from pathlib import Path

from offline_ai.ingestion.json_ingest import item_from_mapping
from offline_ai.ingestion.base import IngestItem


def load_csv_items(path: Path) -> list[IngestItem]:
    items: list[IngestItem] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            items.append(item_from_mapping(dict(row), default_source=path.name))
    return items
