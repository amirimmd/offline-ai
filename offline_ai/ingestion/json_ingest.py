"""JSON / JSONL ingestion."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from dateutil import parser as date_parser

from offline_ai.ingestion.base import IngestItem


def _parse_ts(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    try:
        return date_parser.parse(str(value))
    except (ValueError, TypeError, OverflowError):
        return None


def item_from_mapping(obj: dict[str, Any], *, default_source: str = "json") -> IngestItem:
    text = obj.get("text") or obj.get("content") or obj.get("body") or obj.get("full_text")
    if not text:
        raise ValueError("JSON object missing text/content/body/full_text")
    return IngestItem(
        text=str(text),
        source=str(obj.get("source") or default_source),
        source_url=obj.get("source_url") or obj.get("url"),
        author=obj.get("author") or obj.get("user") or obj.get("username"),
        timestamp=_parse_ts(obj.get("timestamp") or obj.get("created_at") or obj.get("date")),
        language=obj.get("language") or obj.get("lang"),
        metadata={k: v for k, v in obj.items() if k not in {
            "text", "content", "body", "full_text", "source", "source_url", "url",
            "author", "user", "username", "timestamp", "created_at", "date", "language", "lang",
        }},
        source_type=obj.get("source_type"),
        source_name=obj.get("source_name"),
        collection_method=obj.get("collection_method", "file"),
    )


def load_json_items(path: Path) -> list[IngestItem]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        if "documents" in data and isinstance(data["documents"], list):
            data = data["documents"]
        elif "tweets" in data and isinstance(data["tweets"], list):
            data = data["tweets"]
        else:
            data = [data]
    if not isinstance(data, list):
        raise ValueError("JSON root must be a list or object with documents/tweets")
    items: list[IngestItem] = []
    for i, obj in enumerate(data):
        if isinstance(obj, str):
            items.append(IngestItem(text=obj, source=path.name, collection_method="file"))
        elif isinstance(obj, dict):
            items.append(item_from_mapping(obj, default_source=path.name))
        else:
            raise ValueError(f"Invalid item at index {i}")
    return items


def load_jsonl_items(path: Path) -> list[IngestItem]:
    items: list[IngestItem] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if isinstance(obj, str):
                items.append(IngestItem(text=obj, source=path.name, collection_method="file"))
            elif isinstance(obj, dict):
                items.append(item_from_mapping(obj, default_source=path.name))
            else:
                raise ValueError(f"Invalid JSONL at line {line_no}")
    return items
