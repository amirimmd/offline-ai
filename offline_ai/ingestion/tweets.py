"""Tweet / social post ingestion helpers."""

from __future__ import annotations

from typing import Any

from offline_ai.ingestion.base import IngestItem
from offline_ai.ingestion.json_ingest import item_from_mapping


def tweets_to_items(tweets: list[dict[str, Any]] | list[str]) -> list[IngestItem]:
    items: list[IngestItem] = []
    for t in tweets:
        if isinstance(t, str):
            items.append(
                IngestItem(
                    text=t,
                    source="tweet",
                    source_type="social",
                    collection_method="api_or_manual",
                )
            )
        elif isinstance(t, dict):
            item = item_from_mapping(t, default_source="tweet")
            item.source_type = item.source_type or "social"
            item.collection_method = item.collection_method or "api_or_manual"
            items.append(item)
        else:
            raise TypeError(f"Unsupported tweet type: {type(t)}")
    return items
