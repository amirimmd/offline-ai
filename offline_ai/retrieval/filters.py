"""Metadata / temporal / entity filters for retrieval."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from dateutil import parser as date_parser
from sqlalchemy import select
from sqlalchemy.orm import Session

from offline_ai.database.models import Document, Entity, EntityMention


def parse_relative_date(text: str, *, now: datetime | None = None) -> tuple[datetime | None, datetime | None]:
    """
    Parse simple relative/absolute constraints from query-ish text.
    Returns (start, end) inclusive-ish window, or (None, None).
    """
    now = now or datetime.now(timezone.utc)
    lower = text.lower()
    if "last 30 days" in lower or "past 30 days" in lower:
        return now - timedelta(days=30), now
    if "last 7 days" in lower or "past week" in lower:
        return now - timedelta(days=7), now
    if "last 24 hours" in lower or "past day" in lower:
        return now - timedelta(days=1), now
    return None, None


class MetadataFilter:
    def __init__(self, session: Session) -> None:
        self.session = session

    def filter_document_ids(
        self,
        candidate_ids: list[str] | None = None,
        *,
        source: str | None = None,
        author: str | None = None,
        after: datetime | None = None,
        before: datetime | None = None,
        entity_normalized: str | None = None,
        entity_type: str | None = None,
    ) -> list[str]:
        q = select(Document.document_id)
        if candidate_ids is not None:
            q = q.where(Document.document_id.in_(candidate_ids))
        if source:
            q = q.where(Document.source == source)
        if author:
            q = q.where(Document.author == author)
        if after is not None:
            q = q.where(Document.timestamp >= after)
        if before is not None:
            q = q.where(Document.timestamp <= before)
        if entity_normalized:
            q = (
                q.join(EntityMention, EntityMention.document_id == Document.document_id)
                .join(Entity, Entity.entity_id == EntityMention.entity_id)
                .where(Entity.normalized_value == entity_normalized)
            )
            if entity_type:
                q = q.where(Entity.entity_type == entity_type)
        return list(self.session.execute(q).scalars().unique())


def coerce_datetime(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    try:
        return date_parser.parse(str(value))
    except (ValueError, TypeError, OverflowError):
        return None
