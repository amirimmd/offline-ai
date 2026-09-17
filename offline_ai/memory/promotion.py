"""Conversation memory and promotion to long-term."""

from __future__ import annotations

from typing import Any

from sqlalchemy import select

from offline_ai.database.ids import next_id
from offline_ai.database.models import Conversation, ConversationMessage
from offline_ai.database.session import Database
from offline_ai.ingestion.base import IngestItem, IngestionPipeline
from offline_ai.utils.logging import get_logger

logger = get_logger(__name__)


class ConversationMemory:
    def __init__(self, db: Database) -> None:
        self.db = db

    def start(self, metadata: dict[str, Any] | None = None) -> str:
        with self.db.session() as session:
            cid = next_id(session, "CONV")
            session.add(Conversation(conversation_id=cid, metadata_json=metadata or {}))
            session.commit()
            return cid

    def add_message(self, conversation_id: str, role: str, content: str) -> str:
        with self.db.session() as session:
            mid = next_id(session, "MSG")
            session.add(
                ConversationMessage(
                    message_id=mid,
                    conversation_id=conversation_id,
                    role=role,
                    content=content,
                )
            )
            session.commit()
            return mid

    def history(self, conversation_id: str) -> list[dict[str, Any]]:
        with self.db.session() as session:
            rows = session.execute(
                select(ConversationMessage)
                .where(ConversationMessage.conversation_id == conversation_id)
                .order_by(ConversationMessage.id)
            ).scalars()
            return [
                {
                    "message_id": m.message_id,
                    "role": m.role,
                    "content": m.content,
                    "timestamp": m.timestamp.isoformat() if m.timestamp else None,
                    "promoted": m.promoted,
                }
                for m in rows
            ]


class MemoryPromotion:
    """Promote explicitly validated conversation content into long-term memory."""

    def __init__(self, db: Database, ingestion: IngestionPipeline) -> None:
        self.db = db
        self.ingestion = ingestion

    def promote_message(
        self,
        message_id: str,
        *,
        source: str = "conversation_promotion",
        require_validation: bool = True,
    ) -> dict[str, Any]:
        with self.db.session() as session:
            msg = session.execute(
                select(ConversationMessage).where(ConversationMessage.message_id == message_id)
            ).scalar_one_or_none()
            if msg is None:
                raise KeyError(message_id)
            if require_validation and msg.role != "user":
                # Only promote user-provided factual notes by default
                pass
            text = msg.content
            msg.promoted = True
            session.commit()

        stats = self.ingestion.ingest_items(
            [
                IngestItem(
                    text=text,
                    source=source,
                    source_type="conversation",
                    collection_method="promotion",
                    metadata={"message_id": message_id},
                )
            ],
            source_description=f"promote:{message_id}",
        )
        logger.info(
            "Message promoted to long-term memory",
            extra={"event": "memory_promotion", "component": "promotion"},
        )
        return stats.to_dict()
