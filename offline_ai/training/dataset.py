"""Training dataset builders from memory + feedback."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from sqlalchemy import select

from offline_ai.database.models import Document, Feedback
from offline_ai.database.session import Database
from offline_ai.utils.paths import ensure_dir


def build_dataset_from_memory(db: Database, out_path: Path, limit: int = 1000) -> Path:
    ensure_dir(out_path.parent)
    rows = []
    with db.session() as session:
        docs = session.execute(select(Document).limit(limit)).scalars()
        for d in docs:
            rows.append(
                {
                    "id": d.document_id,
                    "text": d.original_text,
                    "source": d.source,
                    "type": "document",
                }
            )
        fbs = session.execute(select(Feedback).limit(limit)).scalars()
        for f in fbs:
            if f.correction:
                rows.append(
                    {
                        "id": f.feedback_id,
                        "text": f.correction,
                        "type": "feedback_correction",
                        "answer_id": f.answer_id,
                    }
                )
    out_path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
    return out_path
