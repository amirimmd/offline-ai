"""Application-level ID allocation (DOC-, CLAIM-, …)."""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from offline_ai.database.models import IdSequence

PREFIX_WIDTH = {
    "DOC": 9,
    "CHK": 9,
    "CLAIM": 9,
    "ENTITY": 9,
    "EVENT": 9,
    "ANS": 9,
    "RUN": 9,
    "CIT": 9,
    "QRY": 9,
    "FB": 9,
    "CONV": 9,
    "MSG": 9,
    "REL": 9,
    "TOPIC": 9,
    "SRC": 9,
    "AUTH": 9,
}


def next_id(session: Session, prefix: str) -> str:
    prefix = prefix.upper()
    width = PREFIX_WIDTH.get(prefix, 9)
    row = session.execute(
        select(IdSequence).where(IdSequence.prefix == prefix)
    ).scalar_one_or_none()
    if row is None:
        row = IdSequence(prefix=prefix, next_value=1)
        session.add(row)
        session.flush()
    value = row.next_value
    row.next_value = value + 1
    session.flush()
    return f"{prefix}-{value:0{width}d}"
