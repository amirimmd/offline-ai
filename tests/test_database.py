"""Database tests — Phase 2."""

from __future__ import annotations

from pathlib import Path

from offline_ai.database.ids import next_id
from offline_ai.database.repositories import DocumentRepository
from offline_ai.database.session import Database


def test_init_db(tmp_path: Path) -> None:
    db = Database(tmp_path / "t.db")
    assert (tmp_path / "t.db").exists()
    with db.session() as session:
        assert next_id(session, "DOC") == "DOC-000000001"
        assert next_id(session, "DOC") == "DOC-000000002"
        session.commit()
    db.dispose()


def test_document_immutable_and_duplicate(tmp_path: Path) -> None:
    db = Database(tmp_path / "t.db")
    with db.session() as session:
        repo = DocumentRepository(session)
        d1, c1 = repo.create(text="hello world", source="t", source_url="http://x")
        session.commit()
        assert c1 is True
        assert d1.document_id == "DOC-000000001"
        original = d1.original_text
        d2, c2 = repo.create(text="hello world", source="t", source_url="http://x")
        session.commit()
        assert c2 is False
        assert d2.document_id == d1.document_id
        assert d2.original_text == original
    db.dispose()


def test_fts_trigger(tmp_path: Path) -> None:
    db = Database(tmp_path / "t.db")
    with db.session() as session:
        DocumentRepository(session).create(text="alpha beta gamma", source="s")
        session.commit()
        rows = session.execute(
            __import__("sqlalchemy").text(
                "SELECT document_id FROM documents_fts WHERE documents_fts MATCH 'alpha'"
            )
        ).fetchall()
        assert len(rows) == 1
    db.dispose()
