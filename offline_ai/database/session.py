"""Database session and schema initialization."""

from __future__ import annotations

from pathlib import Path

from sqlalchemy import create_engine, event, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from offline_ai.database.models import Base, enable_sqlite_foreign_keys
from offline_ai.utils.logging import get_logger

logger = get_logger(__name__)


def make_engine(db_path: Path, *, echo: bool = False) -> Engine:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    url = f"sqlite:///{db_path.as_posix()}"
    engine = create_engine(
        url,
        echo=echo,
        future=True,
        connect_args={"check_same_thread": False, "timeout": 60},
    )
    enable_sqlite_foreign_keys(engine)

    @event.listens_for(engine, "connect")
    def _set_sqlite_pragma(dbapi_connection, _connection_record) -> None:  # type: ignore[no-untyped-def]
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA synchronous=NORMAL")
        cursor.execute("PRAGMA temp_store=MEMORY")
        cursor.close()

    return engine


def init_db(engine: Engine) -> None:
    Base.metadata.create_all(engine)
    with engine.begin() as conn:
        # Standalone FTS5 index (application IDs are authoritative)
        conn.execute(
            text(
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS documents_fts USING fts5(
                    document_id UNINDEXED,
                    original_text,
                    source,
                    author,
                    tokenize = "unicode61 remove_diacritics 2"
                );
                """
            )
        )
        conn.execute(
            text(
                """
                CREATE TRIGGER IF NOT EXISTS documents_ai AFTER INSERT ON documents BEGIN
                  INSERT INTO documents_fts(document_id, original_text, source, author)
                  VALUES (new.document_id, new.original_text, new.source, coalesce(new.author, ''));
                END;
                """
            )
        )
        conn.execute(
            text(
                """
                CREATE TRIGGER IF NOT EXISTS documents_ad AFTER DELETE ON documents BEGIN
                  DELETE FROM documents_fts WHERE document_id = old.document_id;
                END;
                """
            )
        )
        conn.execute(
            text(
                """
                CREATE TRIGGER IF NOT EXISTS documents_au AFTER UPDATE ON documents BEGIN
                  DELETE FROM documents_fts WHERE document_id = old.document_id;
                  INSERT INTO documents_fts(document_id, original_text, source, author)
                  VALUES (new.document_id, new.original_text, new.source, coalesce(new.author, ''));
                END;
                """
            )
        )
    logger.info("Database schema initialized", extra={"event": "db_init", "component": "database"})


class Database:
    """Thin wrapper around SQLAlchemy engine/session factory."""

    def __init__(self, db_path: Path, *, echo: bool = False) -> None:
        self.db_path = Path(db_path)
        self.engine = make_engine(self.db_path, echo=echo)
        init_db(self.engine)
        self._session_factory = sessionmaker(
            bind=self.engine, expire_on_commit=False, future=True
        )

    def session(self) -> Session:
        return self._session_factory()

    def dispose(self) -> None:
        self.engine.dispose()
