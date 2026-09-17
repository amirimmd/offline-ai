"""Backup/restore tests — Phase 14."""

from __future__ import annotations

from pathlib import Path

from offline_ai import LocalAI
from offline_ai.utils.backup import BackupManager


def test_backup_restore(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    ai = LocalAI(ws)
    stats = ai.ingest_text("Persistent secret alpha report about Company X.", source="r")
    doc_id = stats["document_ids"][0]
    backup_root = tmp_path / "backups"
    result = BackupManager(ws).backup(backup_root)
    backup_path = Path(result["backup_path"])

    ws2 = tmp_path / "ws2"
    ws2.mkdir()
    # minimal restore target layout
    BackupManager(ws2).restore(backup_path)
    ai2 = LocalAI(ws2)
    doc = ai2.get_document(doc_id)
    assert doc is not None
    assert "Company X" in doc["original_text"]
