"""Backup and restore of portable memory workspace."""

from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from offline_ai.utils.logging import get_logger
from offline_ai.utils.paths import ensure_dir, safe_join

logger = get_logger(__name__)

BACKUP_ITEMS = [
    "db",
    "vectors",
    "config",
    "config.yaml",
    "models.yaml",
    "adapters",
]


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _walk_checksums(root: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not root.exists():
        return out
    for p in root.rglob("*"):
        if p.is_file():
            rel = p.relative_to(root).as_posix()
            out[rel] = _file_sha256(p)
    return out


class BackupManager:
    def __init__(self, workspace: Path) -> None:
        self.workspace = Path(workspace)

    def backup(self, dest: Path) -> dict[str, Any]:
        dest = Path(dest)
        ensure_dir(dest)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        target = dest / f"offline_ai_backup_{stamp}"
        ensure_dir(target)

        copied = []
        for name in BACKUP_ITEMS:
            src = self.workspace / name
            if not src.exists():
                continue
            dst = target / name
            if src.is_dir():
                shutil.copytree(src, dst, dirs_exist_ok=True)
            else:
                ensure_dir(dst.parent)
                shutil.copy2(src, dst)
            copied.append(name)

        # Do not duplicate base model blobs by default; copy model config only.
        manifest = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "workspace": str(self.workspace),
            "copied": copied,
            "checksums": _walk_checksums(target),
            "includes_base_models": False,
        }
        (target / "manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
        logger.info("Backup complete", extra={"event": "backup", "component": "backup"})
        return {"backup_path": str(target), "manifest": manifest}

    def restore(self, backup_path: Path, *, verify: bool = True) -> dict[str, Any]:
        backup_path = Path(backup_path)
        manifest_path = backup_path / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError("manifest.json missing in backup")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if verify:
            current = _walk_checksums(backup_path)
            # exclude manifest itself from expected if stored inside checksums
            expected = {k: v for k, v in (manifest.get("checksums") or {}).items() if k != "manifest.json"}
            for k, v in expected.items():
                if current.get(k) != v:
                    raise ValueError(f"Checksum mismatch for {k}")

        for name in manifest.get("copied") or []:
            src = backup_path / name
            dst = self.workspace / name
            if not src.exists():
                continue
            if src.is_dir():
                if dst.exists():
                    shutil.rmtree(dst)
                shutil.copytree(src, dst)
            else:
                ensure_dir(dst.parent)
                shutil.copy2(src, dst)
        logger.info("Restore complete", extra={"event": "restore", "component": "backup"})
        return {"restored_to": str(self.workspace), "ok": True}
