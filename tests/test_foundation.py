"""Foundation tests — Phase 1."""

from __future__ import annotations

from pathlib import Path

from offline_ai import LocalAI, __version__
from offline_ai.core.hardware import HardwareDetector
from offline_ai.utils.hashing import content_hash, normalize_text, normalized_content_hash
from offline_ai.utils.paths import project_root, safe_join


def test_version() -> None:
    assert __version__


def test_project_root_exists() -> None:
    root = project_root()
    assert (root / "pyproject.toml").exists()
    assert (root / "ARCHITECTURE.md").exists()


def test_local_ai_init(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    ai = LocalAI(ws)
    assert ai.workspace.exists()
    assert (ai.workspace / "config.yaml").exists()
    assert (ai.workspace / "models.yaml").exists()
    assert (ai.workspace / "config" / "policies.yaml").exists()
    assert ai.settings.db_path.parent.exists()
    stats = ai.stats()
    assert stats["workspace"] == str(ai.workspace)


def test_doctor(tmp_path: Path) -> None:
    ai = LocalAI(tmp_path / "ws")
    report = ai.doctor(offline=True)
    assert report["ok"] is True
    assert "hardware" in report


def test_hardware_detector() -> None:
    det = HardwareDetector()
    info, rec = det.recommend()
    assert info.cpu_count >= 1
    assert rec.device in {"cpu", "cuda"}
    assert rec.context_length > 0


def test_hashing() -> None:
    assert content_hash("hello") == content_hash("hello")
    assert content_hash("hello") != content_hash("world")
    assert normalize_text("  Hello\nWorld  ") == "hello world"
    assert normalized_content_hash("Hello") == normalized_content_hash("hello")


def test_safe_join(tmp_path: Path) -> None:
    base = tmp_path / "base"
    base.mkdir()
    p = safe_join(base, "a", "b.txt")
    assert str(p).startswith(str(base.resolve()))
    try:
        safe_join(base, "..", "..", "etc", "passwd")
        raised = False
    except ValueError:
        raised = True
    assert raised
