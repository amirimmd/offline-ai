"""Path helpers using pathlib. Rejects directory traversal outside a base root."""

from __future__ import annotations

from pathlib import Path


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_path(base: Path, maybe_relative: str | Path) -> Path:
    p = Path(maybe_relative)
    if p.is_absolute():
        return p
    return (base / p).resolve()


def safe_join(base: Path, *parts: str) -> Path:
    """Join paths and reject traversal outside base."""
    base_resolved = base.resolve()
    candidate = base_resolved.joinpath(*parts).resolve()
    if not str(candidate).startswith(str(base_resolved)):
        raise ValueError(f"Path traversal rejected: {parts}")
    return candidate


def project_root() -> Path:
    """Repository root (parent of the offline_ai package)."""
    return Path(__file__).resolve().parents[2]
