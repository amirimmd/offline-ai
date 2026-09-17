"""Content hashing utilities."""

from __future__ import annotations

import hashlib
import re
import unicodedata


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def normalize_text(text: str) -> str:
    """Normalize for near-duplicate / normalized-hash comparison."""
    text = unicodedata.normalize("NFKC", text)
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def content_hash(text: str) -> str:
    return sha256_text(text)


def normalized_content_hash(text: str) -> str:
    return sha256_text(normalize_text(text))


def source_hash(source: str, source_url: str | None, text: str) -> str:
    payload = f"{source}|{source_url or ''}|{text}"
    return sha256_text(payload)
