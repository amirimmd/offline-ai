"""API tests — Phase 15."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from offline_ai.api.app import create_app


def test_api_ingest_ask(tmp_path: Path) -> None:
    app = create_app(tmp_path / "ws")
    client = TestClient(app)
    r = client.post(
        "/ingest",
        json={"text": "Company X VPN was compromised.", "source": "api"},
    )
    assert r.status_code == 200
    assert r.json()["documents_added"] == 1
    s = client.post("/search", json={"query": "Company X"})
    assert s.status_code == 200
    a = client.post("/ask", json={"query": "What happened to Company X VPN?"})
    assert a.status_code == 200
    body = a.json()
    assert "answer" in body
    assert client.get("/memory/stats").status_code == 200
