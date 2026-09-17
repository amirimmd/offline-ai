"""Simple add + ask usage."""

from __future__ import annotations

from pathlib import Path

from offline_ai import LocalAI


def test_add_and_ask_simple(tmp_path: Path) -> None:
    ai = LocalAI(tmp_path / "ws")

    added = ai.add("Company X VPN was compromised by malware.")
    assert added["documents_added"] == 1

    ai.learn(["Acme Corp was attacked.", "Weather is sunny."])
    assert ai.stats()["documents"] == 3

    answer = ai.ask("What happened to Company X VPN?", text_only=True)
    assert isinstance(answer, str)
    assert "DOC-" in answer or "Company X" in answer or "evidence" in answer.lower()

    full = ai.ask("Company X")
    assert isinstance(full, dict)
    assert "answer" in full
