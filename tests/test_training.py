"""Training / replay / adapter tests — Phase 13."""

from __future__ import annotations

from pathlib import Path

from offline_ai.training.adapters import AdapterManager
from offline_ai.training.replay import ReplayBuffer, ReplayExample


def test_replay_strategies() -> None:
    buf = ReplayBuffer()
    for i in range(10):
        buf.add(ReplayExample(example_id=f"e{i}", text=f"t{i}", stratum=str(i % 3), importance=i))
    batch = buf.sample(5, strategy="importance", new_examples=[], replay_ratio=1.0)
    assert len(batch) == 5
    batch2 = buf.sample(
        6,
        strategy="stratified",
        new_examples=[ReplayExample(example_id="n1", text="new")],
        replay_ratio=0.5,
    )
    assert any(x.example_id == "n1" for x in batch2)


def test_adapter_versioning_and_rollback(tmp_path: Path) -> None:
    mgr = AdapterManager(tmp_path / "adapters")
    a1 = mgr.create_adapter(notes="first")
    a2 = mgr.create_adapter(notes="second")
    assert a1["name"] == "adapter_v001"
    assert a2["name"] == "adapter_v002"
    mgr.activate_adapter(a1["name"])
    mgr.activate_adapter(a2["name"])
    rolled = mgr.rollback_adapter()
    assert rolled["active"] == a1["name"]
