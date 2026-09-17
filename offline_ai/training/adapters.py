"""LoRA / PEFT adapter management with versioning and rollback."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from offline_ai.utils.logging import get_logger
from offline_ai.utils.paths import ensure_dir

logger = get_logger(__name__)


class AdapterManager:
    """
    Manages adapter versions under workspace/adapters/adapter_vNNN.
    Never overwrites previous adapters; activate/rollback by pointer.
    """

    def __init__(self, adapters_root: Path, base_model_key: str = "default") -> None:
        self.root = ensure_dir(Path(adapters_root))
        self.base_model_key = base_model_key
        self.state_path = self.root / "active.json"

    def list_adapters(self) -> list[dict[str, Any]]:
        items = []
        for p in sorted(self.root.glob("adapter_v*")):
            if p.is_dir():
                meta = {}
                mp = p / "meta.json"
                if mp.exists():
                    meta = json.loads(mp.read_text(encoding="utf-8"))
                items.append({"name": p.name, "path": str(p), **meta})
        return items

    def _next_version_name(self) -> str:
        existing = [p.name for p in self.root.glob("adapter_v*") if p.is_dir()]
        nums = []
        for name in existing:
            try:
                nums.append(int(name.split("_v")[-1]))
            except ValueError:
                pass
        n = (max(nums) + 1) if nums else 1
        return f"adapter_v{n:03d}"

    def create_adapter(self, *, notes: str | None = None) -> dict[str, Any]:
        name = self._next_version_name()
        path = ensure_dir(self.root / name)
        meta = {
            "name": name,
            "base_model_key": self.base_model_key,
            "status": "created",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "notes": notes,
        }
        (path / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return meta

    def train_adapter(
        self,
        adapter_name: str | None = None,
        *,
        dataset_path: Path | None = None,
        epochs: int = 1,
    ) -> dict[str, Any]:
        """
        Real training entrypoint. Requires peft+transformers+local base model.
        If unavailable, records a training stub state without inventing weights.
        """
        if adapter_name is None:
            meta = self.create_adapter(notes="auto-created for training")
            adapter_name = meta["name"]
        path = self.root / adapter_name
        if not path.exists():
            raise FileNotFoundError(adapter_name)

        meta_path = path / "meta.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta["status"] = "training"
        meta["dataset_path"] = str(dataset_path) if dataset_path else None
        meta["epochs"] = epochs
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

        try:
            # Attempt real PEFT training only if dataset and deps exist
            if dataset_path and Path(dataset_path).exists():
                self._run_peft_training(path, dataset_path, epochs=epochs)
                meta["status"] = "trained"
            else:
                meta["status"] = "awaiting_dataset"
                meta["warning"] = "No dataset provided; adapter shell created without weights"
        except Exception as exc:  # noqa: BLE001
            meta["status"] = "failed"
            meta["error"] = str(exc)
            logger.error(
                "Adapter training failed",
                extra={"event": "adapter_train_fail", "error": str(exc), "component": "training"},
            )
        meta["finished_at"] = datetime.now(timezone.utc).isoformat()
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return meta

    def _run_peft_training(self, adapter_path: Path, dataset_path: Path, *, epochs: int) -> None:
        # Intentionally conservative: verify imports; full training configs live in training/lora.py
        from offline_ai.training.lora import train_lora_adapter

        train_lora_adapter(adapter_path=adapter_path, dataset_path=dataset_path, epochs=epochs)

    def evaluate_adapter(self, adapter_name: str) -> dict[str, Any]:
        path = self.root / adapter_name
        meta = json.loads((path / "meta.json").read_text(encoding="utf-8"))
        metrics = {"status": meta.get("status"), "placeholder_score": None}
        meta["metrics"] = metrics
        (path / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return metrics

    def activate_adapter(self, adapter_name: str) -> dict[str, Any]:
        path = self.root / adapter_name
        if not path.exists():
            raise FileNotFoundError(adapter_name)
        state = {
            "active": adapter_name,
            "activated_at": datetime.now(timezone.utc).isoformat(),
            "previous": None,
        }
        if self.state_path.exists():
            prev = json.loads(self.state_path.read_text(encoding="utf-8"))
            state["previous"] = prev.get("active")
        self.state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
        return state

    def rollback_adapter(self) -> dict[str, Any]:
        if not self.state_path.exists():
            raise RuntimeError("No active adapter to rollback")
        state = json.loads(self.state_path.read_text(encoding="utf-8"))
        prev = state.get("previous")
        if not prev:
            raise RuntimeError("No previous adapter recorded")
        return self.activate_adapter(prev)

    def load_adapter(self, adapter_name: str) -> Path:
        path = self.root / adapter_name
        if not path.exists():
            raise FileNotFoundError(adapter_name)
        return path

    def unload_adapter(self) -> None:
        # Activation pointer remains; runtime unload is LLM-backend specific
        logger.info("Adapter unload requested", extra={"event": "adapter_unload", "component": "training"})
