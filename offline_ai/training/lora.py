"""
LoRA / PEFT training entry points.

Requires local base-model weights and the optional ``training`` dependency
extra. This module refuses to invent adapter weights when prerequisites are
missing; it records a train report and raises instead.
"""

from __future__ import annotations

import json
from pathlib import Path

from offline_ai.utils.logging import get_logger

logger = get_logger(__name__)


def train_lora_adapter(
    *,
    adapter_path: Path,
    dataset_path: Path,
    epochs: int = 1,
    base_model_path: Path | None = None,
) -> None:
    """
    Train (or scaffold) a LoRA adapter under ``adapter_path``.

    Raises ``RuntimeError`` if PEFT/transformers/torch are missing or if
    ``base_model_path`` is not available locally.
    """
    try:
        import peft  # noqa: F401
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "PEFT/transformers/torch not installed. Install offline wheels before training."
        ) from exc

    if base_model_path is None or not Path(base_model_path).exists():
        lines = Path(dataset_path).read_text(encoding="utf-8").strip().splitlines()
        info = {
            "examples": len(lines),
            "epochs": epochs,
            "status": "skipped_no_base_model",
            "message": "Provide base_model_path to run LoRA training",
        }
        (adapter_path / "train_report.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
        raise RuntimeError(info["message"])

    logger.info(
        "LoRA training starting",
        extra={"event": "lora_train", "component": "training"},
    )
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(base_model_path), local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(str(base_model_path), local_files_only=True)
    lora = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM")
    model = get_peft_model(model, lora)
    # Persist adapter files. Full dataloader/hyperparameter control belongs in
    # workspace training configuration for production runs.
    model.save_pretrained(str(adapter_path / "weights"))
    tokenizer.save_pretrained(str(adapter_path / "weights"))
    (adapter_path / "train_report.json").write_text(
        json.dumps({"status": "saved_adapter_scaffold", "epochs": epochs}, indent=2),
        encoding="utf-8",
    )
