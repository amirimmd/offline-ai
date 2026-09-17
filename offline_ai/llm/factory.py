"""LLM factory from models.yaml + hardware recommendation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from offline_ai.llm.base import LLMBackend
from offline_ai.llm.extractive import ExtractiveLLMBackend
from offline_ai.utils.logging import get_logger
from offline_ai.utils.paths import resolve_path

logger = get_logger(__name__)


def create_llm_backend(
    models_raw: dict[str, Any],
    *,
    workspace: Path,
    device: str = "cpu",
    n_ctx: int = 4096,
    n_gpu_layers: int = 0,
    n_threads: int | None = None,
    prefer_key: str | None = None,
) -> LLMBackend:
    models = models_raw.get("models") or {}
    defaults = models_raw.get("defaults") or {}
    key = prefer_key or defaults.get("llm")
    if key and key in models:
        cfg = models[key]
        local = resolve_path(workspace, cfg.get("local_path", ""))
        backend = cfg.get("backend")
        if local.exists():
            if backend == "llama_cpp":
                from offline_ai.llm.llama_cpp import LlamaCppBackend

                return LlamaCppBackend(
                    local,
                    n_ctx=n_ctx,
                    n_gpu_layers=n_gpu_layers if device == "cuda" else 0,
                    n_threads=n_threads,
                )
            if backend == "transformers":
                from offline_ai.llm.transformers_backend import TransformersBackend

                return TransformersBackend(
                    local,
                    device=device,
                    load_in_4bit=bool(cfg.get("load_in_4bit")),
                    max_context=n_ctx,
                )
        else:
            logger.info(
                "Configured LLM path missing; using extractive grounded backend",
                extra={"event": "llm_fallback", "component": "llm"},
            )
    return ExtractiveLLMBackend()
