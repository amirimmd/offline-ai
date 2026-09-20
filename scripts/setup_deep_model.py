"""Download and activate the deep offline LLM (Qwen2.5 GGUF)."""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

from offline_ai.utils.paths import project_root


# Prefer 7B for depth on RTX 4070 8GB; 3B as compact fallback.
MODELS = {
    "7b": {
        "repo": "bartowski/Qwen2.5-7B-Instruct-GGUF",
        "filename": "Qwen2.5-7B-Instruct-Q4_K_M.gguf",
        "dest_name": "qwen2.5-7b-instruct-q4_k_m.gguf",
        "models_yaml_key": "qwen25-7b-instruct-gguf",
    },
    "3b": {
        "repo": "bartowski/Qwen2.5-3B-Instruct-GGUF",
        "filename": "Qwen2.5-3B-Instruct-Q4_K_M.gguf",
        "dest_name": "qwen2.5-3b-instruct-q4_k_m.gguf",
        "models_yaml_key": "qwen25-3b-instruct-gguf",
    },
}


def _llm_dir(workspace: Path) -> Path:
    d = workspace / "models" / "llm"
    d.mkdir(parents=True, exist_ok=True)
    return d


def download_model(workspace: Path, size: str = "7b") -> Path:
    from huggingface_hub import hf_hub_download

    # Prefer China/region-friendly mirror when HF is slow.
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

    cfg = MODELS[size]
    dest_dir = _llm_dir(workspace)
    dest = dest_dir / cfg["dest_name"]
    if dest.exists() and dest.stat().st_size > 1_000_000_000:
        print(f"already present: {dest}")
        return dest

    print(f"downloading {cfg['repo']} / {cfg['filename']} …")
    path = hf_hub_download(
        repo_id=cfg["repo"],
        filename=cfg["filename"],
        local_dir=str(dest_dir),
    )
    downloaded = Path(path)
    if downloaded.resolve() != dest.resolve():
        if dest.exists():
            dest.unlink()
        shutil.move(str(downloaded), str(dest))
    print(f"saved: {dest} ({dest.stat().st_size / 1e9:.2f} GB)")
    return dest


def set_default_llm(workspace: Path, size: str) -> None:
    import yaml

    key = MODELS[size]["models_yaml_key"]
    for path in (workspace / "models.yaml", project_root() / "models.yaml"):
        if not path.exists():
            continue
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        data.setdefault("defaults", {})["llm"] = key
        # GPU profile prefers the selected deep model
        profiles = data.setdefault("hardware_profiles", {})
        for pname in ("gpu_8gb", "cpu"):
            profiles.setdefault(pname, {})["llm"] = key
        path.write_text(
            yaml.safe_dump(data, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
        print(f"updated defaults.llm={key} in {path}")


def smoke_test(workspace: Path, size: str) -> None:
    from offline_ai.llm.llama_cpp import LlamaCppBackend

    dest = _llm_dir(workspace) / MODELS[size]["dest_name"]
    backend = LlamaCppBackend(dest, n_ctx=2048, n_gpu_layers=0, n_threads=4, chat_format="chatml")
    backend.load()
    from offline_ai.llm.base import LLMMessage

    out = backend.chat(
        [
            LLMMessage(role="system", content="جواب کوتاه و دقیق به فارسی بده."),
            LLMMessage(role="user", content="یک جمله بگو: آماده‌ای؟"),
        ],
        max_tokens=64,
        temperature=0.1,
    )
    print("smoke:", (out.text or "").strip()[:200])
    backend.unload()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Setup deep Qwen GGUF for offline-ai")
    parser.add_argument("-w", "--workspace", default="./workspace")
    parser.add_argument("--size", choices=["7b", "3b"], default="7b")
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--skip-test", action="store_true")
    args = parser.parse_args(argv)
    workspace = Path(args.workspace).resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    if not args.skip_download:
        download_model(workspace, args.size)
    set_default_llm(workspace, args.size)
    if not args.skip_test:
        smoke_test(workspace, args.size)
    print("deep model ready")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
