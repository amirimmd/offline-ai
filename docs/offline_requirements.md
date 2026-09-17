# Offline requirements

Before disconnecting from the Internet, ensure the following are present on disk.

## Software

| Item | Required | Notes |
|------|----------|-------|
| Python 3.11+ (3.12 recommended) | Yes | Match major.minor across machines when possible |
| Virtualenv with installed deps | Yes | Or a local wheelhouse for `pip install --no-index` |
| `offline-ai` package (editable or installed) | Yes | |
| `llama-cpp-python` | Recommended | For GGUF inference; Transformers backend is alternative |
| CUDA toolkit / GPU drivers | Optional | CPU fallback works without CUDA |

## Models (local paths)

Configured in `models.yaml` / `workspace/models.yaml`:

| Role | Default | Local path pattern |
|------|---------|-------------------|
| Embedding | multilingual-e5-base | `workspace/models/embeddings/multilingual-e5-base` |
| LLM | Qwen2.5-3B-Instruct GGUF Q4_K_M | `workspace/models/llm/qwen2.5-3b-instruct-q4_k_m.gguf` |
| Reranker | ms-marco-MiniLM-L-6-v2 | `workspace/models/reranker/ms-marco-MiniLM-L-6-v2` |

Download **once** while online. Do not rely on Hub access at runtime.

## Data directories

| Path | Purpose |
|------|---------|
| `workspace/db/` | SQLite DB (created on first use) |
| `workspace/vectors/` | FAISS index + id map |
| `workspace/config/` | Copied policies |
| `workspace/adapters/` | LoRA versions (later phases) |

## Verification command

```bash
offline-ai --workspace ./workspace doctor --offline
```

Must return `"ok": true` with no network-related issues.

## Explicit non-requirements

- No API keys
- No cloud accounts
- No telemetry
- No automatic model downloads at query time
