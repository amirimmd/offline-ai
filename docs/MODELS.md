# Models

Configured in `models.yaml` (copied into each workspace).

## Defaults

| Role | Key | Notes |
|------|-----|-------|
| LLM | `qwen25-3b-instruct-gguf` | Multilingual FA/EN; Q4_K_M fits ~8GB VRAM |
| Embedding | `multilingual-e5-base` | Multilingual; download once |
| Reranker | `msmarco-minilm-l6` | Optional; lexical fallback if missing |

## Offline rule

Set `local_path` to files already on disk. Runtime must not download.

If paths are missing:

- Embeddings → `HashingEmbeddingBackend`
- LLM → `ExtractiveLLMBackend` (evidence-only, citation-safe)

## Hardware profiles

`HardwareDetector` recommends CUDA vs CPU, context length, and GPU layers.
