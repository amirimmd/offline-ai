# Contributor guide

This repository is intended for continued maintenance by application and ML
engineers. Keep changes small, tested, and aligned with the memory-first design.

## Principles

1. Do not store user facts only in model weights.
2. Never invent citation IDs; validate against the database.
3. Prefer interfaces (`LLMBackend`, `EmbeddingBackend`, `VectorStore`, `Reranker`)
   over hard-coding a single vendor stack.
4. Keep modules focused; avoid multi-thousand-line files.

## Local development

```bash
pip install -e ".[dev]"
pytest
ruff check offline_ai tests
```

## Where to change what

| Task | Start here |
|------|------------|
| Public Python API | `offline_ai/core/engine.py` |
| Ask / grounding behaviour | `offline_ai/evidence/` |
| Ingest formats | `offline_ai/ingestion/` |
| Schema / migrations | `offline_ai/database/` |
| Retrieval scoring | `offline_ai/retrieval/hybrid.py`, `config.yaml` |
| Model selection | `models.yaml`, `offline_ai/llm/factory.py` |
| Policies | `config/policies.yaml` |
| CLI | `offline_ai/cli/main.py` |
| HTTP API | `offline_ai/api/app.py` |

## Pull requests

- Include or update tests for behaviour changes.
- Update `CHANGELOG.md` for user-visible changes.
- Do not commit `workspace/` runtime data, model weights, or `.venv/`.
