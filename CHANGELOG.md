# Changelog

## 0.2.0

Persian-first relation graph, multi-hop ask, desktop UI, and deep GGUF wiring.

### Features

- Persian normalization, retrieval, and grounded answer composition
- Structured fact / claim extraction (meetings, travel, murder, missing-since)
- Persistent claim triples and graph edges for multi-hop questions
- Desktop Persian chat UI (`python -m offline_ai` / `examples/ask_ali.py`)
- Deep LLM path via llama.cpp (Qwen2.5 GGUF) with graph-grounded polish
- `rebuild_knowledge` / `offline-ai rebuild` to refresh relations on existing workspaces
- Setup helper: `scripts/setup_deep_model.py`

### Notes for operators

- Download GGUF weights once into `workspace/models/llm/` (see `models.yaml`).
- Runtime `workspace/` data and `*.gguf` files are not committed.

## 0.1.0

Initial delivery of the offline knowledge system.

### Features

- Persistent document memory (SQLite) with immutable original text
- Local vector index (FAISS) with application-level ID mapping
- Hybrid retrieval (FTS5 lexical + semantic + metadata/temporal filters)
- Grounded ask pipeline with citation validation
- Local entity / claim / topic / event extraction
- SQLite-backed knowledge graph with document provenance
- Pluggable LLM backends (llama.cpp GGUF, Transformers, extractive)
- Hashing embedding backend when neural models are not installed
- Python API (`LocalAI`), CLI (`offline-ai`), optional FastAPI app
- Backup / restore with checksum manifests
- Adapter versioning scaffolding (LoRA / experience replay)
- Automated test suite under `tests/`

### Notes for operators

- Without local GGUF / sentence-transformer weights, the system still runs
  using the hashing embedder and extractive answerer. Install models under
  paths listed in `models.yaml` for production quality.
- PDF / DOCX / HTML ingest is reserved for a later release.
- Full LoRA training requires a local base model and the `training` extra.
