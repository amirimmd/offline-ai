# Changelog

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
