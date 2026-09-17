# Architecture

Offline AI is a local, offline-first knowledge system. User data is stored in
persistent memory (SQLite + FAISS). Language models are used only for reasoning
over retrieved evidence; they are never the source of truth.

## Design invariants

1. **Memory is independent of model weights.** Replacing the LLM, LoRA adapter,
   embedding model, or host machine must not lose ingested documents.
2. **Original text is immutable.** Each document keeps its exact `original_text`.
3. **Citations must resolve.** Every `[DOC-…]` / `[CLAIM-…]` in an answer must
   map to a stored record before the response is returned.
4. **Truth hierarchy:** Original evidence > structured memory > retrieved
   evidence > model reasoning.

## Package layout

```
offline_ai/
  core/           Application entry (LocalAI), config, hardware detection
  database/       SQLAlchemy models, repositories, Alembic migrations
  ingestion/      File and batch loaders (JSON, JSONL, CSV, TXT, tweets)
  memory/         Raw docs, embeddings, FAISS, graph, conversation promotion
  retrieval/      Lexical (FTS5), semantic, hybrid fusion, reranking, filters
  extraction/     Local entity / claim / topic / event extractors
  evidence/       Ask pipeline: retrieve -> generate -> cite -> validate
  llm/            Pluggable backends (llama.cpp, Transformers, extractive)
  training/       LoRA adapters, experience replay, evaluation helpers
  api/            Optional FastAPI app (localhost by default)
  cli/            `offline-ai` command-line interface
  utils/          Hashing, logging, paths, backup/restore
```

Runtime data lives under a **workspace directory** (default `./workspace`):

```
workspace/
  db/offline_ai.db
  vectors/faiss.index
  vectors/id_map.json
  config/policies.yaml
  models/                 # optional local model files
  adapters/               # LoRA versions (adapter_v001, …)
  training/
  logs/
  backups/
```

## Component map

| Concern | Primary modules | Notes |
|---------|-----------------|-------|
| Public API | `core.engine.LocalAI` | `add` / `ask` / `search` / ingest helpers |
| Config | `core.config`, `config.yaml`, `models.yaml`, `config/policies.yaml` | Policies enforced in code, not only prompts |
| Hardware | `core.hardware.HardwareDetector` | CUDA / CPU, VRAM, recommended context |
| Persistence | `database.*`, `memory.raw` | SQLite + FTS5 |
| Vectors | `memory.semantic`, `memory.embedding_backends` | FAISS; app IDs are authoritative |
| Retrieval | `retrieval.hybrid` | Weighted fusion of lexical + semantic + filters |
| Grounded QA | `evidence.engine` | Citation validation before return |
| Models | `llm.factory` | Loads GGUF/HF if present; otherwise local extractive backend |
| CLI | `cli.main` | Typer entry point `offline-ai` |
| HTTP | `api.app.create_app` | Optional; binds `127.0.0.1` by default |

## Ask pipeline

```
query
  -> hybrid retrieval (FTS5 + FAISS + metadata/temporal filters)
  -> rerank
  -> evidence selection
  -> LLM generation (evidence-only prompt)
  -> citation validation + grounding checks
  -> structured answer  OR  "Insufficient evidence in stored memory."
```

## Identifier scheme

| Kind | Format | Example |
|------|--------|---------|
| Document | `DOC-{n:09d}` | `DOC-000000001` |
| Chunk | `CHK-{n:09d}` | `CHK-000000001` |
| Claim | `CLAIM-{n:09d}` | `CLAIM-000000001` |
| Entity | `ENTITY-{n:09d}` | `ENTITY-000000001` |
| Answer | `ANS-{n:09d}` | `ANS-000000001` |

FAISS internal row indices are not permanent keys. Application IDs in
`vector_mappings` / `id_map.json` are authoritative.

## Extending the system

Suggested extension points (keep interfaces stable):

- **New ingest format:** add a loader under `ingestion/` and branch in
  `IngestionPipeline.ingest_file`.
- **New embedding model:** implement `EmbeddingBackend` and register via
  `create_embedding_backend`.
- **New LLM backend:** implement `LLMBackend` and wire in `llm.factory`.
- **New vector store:** implement `VectorStore` (see `FaissVectorStore`).
- **Richer NER:** replace rule extractors in `extraction/` while preserving
  provenance fields (`source_document_id`, offsets, confidence).

## Related docs

- [docs/MEMORY.md](docs/MEMORY.md) - memory layers
- [docs/MODELS.md](docs/MODELS.md) - model registry
- [docs/API.md](docs/API.md) - Python API
- [docs/CLI.md](docs/CLI.md) - CLI reference
- [docs/TRAINING.md](docs/TRAINING.md) - adapters / LoRA
- [docs/SECURITY.md](docs/SECURITY.md) - offline / security posture
- [INSTALL.md](INSTALL.md) / [OFFLINE_INSTALL.md](OFFLINE_INSTALL.md)
