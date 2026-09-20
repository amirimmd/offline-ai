# Architecture

**Offline AI** (`offline-ai` v0.2.0) is a local, offline-first knowledge system.
User data lives in persistent memory (SQLite + FAISS + claim/relation graph).
Language models reason over retrieved evidence and stored relations only; they
are never the source of truth.

This document is the technical map for senior engineers reviewing or extending
the codebase. Product-facing setup lives in [README.md](README.md).

---

## Design invariants

1. **Memory is independent of model weights.** Replacing the LLM, LoRA adapter,
   embedding model, or host machine must not lose ingested documents.
2. **Original text is immutable.** Each document keeps its exact `original_text`.
   Extraction and indexing derive from it; they never rewrite it.
3. **Citations must resolve.** Every `[DOC-…]` / `[CLAIM-…]` in an answer must
   map to a stored record before the response is returned (or the answer is
   rejected / replaced with the insufficient-evidence message).
4. **Truth hierarchy:** Original evidence → structured memory (claims/graph) →
   retrieved evidence → model reasoning / polish.
5. **Offline by default.** Runtime does not require network. Model weights are
   optional local files under the workspace; without them the extractive path
   still answers from the graph and retrieved text.
6. **Persian-first correctness.** Normalization, intent detection, relation
   predicates, and UI handle Persian (and Arabic-form variants) explicitly;
   English paths remain supported.

---

## System overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│  Surfaces                                                                │
│  LocalAI (Python) │ offline-ai CLI │ persian_ui (tk) │ optional FastAPI │
└───────────────────────────────┬─────────────────────────────────────────┘
                                │
                    ┌───────────▼───────────┐
                    │   core.engine.LocalAI │  facade / wiring
                    └───────────┬───────────┘
         ┌──────────────────────┼──────────────────────┐
         ▼                      ▼                      ▼
┌─────────────────┐   ┌──────────────────┐   ┌────────────────────┐
│ IngestionPipeline│   │ EvidenceEngine   │   │ ExtractionPipeline │
│  + RawMemory     │──▶│  ask / cite /    │◀──│ entities, claims,  │
│  + embed index   │   │  ground          │   │ topics, events,    │
└────────┬────────┘   └────────┬─────────┘   │ graph edges        │
         │                     │             └─────────┬──────────┘
         ▼                     ▼                       ▼
┌─────────────────┐   ┌──────────────────┐   ┌────────────────────┐
│ SQLite documents│   │ RelationStore    │   │ claims / entities /│
│ + FTS5 + chunks │   │ multi-hop graph  │   │ relationships      │
└────────┬────────┘   └──────────────────┘   └────────────────────┘
         │
         ▼
┌─────────────────┐   ┌──────────────────┐
│ FAISS + id_map  │   │ LLMBackend       │
│ (optional E5)   │   │ llama.cpp / HF / │
└─────────────────┘   │ extractive       │
                      └──────────────────┘
```

**Facade:** application code should use `offline_ai.LocalAI`. Lower packages are
extension and test surfaces.

---

## Package layout

```
offline_ai/
  core/             LocalAI facade, Settings, hardware detection
  database/         SQLAlchemy models, repositories, ID minting, Alembic
  ingestion/        File / batch loaders (JSON, JSONL, CSV, TXT, tweets)
  memory/           Raw docs, embeddings, FAISS, graph helpers, RelationStore,
                    conversation promotion
  retrieval/        Lexical (FTS5), semantic, hybrid fusion, rerank, filters
  extraction/       Entities, claims, facts, topics, events → structured memory
  evidence/         Ask pipeline: retrieve → generate → cite → validate
  llm/              Pluggable backends + grounded_qa (intent / extractive compose)
  training/         LoRA adapters, experience replay, evaluation helpers
  api/              Optional FastAPI (localhost by default)
  cli/              Typer CLI, REPL chat, Persian desktop UI
  utils/            Hashing, paths, logging, backup, Persian normalize, console RTL

scripts/
  setup_deep_model.py   Register / verify local GGUF under workspace/models

tests/                  Unit + integration (Persian, multi-hop, UI, grounding)
```

Runtime data lives under a **workspace directory** (default `./workspace`):

```
workspace/
  db/offline_ai.db          # documents, chunks, claims, entities, answers, …
  vectors/faiss.index       # optional neural / hashing vectors
  vectors/id_map.json       # FAISS row ↔ application IDs (authoritative)
  config/policies.yaml      # citation / security policies (enforced in code)
  models/                   # optional local weights (GGUF, HF dirs) — not in git
  adapters/                 # LoRA versions (adapter_v001, …)
  training/
  logs/
  backups/
```

Repo root also holds `config.yaml` and `models.yaml` (model registry defaults).

---

## Component map

| Concern | Primary modules | Role |
|---------|-----------------|------|
| Public API | `core.engine.LocalAI` | `add` / `ask` / `search` / ingest / `rebuild_knowledge` / `doctor` |
| Config | `core.config`, `config.yaml`, `models.yaml`, `workspace/config/policies.yaml` | Policies enforced in validators, not only prompts |
| Hardware | `core.hardware.HardwareDetector` | CUDA / CPU, VRAM, `n_gpu_layers`, context recommendation |
| Persistence | `database.*`, `memory.raw` | SQLite + FTS5; Alembic migrations |
| Vectors | `memory.semantic`, `memory.embedding_backends` | FAISS; app IDs authoritative |
| Retrieval | `retrieval.hybrid` | Weighted fusion: lexical + semantic + entity + metadata + graph |
| Structured memory | `extraction.*`, `memory.relations`, `memory.graph` | Claims / relationships for multi-hop |
| Grounded QA | `evidence.engine`, `llm.grounded_qa`, `evidence.citation`, `evidence.grounding` | Answer composition + validation |
| Models | `llm.factory` | GGUF / Transformers if present; else `ExtractiveLLMBackend` |
| Persian UX | `cli.persian_ui`, `utils.persian`, `utils.console` | tkinter RTL window; normalize ي/ک, Jalali dates |
| CLI | `cli.main` | `offline-ai` entry point |
| HTTP | `api.app.create_app` | Optional; binds `127.0.0.1` by default |

---

## Write path: `add` / ingest

```
input (str | list | dict | file)
  → IngestionPipeline / IngestItem
  → RawMemory: mint DOC-…, store immutable original_text, content/source hashes
  → de-dupe on source_hash (same source+text → existing doc)
  → chunk → DocumentChunk (CHK-…)
  → after_document hook (LocalAI):
        MemoryManager.index_document  (embed + FAISS upsert)
        ExtractionPipeline.process_document
            → entities + mentions
            → claims (subject/predicate/object + source_span)
            → topics / events
            → Relationship edges (ENTITY —pred→ ENTITY|literal)
  → return {document_id, created, …}
```

**Operators:** `ai.add(...)`, `offline-ai ingest <path>`, desktop UI “ذخیره”.
**Rebuild:** `ai.rebuild_knowledge()` / `offline-ai rebuild` re-runs extraction
over existing documents when claim patterns change (no re-ingest required).

---

## Read path: `ask`

`EvidenceEngine.ask` is intentionally **graph-first**, then hybrid retrieval.

```
query
  ├─① RelationStore.answer_role_question (structured multi-hop)
  │     e.g. قاتل علی از چه زمانی پیدا نشده
  │          → killed(?, علی) → missing_since(killer, date)
  │     if hop resolved:
  │       • extractive: deterministic Persian answer + [DOC-…] + spans
  │       • neural: polish ONLY the graph facts (must keep killer/victim/date)
  │     return grounded answer (skip free RAG invent)
  │
  ├─② Hybrid retrieval (FTS5 + FAISS + filters + rerank)
  │     + multi-hop bridge queries (victim, قتل X, related DOC ids)
  │     + Persian token-overlap bridging across docs
  │
  ├─③ Claim triples for query (RelationStore.claims_for_query)
  │
  ├─④ Generation
  │     • neural: evidence-only / relations-first system prompt
  │     • extractive: llm.grounded_qa (intent → FactSheet → composed answer)
  │
  ├─⑤ CitationManager: extract + resolve [DOC-|CLAIM-|…]
  ├─⑥ GroundingValidator: reject unknown cites / weak overlap
  └─⑦ Persist QueryRecord + AnswerRecord → structured response dict
```

Insufficient evidence surfaces a fixed message (Persian or English per query
language). The model is not allowed to invent compensating facts.

### Multi-hop example (delivery scenario)

| Stored text | Extracted claims |
|-------------|------------------|
| محمد امیری علی را به قتل رساند | `محمد امیری —[killed]→ علی` |
| محمد امیری از ۱۳ فروردین به بعد پیدا نشد | `محمد امیری —[missing_since]→ ۱۳ فروردین …` |

Question: «قاتل علی از چه زمانی پیدا نشده؟»

1. Intent / role: victim = `علی`, wants missing-since of killer.
2. `resolve_killer("علی")` → محمد امیری.
3. `resolve_missing_since("محمد امیری")` → ۱۳ فروردین ….
4. Answer cites both source documents; LLM (if present) may only rewrite wording.

---

## Memory layers

| Layer | Storage | Guarantees |
|-------|---------|------------|
| Raw | `documents.original_text` | Immutable; hash-based de-dupe per source |
| Chunks | `document_chunks` | Offsets into original; embedding unit |
| Semantic | FAISS + `vector_mappings` / `id_map.json` | Rebuildable; IDs not FAISS row indices |
| Structured | `entities`, `claims`, `relationships`, topics, events | Provenance: `source_document_id`, spans, confidence |
| Conversational | query/answer tables + optional promotion | Short-term unless explicitly promoted |

See [docs/MEMORY.md](docs/MEMORY.md).

---

## Retrieval

`HybridRetriever` fuses scored channels (weights configurable via Settings):

| Channel | Typical weight | Source |
|---------|----------------|--------|
| Semantic | 0.45 | Embedding + FAISS |
| Lexical | 0.30 | SQLite FTS5 |
| Entity | 0.10 | Entity mentions overlap |
| Metadata | 0.05 | author / source filters |
| Graph | 0.10 | relation-linked documents |

Post-fusion: local reranker (`retrieval.reranker`), optional author/source/date
filters (including relative Persian/English date phrases).

Without a neural embedding model, a **hashing embedding backend** keeps FAISS
usable so the rest of the stack does not hard-depend on torch.

---

## Extraction and claim graph

`ExtractionPipeline` runs on every newly created document:

- **Entities** — person / place / org heuristics (`extraction.entities`)
- **Claims** — predicate patterns (`extraction.claims`), including Persian:
  - `killed`, `missing_since`, `had_meeting`, `met_with`, `traveled_to`,
    `located_in`, `occurred_on`, `associated_with`, …
- **Facts** — richer typed sheets for extractive QA (`extraction.facts`:
  meetings, trips, murder, missing)
- **Topics / events** — lightweight classifiers
- **Relationships** — edges between entity IDs (and date/place literals)

`RelationStore` is the **read API** over claims for ask-time multi-hop.
`KnowledgeGraph` supports broader graph queries / traversal helpers.

Normalization (`utils.persian.normalize_persian`) collapses Arabic/Persian
letter variants and whitespace so «علي» / «علی» resolve consistently.

---

## LLM backends and grounding

### Selection (`llm.factory.create_llm_backend`)

1. Prefer `models.yaml` default LLM key if `local_path` exists under workspace.
2. Else try registered GGUF (e.g. Qwen2.5-7B / 3B) via `llama_cpp`.
3. Else Transformers HF dirs if configured and present.
4. Else **`ExtractiveLLMBackend`** — no weights; compose from evidence + graph.

Hardware detector sets device, context length, and `n_gpu_layers` for GGUF.

### Grounding contract

- Prompts instruct: use only provided relations/evidence; never invent.
- Neural polish of graph answers is checked for required entities/dates; on
  failure the deterministic graph answer is kept.
- `CitationManager` resolves IDs against DB; unknown citations fail closed.
- `GroundingValidator` enforces `policies.citation_required` and token-overlap
  heuristics; `allow_unsupported_claims` is off by default.

Policies live in `workspace/config/policies.yaml` and are applied in code.

---

## Surfaces

| Surface | Entry | Notes |
|---------|-------|-------|
| Python | `from offline_ai import LocalAI` | Primary integration API |
| Module UI | `python -m offline_ai` / `examples/ask_ali.py` | tkinter «ai» window; reliable RTL typing on Windows |
| CLI | `offline-ai …` | `ask`, `chat`, `ingest`, `rebuild`, `doctor`, model/adapter subcommands |
| REPL | `offline-ai chat` | Terminal chat (RTL display helpers) |
| HTTP | `offline_ai.api` optional extra | Localhost-only default |

---

## Identifier scheme

| Kind | Format | Example |
|------|--------|---------|
| Document | `DOC-{n:09d}` | `DOC-000000001` |
| Chunk | `CHK-{n:09d}` | `CHK-000000001` |
| Claim | `CLAIM-{n:09d}` | `CLAIM-000000001` |
| Entity | `ENTITY-{n:09d}` | `ENTITY-000000001` |
| Answer | `ANS-{n:09d}` | `ANS-000000001` |

FAISS internal row indices are **not** permanent keys. Application IDs in
`vector_mappings` / `id_map.json` are authoritative.

---

## Configuration and security posture

- **Workspace-scoped state:** all durable memory under `workspace/`.
- **Policies:** citation required, unsupported claims disabled, optional
  `allow_network` / telemetry flags checked by `doctor --offline`.
- **Secrets:** no cloud API keys required at runtime.
- **Git delivery:** `workspace/`, `*.gguf`, `.venv/` are gitignored; clone +
  `pip install -e ".[dev]"` + optional model download is the operator path.

See [docs/SECURITY.md](docs/SECURITY.md) and [OFFLINE_INSTALL.md](OFFLINE_INSTALL.md).

---

## Extending the system

Keep interfaces stable; prefer adapters over rewriting the facade.

| Goal | Where |
|------|--------|
| New ingest format | Loader under `ingestion/` + branch in `IngestionPipeline.ingest_file` |
| New embedding model | Implement `EmbeddingBackend`; register in `create_embedding_backend` |
| New LLM backend | Implement `LLMBackend`; wire in `llm.factory` |
| New vector store | Implement `VectorStore` (see `FaissVectorStore`) |
| Richer NER / IE | Replace rule extractors in `extraction/` while preserving provenance fields (`source_document_id`, offsets, confidence) |
| New multi-hop pattern | Add predicate in `extraction.claims` + hop logic in `RelationStore.answer_role_question` |
| New question intent | Extend `llm.grounded_qa.analyze_question` + compose path |
| Policy change | `policies.yaml` + enforce in `GroundingValidator` / `doctor` |

---

## Testing map (architecture-relevant)

| Area | Tests (indicative) |
|------|--------------------|
| Persian normalize / RTL | `tests/test_persian.py`, `test_console_rtl.py`, `test_persian_ui.py` |
| Multi-hop / graph | `tests/test_multihop_persian.py`, `test_relation_graph.py` |
| Grounded compose | `tests/test_grounded_qa.py` |
| Broader suite | `tests/` — retrieval, ingestion, citations, backup |

Run: `pytest` from the repo root with the package editable-installed.

---

## Related docs

- [docs/MEMORY.md](docs/MEMORY.md) — memory layers
- [docs/MODELS.md](docs/MODELS.md) — model registry
- [docs/API.md](docs/API.md) — Python API
- [docs/CLI.md](docs/CLI.md) — CLI reference
- [docs/TRAINING.md](docs/TRAINING.md) — adapters / LoRA
- [docs/SECURITY.md](docs/SECURITY.md) — offline / security posture
- [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) — operator issues
- [INSTALL.md](INSTALL.md) / [OFFLINE_INSTALL.md](OFFLINE_INSTALL.md)
- [CHANGELOG.md](CHANGELOG.md) — 0.2.0 delivery notes
