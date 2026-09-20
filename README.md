# Offline AI

**Local, offline-first knowledge system** with persistent memory, hybrid retrieval, a relation graph, and grounded answers that cite real stored documents.

The language model reasons over retrieved evidence and stored relations only. Ingested documents survive model swaps, adapter changes, restarts, and machine moves.

**Repository:** [github.com/amirimmd/offline-ai](https://github.com/amirimmd/offline-ai)

---

## Why this design

| Problem | Approach |
|---------|----------|
| Chatbots forget after restart | SQLite + FAISS workspace persistence |
| Models invent sources | Citation validation before every answer |
| Cloud / API lock-in | Fully local inference; no runtime API keys |
| Fine-tuning after every file | Memory first; optional LoRA on a schedule |
| Shallow keyword answers | Claim / relation graph + multi-hop ask |

**Truth hierarchy:** original evidence → structured memory → retrieved evidence → model reasoning.

---

## Features

- Immutable document store with duplicate detection
- Hybrid search (FTS5 lexical + vector semantic + metadata / date filters)
- **Persian-first** retrieval, normalization, and grounded answers
- Claim extraction and **knowledge-graph relations** (`killed`, `missing_since`, `had_meeting`, …)
- Multi-hop ask (e.g. resolve “قاتل علی” then answer from linked facts)
- Grounded `ask` pipeline with resolvable `[DOC-…]` citations
- Local entity / claim / topic / event extraction
- Pluggable LLM backends (llama.cpp GGUF / Qwen2.5, Transformers, extractive fallback)
- Desktop Persian chat UI (`ai` window) for reliable RTL typing on Windows
- Optional LoRA adapter versioning and experience replay
- Python API, CLI (`offline-ai`), optional localhost FastAPI
- Backup / restore with checksum manifests
- Automated test suite

---

## Requirements

- Python **3.11+** (**3.12** recommended on Windows)
- Optional NVIDIA GPU (CPU mode supported; RTX 4070 8GB tested with Qwen2.5-3B GGUF)

---

## Install

```bash
py -3.12 -m venv .venv

# Windows
.venv\Scripts\activate

# Linux / macOS
# source .venv/bin/activate

pip install -U pip
pip install -e ".[dev]"
```

For the deep local LLM (recommended for delivery demos):

```bash
pip install -e ".[llm]"
# download GGUF once (Windows example via mirror):
# curl.exe -L -o workspace/models/llm/qwen2.5-3b-instruct-q4_k_m.gguf ^
#   "https://hf-mirror.com/bartowski/Qwen2.5-3B-Instruct-GGUF/resolve/main/Qwen2.5-3B-Instruct-Q4_K_M.gguf"
python scripts/setup_deep_model.py --size 3b --skip-download
```

Optional extras:

| Extra | Purpose |
|-------|---------|
| `api` | FastAPI + uvicorn |
| `embeddings` | sentence-transformers / torch |
| `llm` | llama-cpp-python / Transformers |
| `training` | peft / datasets |
| `all` | everything above |

Without neural weights the system still runs (hashing embeddings + extractive grounded answers over the relation graph). See [`models.yaml`](models.yaml) and [OFFLINE_INSTALL.md](OFFLINE_INSTALL.md).

---

## Quick start

### Persian desktop UI (recommended on Windows)

```bash
python examples/ask_ali.py
# or
python -m offline_ai
```

Opens the **ai** window: type Persian questions, store knowledge, get grounded answers.

### CLI

```bash
offline-ai --workspace ./workspace doctor --offline
offline-ai --workspace ./workspace chat
offline-ai --workspace ./workspace rebuild
offline-ai --workspace ./workspace ingest examples/sample_tweets.json
offline-ai --workspace ./workspace ask "Find reports related to Company X VPN"
```

### Python

```python
from offline_ai import LocalAI

ai = LocalAI("./workspace")

ai.add("علی در تاریخ 18 و 19 دی جلسه مهم داشته")
ai.add(
    "محمد امیری علی رو به قتل رسوند و در تاریخ 13 فروردین "
    "به بعد دیگر محمد امیری را پیدا نکردیم"
)

print(ai.ask("قاتل علی از چه زمانی پیدا نشده است ؟", text_only=True))
# → links killer relation to missing-since date with [DOC-…] citations

ai.rebuild_knowledge()  # refresh claims/graph after upgrades
```

English example: [`examples/simple_ask.py`](examples/simple_ask.py).

### Local HTTP API (optional)

```bash
pip install -e ".[api]"
uvicorn offline_ai.api.app:create_app --factory --host 127.0.0.1 --port 8765
```

---

## Documentation

| Document | Description |
|----------|-------------|
| [ARCHITECTURE.md](ARCHITECTURE.md) | System design and extension points |
| [CONTRIBUTING.md](CONTRIBUTING.md) | How to extend and contribute |
| [INSTALL.md](INSTALL.md) | Detailed installation |
| [OFFLINE_INSTALL.md](OFFLINE_INSTALL.md) | Air-gapped / offline setup |
| [docs/API.md](docs/API.md) | Python API reference |
| [docs/CLI.md](docs/CLI.md) | CLI reference |
| [docs/MEMORY.md](docs/MEMORY.md) | Memory layers |
| [docs/MODELS.md](docs/MODELS.md) | Model registry |
| [docs/TRAINING.md](docs/TRAINING.md) | Continual learning / LoRA |
| [docs/SECURITY.md](docs/SECURITY.md) | Security posture |
| [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) | Common issues |
| [docs/offline_requirements.md](docs/offline_requirements.md) | Pre-disconnect checklist |
| [CHANGELOG.md](CHANGELOG.md) | Release notes |

---

## Project layout

```
offline_ai/     Application package
config/         Behaviour policies
docs/           Operator and developer docs
examples/       Sample data and scripts
scripts/        Setup helpers (deep model download)
tests/          Pytest suite
workspace/      Runtime data (local; not committed)
```

See [ARCHITECTURE.md](ARCHITECTURE.md) for module responsibilities.

---

## Tests

```bash
pytest
```

---

## License

[MIT](LICENSE)
