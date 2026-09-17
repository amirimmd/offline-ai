# Offline AI

**Local, offline-first knowledge system** with persistent memory, hybrid retrieval, and grounded answers that cite real stored documents.

The language model reasons over retrieved evidence only. Ingested documents survive model swaps, adapter changes, restarts, and machine moves.

---

## Why this design

| Problem | Approach |
|---------|----------|
| Chatbots forget after restart | SQLite + FAISS workspace persistence |
| Models invent sources | Citation validation before every answer |
| Cloud / API lock-in | Fully local inference; no runtime API keys |
| Fine-tuning after every file | Memory first; optional LoRA on a schedule |

**Truth hierarchy:** original evidence → structured memory → retrieved evidence → model reasoning.

---

## Features

- Immutable document store with duplicate detection
- Hybrid search (FTS5 lexical + vector semantic + metadata / date filters)
- Grounded `ask` pipeline with resolvable `[DOC-…]` citations
- Local entity / claim / topic / event extraction
- Knowledge graph with document provenance
- Pluggable LLM backends (llama.cpp GGUF, Transformers, extractive)
- Optional LoRA adapter versioning and experience replay
- Python API, CLI (`offline-ai`), optional localhost FastAPI
- Backup / restore with checksum manifests
- Automated test suite

---

## Requirements

- Python **3.11+** (**3.12** recommended on Windows)
- Optional NVIDIA GPU (CPU mode supported)

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

Optional extras:

| Extra | Purpose |
|-------|---------|
| `api` | FastAPI + uvicorn |
| `embeddings` | sentence-transformers / torch |
| `llm` | llama-cpp-python / Transformers |
| `training` | peft / datasets |
| `all` | everything above |

Without neural model weights the system still runs (hashing embeddings + extractive grounded answers). For production quality, download models once into paths listed in [`models.yaml`](models.yaml). See [OFFLINE_INSTALL.md](OFFLINE_INSTALL.md).

---

## Quick start

### CLI

```bash
offline-ai --workspace ./workspace doctor --offline
offline-ai --workspace ./workspace ingest examples/sample_tweets.json
offline-ai --workspace ./workspace ask "Find reports related to Company X VPN"
offline-ai --workspace ./workspace document DOC-000000001
```

### Python

```python
from offline_ai import LocalAI

ai = LocalAI("./workspace")

ai.add("Company X announced that its VPN infrastructure was compromised.")
print(ai.ask("What happened to Company X VPN?", text_only=True))

result = ai.ask("What happened to Company X VPN?")
print(result["answer"])
print(result["documents"])   # real DOC-… records
```

Full minimal example: [`examples/simple_ask.py`](examples/simple_ask.py).

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
