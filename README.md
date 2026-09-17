# Offline AI

Local, offline-first knowledge system with persistent memory, hybrid search,
and grounded answers with verifiable source IDs.

Stored documents outlive model swaps, adapter changes, and machine moves.
The language model is used for reasoning over retrieved evidence only.

## Requirements

- Python 3.11+ (3.12 recommended on Windows)
- Optional NVIDIA GPU (CPU mode supported)

## Install

```bash
py -3.12 -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
# source .venv/bin/activate

pip install -U pip
pip install -e ".[dev]"
```

Optional extras: `embeddings`, `llm`, `training`, `api`, `all`.

## Quick start

```bash
offline-ai --workspace ./workspace doctor --offline
offline-ai --workspace ./workspace ingest examples/sample_tweets.json
offline-ai --workspace ./workspace ask "Find reports related to Company X VPN"
```

```python
from offline_ai import LocalAI

ai = LocalAI("./workspace")
ai.add("Company X announced that its VPN infrastructure was compromised.")
print(ai.ask("What happened to Company X VPN?", text_only=True))
```

See `examples/simple_ask.py`.

## Documentation

| Document | Description |
|----------|-------------|
| [ARCHITECTURE.md](ARCHITECTURE.md) | System design and extension points |
| [INSTALL.md](INSTALL.md) | Installation |
| [OFFLINE_INSTALL.md](OFFLINE_INSTALL.md) | Air-gapped setup |
| [docs/API.md](docs/API.md) | Python API |
| [docs/CLI.md](docs/CLI.md) | CLI |
| [docs/MEMORY.md](docs/MEMORY.md) | Memory layers |
| [docs/MODELS.md](docs/MODELS.md) | Model configuration |
| [docs/TRAINING.md](docs/TRAINING.md) | Continual learning |
| [docs/SECURITY.md](docs/SECURITY.md) | Security |
| [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) | Common issues |
| [CHANGELOG.md](CHANGELOG.md) | Release notes |

## Tests

```bash
pytest
```

## License

MIT
