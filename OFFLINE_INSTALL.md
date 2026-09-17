# Offline install / air-gap preparation

Do this on a machine with internet, then copy the project + model cache to the offline machine.

## Checklist before disconnecting

1. Create venv and `pip install -e ".[dev]"` (and `llama-cpp-python` if used).
2. Download embedding model into `workspace/models/embeddings/multilingual-e5-base`.
3. Download LLM GGUF into `workspace/models/llm/` (see `models.yaml`).
4. Optionally download reranker into `workspace/models/reranker/`.
5. Run `offline-ai doctor --offline` — must report `ok: true`.
6. Copy entire project directory + `workspace/` (or pack a backup) to the target PC.
7. On the target PC, use the same major Python version and activate the copied venv **or** reinstall wheels from a local wheelhouse (no PyPI).

## No runtime network

Core paths must not call:

- OpenAI / Anthropic / Google APIs
- Hugging Face Hub
- Cloud vector DBs

Set `policies.security.allow_network: false` (default).

See [docs/offline_requirements.md](docs/offline_requirements.md).
