# Installation

## Prerequisites

- Python **3.11+** (use **3.12** on Windows when possible for better binary wheels)
- `pip` / `venv`
- Optional: NVIDIA drivers for CUDA acceleration

## Create a virtual environment

```bash
py -3.12 -m venv .venv

# Windows
.venv\Scripts\activate

# Linux / macOS
# source .venv/bin/activate

python -m pip install -U pip setuptools wheel
```

## Install the package

Development install with tests and lint tools:

```bash
pip install -e ".[dev]"
```

Extras:

```bash
pip install -e ".[api]"          # FastAPI server
pip install -e ".[embeddings]"   # sentence-transformers
pip install -e ".[llm]"          # llama-cpp-python / Transformers
pip install -e ".[training]"     # peft / datasets
pip install -e ".[all]"          # all optional stacks
```

### llama-cpp-python on Windows

If building from source fails, install a matching prebuilt wheel from the
[llama-cpp-python releases](https://github.com/abetlen/llama-cpp-python/releases),
or use the Transformers backend only.

## Verify

```bash
offline-ai --workspace ./workspace doctor --offline
python -c "from offline_ai import LocalAI; print(LocalAI('./workspace').detect_hardware())"
pytest
```

## Models

Download embedding / LLM weights **once** while online and place them under the
paths configured in [`models.yaml`](models.yaml).

See [OFFLINE_INSTALL.md](OFFLINE_INSTALL.md) and
[docs/offline_requirements.md](docs/offline_requirements.md).
