# Install (online machine)

## 1. Python

Use Python **3.12** when possible (best wheel support for PyTorch / FAISS).

```bash
py -3.12 -m venv .venv
.venv\Scripts\activate
python -m pip install -U pip setuptools wheel
```

## 2. Package

```bash
pip install -e ".[dev]"
```

Optional llama.cpp backend:

```bash
pip install llama-cpp-python
```

On Windows, if the wheel build fails, install a prebuilt wheel matching your Python/CUDA version from the llama-cpp-python releases, or use the Transformers backend only.

## 3. Verify

```bash
offline-ai --workspace ./workspace doctor
python -c "from offline_ai import LocalAI; print(LocalAI('./workspace').detect_hardware())"
pytest
```

## 4. Models

Download models **once** while online. See [OFFLINE_INSTALL.md](OFFLINE_INSTALL.md) and [models.yaml](models.yaml).
