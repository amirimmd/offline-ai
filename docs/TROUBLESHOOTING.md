# Troubleshooting

| Symptom | Fix |
|---------|-----|
| `GGUF model not found` | Place model under path in `models.yaml` or rely on extractive fallback |
| Weak semantic search | Install `sentence-transformers` + download e5 model locally |
| CUDA not detected | Install torch with CUDA; driver present; or use CPU mode |
| FAISS missing after copy | Restore from backup including `workspace/vectors/` |
| FK / DB errors | Delete corrupt DB only if acceptable; prefer restore from backup |
| `llama-cpp-python` build fails on Windows | Use prebuilt wheel or Transformers backend |
| Citations rejected | Expected when model invents IDs; grounding returns insufficient evidence |

Python: prefer 3.12 on Windows for wheels.
