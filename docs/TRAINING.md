# Training / continual learning

Do **not** fine-tune after every document.

Pipeline:

```
Daily ingest → Persistent memory
Periodically → Dataset export → Experience replay → LoRA → adapter_vNNN
```

## AdapterManager

```python
from offline_ai.training.adapters import AdapterManager
mgr = AdapterManager("./workspace/adapters")
mgr.create_adapter()
mgr.train_adapter(dataset_path="workspace/training/data.jsonl")
mgr.activate_adapter("adapter_v001")
mgr.rollback_adapter()
```

Adapters are never overwritten. Rollback uses `active.json` previous pointer.

## Replay

`ReplayBuffer` strategies: `random`, `stratified`, `importance`.
