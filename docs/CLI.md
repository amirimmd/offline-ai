# CLI

```bash
offline-ai --workspace ./workspace doctor --offline
offline-ai --workspace ./workspace ingest file.json
offline-ai --workspace ./workspace search "query"
offline-ai --workspace ./workspace ask "query"
offline-ai --workspace ./workspace document DOC-000000001
offline-ai --workspace ./workspace entity ENTITY-000000001
offline-ai --workspace ./workspace claim CLAIM-000000001
offline-ai --workspace ./workspace stats
offline-ai --workspace ./workspace backup ./backups
offline-ai --workspace ./workspace restore ./backups/offline_ai_backup_...
offline-ai --workspace ./workspace model list
offline-ai --workspace ./workspace model status
offline-ai --workspace ./workspace adapter list
offline-ai --workspace ./workspace adapter train
offline-ai --workspace ./workspace adapter activate adapter_v001
```

Local API:

```bash
uvicorn offline_ai.api.app:create_app --factory --host 127.0.0.1 --port 8765
```
