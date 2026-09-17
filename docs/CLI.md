# CLI reference

Global option:

```bash
offline-ai --workspace ./workspace <command>
```

Default workspace is `./workspace`.

## Commands

| Command | Description |
|---------|-------------|
| `doctor [--offline]` | Environment and policy checks |
| `stats` | Workspace / component statistics |
| `ingest <file>` | Ingest JSON / JSONL / CSV / TXT / Markdown |
| `search "<query>"` | Hybrid retrieval |
| `ask "<query>"` | Grounded answer with citations |
| `document <DOC-…>` | Resolve a document |
| `entity <ENTITY-…>` | Resolve an entity |
| `claim <CLAIM-…>` | Resolve a claim |
| `backup <dest>` | Create a portable backup |
| `restore <src>` | Restore from a backup directory |
| `model list` | Show configured models |
| `model status` | Hardware / inference recommendation |
| `adapter list` | List LoRA adapter versions |
| `adapter train` | Create / train an adapter shell |
| `adapter activate <name>` | Activate an adapter version |
| `version` | Package version |

## Examples

```bash
offline-ai --workspace ./workspace doctor --offline
offline-ai --workspace ./workspace ingest examples/sample_tweets.json
offline-ai --workspace ./workspace search "Company X VPN"
offline-ai --workspace ./workspace ask "Find reports related to Company X"
offline-ai --workspace ./workspace document DOC-000000001
offline-ai --workspace ./workspace backup ./backups
```

## Local HTTP API

```bash
pip install -e ".[api]"
uvicorn offline_ai.api.app:create_app --factory --host 127.0.0.1 --port 8765
```

Binds to localhost by default. See [docs/API.md](API.md) for the Python surface;
HTTP routes mirror ingest / ask / search / document / feedback.
