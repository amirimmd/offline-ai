# Security

- No telemetry
- No automatic network calls in core paths
- No API keys required for core functionality
- `policies.security.allow_network: false` by default
- Path traversal rejected via `safe_join`
- Metadata stored as JSON; documents are not executed
- Structured logs omit raw document text by default

Run:

```bash
offline-ai doctor --offline
```
