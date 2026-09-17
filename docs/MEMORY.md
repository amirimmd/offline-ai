# Memory architecture

## Layers

1. **Raw memory** — immutable `original_text` in SQLite (`documents`). Never rewritten.
2. **Semantic memory** — FAISS vectors keyed by application `chunk_id` → `document_id`.
3. **Structured memory** — entities, claims, topics, events, relationships.
4. **Conversation memory** — short-term only; promotion to long-term is explicit.

## Guarantees

- Replacing LLM / LoRA / embedding model does not delete documents.
- Duplicate ingestion of identical source+text is de-duplicated via `source_hash`.
- Same text from different sources is kept as separate documents.
- Citations must resolve to stored IDs before answers are returned.

## Paths

| Artifact | Location |
|----------|----------|
| SQLite | `workspace/db/offline_ai.db` |
| FAISS | `workspace/vectors/faiss.index` |
| ID map | `workspace/vectors/id_map.json` |
