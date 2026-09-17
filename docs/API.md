# Python API

```python
from offline_ai import LocalAI

ai = LocalAI("./workspace")

# Store knowledge
ai.add("Company X VPN was compromised.")
ai.learn(["fact one", "fact two"])
ai.add({"text": "...", "author": "alice", "source_url": "https://example.test/1"})

# Ask (answer text only)
print(ai.ask("What happened to Company X?", text_only=True))

# Ask (full structured response)
result = ai.ask("What happened to Company X?")
print(result["answer"])
print(result["documents"])
print(result["citations"])
```

## Other methods

```python
ai.ingest_tweets([{"text": "...", "author": "u"}])
ai.ingest_file("data.json")
ai.search("Company X attack")
ai.get_document("DOC-000000001")
ai.get_claim("CLAIM-000000001")
ai.get_entity("ENTITY-000000001")
ai.feedback(answer_id=result["answer_id"], rating=5, correction="...")
ai.stats()
ai.doctor(offline=True)
```

## Ask response fields

`answer`, `documents`, `claims`, `citations`, `evidence`, `retrieval`,
`warnings`, `grounded`, `answer_id`, `query_id`.
