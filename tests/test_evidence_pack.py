"""Evidence packing must stay inside small GGUF context windows."""

from __future__ import annotations

from offline_ai.evidence.engine import EvidenceEngine
from offline_ai.llm.extractive import ExtractiveLLMBackend


class _FakeCtxLLM(ExtractiveLLMBackend):
    n_ctx = 4096

    @property
    def model_name(self) -> str:
        return "fake-gguf-qwen"


def test_pack_prompt_evidence_truncates_huge_docs(tmp_path) -> None:
    from offline_ai import LocalAI

    ai = LocalAI(tmp_path / "ws")
    engine = EvidenceEngine(ai.db, ai.memory_manager, _FakeCtxLLM(), ai.settings.policies)
    huge = "علی " * 50_000  # would blow past 4096 if sent raw
    hits = [
        {"document_id": f"DOC-{i:09d}", "text": huge, "snippet": huge[:80]}
        for i in range(20)
    ]
    claims = [
        {
            "triple": f"شخص{i} —[met_with]→ شخص{i+1}",
            "document_id": f"DOC-{i:09d}",
            "subject": f"شخص{i}",
            "predicate": "met_with",
            "object": f"شخص{i+1}",
        }
        for i in range(30)
    ]
    rel, blocks, texts, doc_ids, safe_max = engine._pack_prompt_evidence(
        query="قاتل علی کیست؟",
        relations_block="\n".join(c["triple"] for c in claims),
        evidence_hits=hits,
        claim_rows=claims,
        completion_tokens=384,
    )
    packed = rel + "\n".join(blocks)
    assert engine._estimate_tokens(packed) < 3500
    assert len(blocks) <= 5
    assert all(len(t) <= 400 for t in texts)
    assert safe_max >= 64
    assert len(doc_ids) == len(blocks)
