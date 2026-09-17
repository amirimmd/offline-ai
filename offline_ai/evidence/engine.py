"""Ask pipeline: retrieve evidence, call the LLM, validate citations."""

from __future__ import annotations

from typing import Any

from offline_ai.core.config import Policies
from offline_ai.database.ids import next_id
from offline_ai.database.models import AnswerRecord, QueryRecord
from offline_ai.database.session import Database
from offline_ai.evidence.citation import CitationManager
from offline_ai.evidence.grounding import GroundingValidator
from offline_ai.llm.base import LLMBackend, LLMMessage
from offline_ai.memory.manager import MemoryManager
from offline_ai.utils.logging import get_logger

logger = get_logger(__name__)


class EvidenceEngine:
    def __init__(
        self,
        db: Database,
        memory: MemoryManager,
        llm: LLMBackend,
        policies: Policies,
        *,
        top_k: int = 20,
        rerank_top_k: int = 8,
        min_evidence_score: float = 0.15,
    ) -> None:
        self.db = db
        self.memory = memory
        self.llm = llm
        self.policies = policies
        self.citations = CitationManager(db)
        self.grounding = GroundingValidator(
            citation_required=policies.citation_required,
            allow_unsupported_claims=policies.allow_unsupported_claims,
            insufficient_message=policies.insufficient_evidence_message,
        )
        self.top_k = top_k
        self.rerank_top_k = rerank_top_k
        self.min_evidence_score = min_evidence_score

    def ask(self, query: str, **kwargs: Any) -> dict[str, Any]:
        retrieval = self.memory.search(
            query,
            top_k=kwargs.get("top_k", self.top_k),
            rerank_top_k=kwargs.get("rerank_top_k", self.rerank_top_k),
            source=kwargs.get("source"),
            author=kwargs.get("author"),
            after=kwargs.get("after"),
            before=kwargs.get("before"),
        )
        evidence_hits = [
            h
            for h in retrieval.get("results") or []
            if float(h.get("rerank_score") or h.get("final_score") or h.get("score") or 0)
            >= self.min_evidence_score
            or h.get("lexical_score", 0) > 0
            or h.get("semantic_score", 0) > 0
        ]
        # If scores are low but we have results, still take top ones
        if not evidence_hits:
            evidence_hits = (retrieval.get("results") or [])[: self.rerank_top_k]

        if not evidence_hits:
            return self._finalize(
                query,
                answer=self.policies.insufficient_evidence_message,
                documents=[],
                claims=[],
                citations=[],
                evidence=[],
                retrieval=retrieval,
                warnings=["No documents retrieved"],
                grounded=False,
            )

        evidence_blocks = []
        evidence_texts = []
        doc_ids = []
        for h in evidence_hits:
            did = h["document_id"]
            text = h.get("text") or h.get("snippet") or ""
            evidence_blocks.append(f"[{did}] {text}")
            evidence_texts.append(text)
            doc_ids.append(did)

        system = (
            "You are a grounded analyst. Answer ONLY using the evidence. "
            "Cite documents as [DOC-...]. If evidence is insufficient, say exactly: "
            f"{self.policies.insufficient_evidence_message}"
        )
        user = (
            "EVIDENCE:\n"
            + "\n\n".join(evidence_blocks)
            + f"\n\nQUESTION:\n{query}\n\n"
            + "Write a concise answer with citations."
        )
        result = self.llm.chat(
            [
                LLMMessage(role="system", content=system),
                LLMMessage(role="user", content=user),
            ],
            max_tokens=kwargs.get("max_tokens", 512),
            temperature=kwargs.get("temperature", 0.1),
        )
        answer_text = result.text.strip()
        cited = self.citations.extract_citation_ids(answer_text)
        ok_cites, bad = self.citations.validate_answer_citations(answer_text)
        grounding = self.grounding.validate(
            answer_text,
            evidence_texts=evidence_texts,
            valid_citation_ids=doc_ids,
            cited_ids=cited,
            bad_citations=bad,
        )
        answer_text = grounding.answer
        cited = self.citations.extract_citation_ids(answer_text)
        _, bad2 = self.citations.validate_answer_citations(answer_text)
        if bad2:
            answer_text = self.policies.insufficient_evidence_message
            grounding.ok = False
            grounding.warnings.append(f"Post-check rejected citations: {bad2}")

        citation_records = []
        for cid in cited:
            try:
                citation_records.append(
                    {
                        "ref": f"[{cid}]",
                        "resolved": self.citations.resolve_citation(cid),
                    }
                )
            except Exception:
                pass

        documents = []
        for did in doc_ids:
            src = self.citations.get_source("document", did)
            if src:
                hit = next((h for h in evidence_hits if h["document_id"] == did), {})
                documents.append(
                    {
                        **src,
                        "score": hit.get("rerank_score") or hit.get("final_score"),
                        "why_retrieved": hit.get("why_retrieved"),
                    }
                )

        return self._finalize(
            query,
            answer=answer_text,
            documents=documents,
            claims=[],
            citations=citation_records,
            evidence=[
                {
                    "document_id": h["document_id"],
                    "span": (h.get("text") or h.get("snippet") or "")[:500],
                    "score": h.get("rerank_score") or h.get("final_score"),
                    "why": h.get("why_retrieved"),
                }
                for h in evidence_hits
            ],
            retrieval=retrieval,
            warnings=grounding.warnings,
            grounded=grounding.ok,
        )

    def _finalize(
        self,
        query: str,
        *,
        answer: str,
        documents: list,
        claims: list,
        citations: list,
        evidence: list,
        retrieval: dict,
        warnings: list,
        grounded: bool,
    ) -> dict[str, Any]:
        with self.db.session() as session:
            qid = next_id(session, "QRY")
            aid = next_id(session, "ANS")
            session.add(QueryRecord(query_id=qid, text=query))
            session.flush()
            payload = {
                "answer": answer,
                "documents": documents,
                "claims": claims,
                "citations": citations,
                "evidence": evidence,
                "retrieval": {
                    "lexical_hits": retrieval.get("lexical_hits"),
                    "semantic_hits": retrieval.get("semantic_hits"),
                    "filters_applied": retrieval.get("filters_applied"),
                },
                "warnings": warnings,
                "answer_id": aid,
                "query_id": qid,
                "grounded": grounded,
            }
            session.add(
                AnswerRecord(
                    answer_id=aid,
                    query_id=qid,
                    answer_text=answer,
                    structured_json=payload,
                    grounded=grounded,
                )
            )
            session.commit()
        logger.info(
            "Ask completed",
            extra={"event": "ask", "component": "evidence"},
        )
        return payload
