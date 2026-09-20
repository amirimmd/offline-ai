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
from offline_ai.memory.relations import RelationStore
from offline_ai.utils.logging import get_logger
from offline_ai.utils.persian import (
    contains_persian,
    insufficient_message,
    role_focus_person,
    token_overlap,
)

logger = get_logger(__name__)

# Keep prompts inside small GGUF context windows (often 2048–4096).
_MAX_EVIDENCE_DOCS_NEURAL = 5
_MAX_SPAN_CHARS = 360
_MAX_RELATIONS_CHARS = 1200
_MAX_CLAIMS_IN_PROMPT = 8


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
        self.relations = RelationStore(db)

    def _context_limit(self) -> int:
        for attr in ("n_ctx", "max_context", "context_length"):
            val = getattr(self.llm, attr, None)
            if isinstance(val, int) and val > 0:
                return val
        return 4096

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        # Conservative for Persian/English mix (llama.cpp counts can be stricter).
        return max(1, (len(text or "") + 1) // 2)

    @staticmethod
    def _clip(text: str, limit: int = _MAX_SPAN_CHARS) -> str:
        clean = " ".join((text or "").replace("\u200c", " ").split())
        if len(clean) <= limit:
            return clean
        return clean[: limit - 1].rstrip() + "…"

    def _pack_prompt_evidence(
        self,
        *,
        query: str,
        relations_block: str,
        evidence_hits: list[dict[str, Any]],
        claim_rows: list[dict[str, Any]],
        completion_tokens: int,
    ) -> tuple[str, list[str], list[str], list[str], int]:
        """
        Fit relations + evidence into the model context window.

        Returns: relations_block, evidence_blocks, evidence_texts, doc_ids, safe_max_tokens
        """
        ctx = self._context_limit()
        # Reserve system prompt + query + completion + safety margin
        overhead = self._estimate_tokens(query) + 450
        budget_tokens = max(256, ctx - completion_tokens - overhead)
        budget_chars = budget_tokens * 2

        rel = self._clip(relations_block or "", _MAX_RELATIONS_CHARS)
        if claim_rows:
            # Rebuild a short relations block from top claims
            lines = []
            for c in claim_rows[:_MAX_CLAIMS_IN_PROMPT]:
                triple = c.get("triple") or f"{c.get('subject')} —[{c.get('predicate')}]→ {c.get('object')}"
                did = c.get("document_id") or ""
                lines.append(f"- {triple}" + (f" ({did})" if did else ""))
            rel = self._clip("\n".join(lines), _MAX_RELATIONS_CHARS)

        blocks: list[str] = []
        texts: list[str] = []
        doc_ids: list[str] = []
        used = len(rel)
        max_docs = _MAX_EVIDENCE_DOCS_NEURAL if self._is_neural_llm() else min(8, self.rerank_top_k)

        for h in evidence_hits[:max_docs]:
            did = h.get("document_id") or ""
            raw = h.get("snippet") or h.get("text") or ""
            # Prefer short snippet; never dump full multi-KB documents into the prompt.
            span = self._clip(raw, _MAX_SPAN_CHARS)
            if not span:
                continue
            block = f"[{did}] {span}"
            if used + len(block) + 2 > budget_chars and blocks:
                break
            blocks.append(block)
            texts.append(span)
            if did:
                doc_ids.append(did)
            used += len(block) + 2

        prompt_chars = used + len(query) + 800
        prompt_tokens = self._estimate_tokens("x" * prompt_chars)
        safe_max = max(64, min(completion_tokens, ctx - prompt_tokens - 32))
        return rel, blocks, texts, doc_ids, safe_max

    def _safe_chat(
        self,
        messages: list[LLMMessage],
        *,
        max_tokens: int,
        temperature: float,
    ) -> str:
        """Call LLM with context-safe max_tokens; raise only unexpected errors."""
        packed = "\n".join(m.content for m in messages)
        safe = max(64, min(max_tokens, self._context_limit() - self._estimate_tokens(packed) - 32))
        if safe < 64:
            raise RuntimeError("Prompt too large for model context window")
        result = self.llm.chat(messages, max_tokens=safe, temperature=temperature)
        return (result.text or "").strip()

    def _extractive_fallback(
        self,
        query: str,
        *,
        evidence_hits: list[dict[str, Any]],
        missing_msg: str,
    ) -> str:
        from offline_ai.llm.grounded_qa import compose_grounded_answer

        pairs: list[tuple[str, str]] = []
        for h in evidence_hits[:12]:
            did = str(h.get("document_id") or "")
            text = self._clip(h.get("snippet") or h.get("text") or "", 500)
            if did and text:
                pairs.append((did, text))
        if not pairs:
            return missing_msg
        try:
            return compose_grounded_answer(query, pairs) or missing_msg
        except Exception:
            top_id, top_text = pairs[0]
            return f"{top_text}\n[{top_id}]"

    @staticmethod
    def _format_relation_answer(hop: dict[str, Any]) -> str:
        kind = hop.get("kind")
        details: list[str] = []
        if kind == "killer_missing_since":
            lead = (
                f"بر اساس روابط ذخیره‌شده، قاتل {hop['victim']} یعنی {hop['killer']} "
                f"از {hop['since']} به بعد پیدا نشده است."
            )
            details.extend(
                [
                    f"رابطه: {hop['killer']} —[killed]→ {hop['victim']}",
                    f"رابطه: {hop['killer']} —[missing_since]→ {hop['since']}",
                    f"قاتل: {hop['killer']}",
                    f"قربانی: {hop['victim']}",
                    f"زمان پیدا نشدن: {hop['since']}",
                ]
            )
        else:
            lead = f"بر اساس روابط ذخیره‌شده، قاتل {hop['victim']} {hop['killer']} است."
            details.append(f"رابطه: {hop['killer']} —[killed]→ {hop['victim']}")
            if hop.get("since"):
                lead = (
                    f"بر اساس روابط ذخیره‌شده، قاتل {hop['victim']} {hop['killer']} است "
                    f"و از {hop['since']} به بعد پیدا نشده است."
                )
                details.append(f"رابطه: {hop['killer']} —[missing_since]→ {hop['since']}")
        lines = [lead, "", "جزئیات مستند"]
        lines.extend(details)
        lines.append("")
        lines.append("شاهد")
        seen_spans: set[str] = set()
        for span in hop.get("spans") or []:
            clean = " ".join((span or "").split())[:240]
            if clean and clean not in seen_spans:
                seen_spans.add(clean)
                lines.append(clean)
        for did in hop.get("document_ids") or []:
            lines.append(f"[{did}]")
        return "\n".join(lines)

    def _is_neural_llm(self) -> bool:
        name = (getattr(self.llm, "model_name", "") or "").lower()
        return "extractive" not in name

    def _deep_synthesize(
        self,
        query: str,
        *,
        relations_block: str,
        evidence_blocks: list[str],
        missing_msg: str,
        max_tokens: int,
        temperature: float,
    ) -> str:
        if contains_persian(query):
            system = (
                "تو یک تحلیل‌گر ارشد اطلاعاتی هستی. فقط از روابط و شواهد داده‌شده استفاده کن. "
                "استدلال چندگامی انجام بده: نقش‌ها را حل کن (مثلاً قاتلِ X)، زمان‌ها را وصل کن، "
                "تناقض‌ها را ذکر کن، و چیزی اختراع نکن. "
                "ساختار پاسخ:\n"
                "۱) جمله اول: جواب مستقیم و قطعی به همان سؤال\n"
                "۲) استدلال کوتاه چندگامی\n"
                "۳) جزئیات مستند (افراد، روابط، تاریخ‌ها)\n"
                "۴) حتماً ارجاع سند به صورت [DOC-...]\n"
                "هرگز خارج از شواهد/روابط چیزی نگو. "
                f"اگر شواهد کافی نیست دقیقاً بنویس: {missing_msg}"
            )
            user = (
                f"سؤال:\n{query}\n\n"
                f"روابط ذخیره‌شده (اولویت با این‌هاست):\n{relations_block or '(ندارد)'}\n\n"
                f"شواهد متنی:\n" + "\n\n".join(evidence_blocks) + "\n\n"
                "پاسخ عمیق و دقیق به فارسی بنویس و حداقل یک [DOC-...] بیاور."
            )
        else:
            system = (
                "You are a senior intelligence analyst. Use ONLY provided relations and evidence. "
                "Perform multi-hop reasoning; never invent facts. "
                "Structure: (1) direct answer (2) brief reasoning (3) documented details (4) [DOC-...] cites. "
                f"If insufficient, say exactly: {missing_msg}"
            )
            user = (
                f"QUESTION:\n{query}\n\n"
                f"STORED RELATIONS:\n{relations_block or '(none)'}\n\n"
                f"EVIDENCE:\n" + "\n\n".join(evidence_blocks) + "\n\n"
                "Write a deep, precise, professional answer."
            )
        result_text = self._safe_chat(
            [LLMMessage(role="system", content=system), LLMMessage(role="user", content=user)],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return result_text

    def ask(self, query: str, **kwargs: Any) -> dict[str, Any]:
        missing_msg = (
            insufficient_message(query)
            if contains_persian(query)
            else self.policies.insufficient_evidence_message
        )
        # Keep completion small so prompt + answer fit in 4k context.
        default_out = 384 if self._is_neural_llm() else 512
        max_tokens = int(kwargs.get("max_tokens", default_out))
        max_tokens = min(max_tokens, max(128, self._context_limit() // 8))
        temperature = float(kwargs.get("temperature", 0.05 if self._is_neural_llm() else 0.1))

        # Structured multi-hop from claim graph
        hop = self.relations.answer_role_question(query)
        claim_rows = self.relations.claims_for_query(query, limit=16)
        relations_block = "\n".join(
            f"- {c['triple']} (منبع {c['document_id']})" for c in claim_rows
        )
        if hop:
            relations_block = (
                f"- {hop.get('killer')} —[killed]→ {hop.get('victim')}\n"
                + (
                    f"- {hop.get('killer')} —[missing_since]→ {hop.get('since')}\n"
                    if hop.get("since")
                    else ""
                )
                + relations_block
            )

        # Neural deep path: rewrite the precise graph answer (no free invention).
        if hop and self._is_neural_llm():
            evidence = []
            documents = []
            doc_ids = []
            for did in hop.get("document_ids") or []:
                src = self.citations.get_source("document", did)
                text = (src or {}).get("original_text") or ""
                if src:
                    documents.append({**src, "score": 1.0, "why_retrieved": "relation_graph"})
                if text:
                    evidence.append(
                        {
                            "document_id": did,
                            "span": text[:500],
                            "score": 1.0,
                            "why": "stored_relation",
                        }
                    )
                    doc_ids.append(did)
            base = self._format_relation_answer(hop)
            try:
                polish_system = (
                    "تو یک تحلیل‌گر ارشد هستی. فقط حقایق داده‌شده را به نثر حرفه‌ای و عمیق "
                    "بازنویسی کن. هیچ شخص، مکان، تاریخ یا رابطه‌ای اضافه نکن. "
                    "ساختار: جواب مستقیم، استدلال کوتاه چندگامی، جزئیات، ارجاع [DOC-...]."
                )
                polish_user = (
                    f"سؤال:\n{query}\n\n"
                    f"حقایق قطعی از گراف دانش:\n{self._clip(base, 1500)}\n\n"
                    "همین را عمیق و دقیق به فارسی بنویس؛ چیزی اختراع نکن."
                )
                polished = self._safe_chat(
                    [
                        LLMMessage(role="system", content=polish_system),
                        LLMMessage(role="user", content=polish_user),
                    ],
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                if doc_ids and not self.citations.extract_citation_ids(polished):
                    polished = polished.rstrip() + "\n" + " ".join(f"[{d}]" for d in doc_ids)
                must = [hop.get("killer"), hop.get("victim"), hop.get("since")]
                must = [m for m in must if m]
                if polished and all(
                    (str(m).split()[0] in polished) or (str(m) in polished) for m in must
                ):
                    answer_text = polished
                    warnings = ["Deep LLM polished relation-graph answer"]
                else:
                    answer_text = base
                    warnings = ["Kept raw relation-graph answer for precision"]
            except Exception as exc:
                logger.warning("Deep LLM polish failed", exc_info=exc)
                answer_text = base
                warnings = ["Deep LLM unavailable; relation-graph answer"]

            cited = self.citations.extract_citation_ids(answer_text)
            citation_records = []
            for cid in cited:
                try:
                    citation_records.append(
                        {"ref": f"[{cid}]", "resolved": self.citations.resolve_citation(cid)}
                    )
                except Exception:
                    pass
            return self._finalize(
                query,
                answer=answer_text,
                documents=documents,
                claims=hop.get("claims") or claim_rows,
                citations=citation_records,
                evidence=evidence,
                retrieval={
                    "lexical_hits": 0,
                    "semantic_hits": 0,
                    "filters_applied": {"relations": True, "deep_llm": True},
                },
                warnings=warnings,
                grounded=True,
            )

        # Extractive / no-neural: keep deterministic relation answer
        if hop:
            answer_text = self._format_relation_answer(hop)
            evidence = []
            documents = []
            claim_payload = []
            for c in hop.get("claims") or []:
                claim_payload.append(c)
                evidence.append(
                    {
                        "document_id": c.get("document_id"),
                        "span": (c.get("span") or "")[:500],
                        "score": 1.0,
                        "why": "stored_relation",
                    }
                )
            for did in hop.get("document_ids") or []:
                src = self.citations.get_source("document", did)
                if src:
                    documents.append({**src, "score": 1.0, "why_retrieved": "relation_graph"})
            cited = self.citations.extract_citation_ids(answer_text)
            citation_records = []
            for cid in cited:
                try:
                    citation_records.append(
                        {"ref": f"[{cid}]", "resolved": self.citations.resolve_citation(cid)}
                    )
                except Exception:
                    pass
            return self._finalize(
                query,
                answer=answer_text,
                documents=documents,
                claims=claim_payload,
                citations=citation_records,
                evidence=evidence,
                retrieval={"lexical_hits": 0, "semantic_hits": 0, "filters_applied": {"relations": True}},
                warnings=[],
                grounded=True,
            )

        top_k = kwargs.get("top_k", self.top_k)
        rerank_top_k = kwargs.get("rerank_top_k", self.rerank_top_k)
        common = {
            "source": kwargs.get("source"),
            "author": kwargs.get("author"),
            "after": kwargs.get("after"),
            "before": kwargs.get("before"),
        }
        retrieval = self.memory.search(query, top_k=top_k, rerank_top_k=rerank_top_k, **common)

        # Multi-hop bridge: role questions need victim + crime docs, not only lexical hits.
        extra_queries: list[str] = []
        victim = role_focus_person(query) if contains_persian(query) else None
        if victim:
            extra_queries.extend(
                [
                    victim,
                    f"قتل {victim}",
                    f"{victim} پیدا",
                    "به قتل",
                    "پیدا نکردیم",
                ]
            )
        for did in self.relations.related_document_ids(query):
            extra_queries.append(did)

        merged: dict[str, dict] = {}
        for h in retrieval.get("results") or []:
            merged[h["document_id"]] = h
        for eq in extra_queries:
            if eq.startswith("DOC-"):
                src = self.citations.get_source("document", eq)
                if src and eq not in merged:
                    merged[eq] = {
                        "document_id": eq,
                        "text": src.get("original_text") or "",
                        "snippet": (src.get("original_text") or "")[:280],
                        "rerank_score": 0.9,
                        "final_score": 0.9,
                        "lexical_score": 1,
                        "why_retrieved": "relation_store",
                    }
                continue
            extra = self.memory.search(eq, top_k=max(8, top_k // 2), rerank_top_k=rerank_top_k, **common)
            for h in extra.get("results") or []:
                did = h["document_id"]
                if did not in merged:
                    merged[did] = h
                else:
                    old = float(merged[did].get("rerank_score") or merged[did].get("final_score") or 0)
                    new = float(h.get("rerank_score") or h.get("final_score") or 0)
                    if new > old:
                        merged[did] = h

        ranked = sorted(
            merged.values(),
            key=lambda h: float(h.get("rerank_score") or h.get("final_score") or h.get("score") or 0),
            reverse=True,
        )
        evidence_hits = [
            h
            for h in ranked
            if float(h.get("rerank_score") or h.get("final_score") or h.get("score") or 0)
            >= self.min_evidence_score
            or h.get("lexical_score", 0) > 0
            or h.get("entity_score", 0) > 0
            or h.get("why_retrieved") == "relation_store"
        ]
        if contains_persian(query):
            direct = [
                h
                for h in evidence_hits
                if token_overlap(query, h.get("text") or h.get("snippet") or "") > 0
                or h.get("why_retrieved") == "relation_store"
            ]
            if direct:
                seed_text = " ".join((h.get("text") or h.get("snippet") or "") for h in direct)
                bridged = []
                for h in evidence_hits:
                    text = h.get("text") or h.get("snippet") or ""
                    if h in direct:
                        bridged.append(h)
                    elif token_overlap(seed_text, text) > 0:
                        bridged.append(h)
                    elif any(k in text for k in ("قتل", "قاتل", "پیدا نکرد", "پیدا نشد", "به بعد")):
                        if victim and victim in text:
                            bridged.append(h)
                evidence_hits = bridged or direct
            else:
                evidence_hits = [
                    h
                    for h in evidence_hits
                    if token_overlap(query, h.get("text") or h.get("snippet") or "") > 0
                ]
        if not evidence_hits:
            evidence_hits = [
                h
                for h in ranked[: self.rerank_top_k]
                if not contains_persian(query)
                or token_overlap(query, h.get("text") or h.get("snippet") or "") > 0
                or (victim and victim in (h.get("text") or ""))
                or h.get("why_retrieved") == "relation_store"
            ]

        if not evidence_hits:
            return self._finalize(
                query,
                answer=missing_msg,
                documents=[],
                claims=[],
                citations=[],
                evidence=[],
                retrieval=retrieval,
                warnings=["No documents retrieved"],
                grounded=False,
            )

        rel_packed, evidence_blocks, evidence_texts, doc_ids, safe_max = self._pack_prompt_evidence(
            query=query,
            relations_block=relations_block,
            evidence_hits=evidence_hits,
            claim_rows=claim_rows,
            completion_tokens=max_tokens,
        )
        max_tokens = safe_max
        warnings_extra: list[str] = []
        if len(evidence_hits) > len(doc_ids):
            warnings_extra.append(
                f"Truncated evidence to {len(doc_ids)} docs for model context ({self._context_limit()})"
            )

        answer_text = ""
        try:
            if self._is_neural_llm():
                answer_text = self._deep_synthesize(
                    query,
                    relations_block=rel_packed,
                    evidence_blocks=evidence_blocks,
                    missing_msg=missing_msg,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
            elif contains_persian(query):
                system = (
                    "تو یک تحلیل‌گر ارشد مبتنی بر شواهد هستی. فقط از شواهد و روابط ذخیره‌شده استفاده کن. "
                    "نیت سؤال را بفهم و اگر لازم است چند سند/رابطه را به هم وصل کن. "
                    "ساختار: ۱) جمله اول جواب مستقیم ۲) جزئیات مستند ۳) ارجاع [DOC-...]. "
                    f"اگر شواهد کافی نیست دقیقاً بگو: {missing_msg}"
                )
                user = (
                    "پاسخ حرفه‌ای و مستند به زبان فارسی بنویس.\n\nشواهد:\n"
                    + (f"روابط:\n{rel_packed}\n\n" if rel_packed else "")
                    + "\n\n".join(evidence_blocks)
                    + f"\n\nQUESTION:\n{query}"
                )
                answer_text = self._safe_chat(
                    [LLMMessage(role="system", content=system), LLMMessage(role="user", content=user)],
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
            else:
                system = (
                    "You are a senior grounded analyst. Answer ONLY using evidence and stored relations. "
                    f"If insufficient, say exactly: {self.policies.insufficient_evidence_message}"
                )
                user = (
                    "EVIDENCE:\n"
                    + (f"RELATIONS:\n{rel_packed}\n\n" if rel_packed else "")
                    + "\n\n".join(evidence_blocks)
                    + f"\n\nQUESTION:\n{query}"
                )
                answer_text = self._safe_chat(
                    [LLMMessage(role="system", content=system), LLMMessage(role="user", content=user)],
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
        except Exception as exc:
            err = str(exc).lower()
            logger.warning("LLM ask failed; using extractive fallback", exc_info=exc)
            answer_text = self._extractive_fallback(
                query, evidence_hits=evidence_hits, missing_msg=missing_msg
            )
            warnings_extra.append(f"LLM fallback: {exc}" if "context" in err or "token" in err else "LLM fallback to extractive")

        if not (answer_text or "").strip():
            answer_text = self._extractive_fallback(
                query, evidence_hits=evidence_hits, missing_msg=missing_msg
            )
            warnings_extra.append("Empty LLM answer; extractive fallback")

        cited = self.citations.extract_citation_ids(answer_text)
        ok_cites, bad = self.citations.validate_answer_citations(answer_text)
        grounding = self.grounding.validate(
            answer_text,
            evidence_texts=evidence_texts or [h.get("snippet") or h.get("text") or "" for h in evidence_hits[:5]],
            valid_citation_ids=doc_ids or [h["document_id"] for h in evidence_hits[:8]],
            cited_ids=cited,
            bad_citations=bad,
        )
        answer_text = grounding.answer
        grounding.warnings.extend(warnings_extra)
        cited = self.citations.extract_citation_ids(answer_text)
        _, bad2 = self.citations.validate_answer_citations(answer_text)
        if bad2:
            answer_text = missing_msg
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
            claims=claim_rows,
            citations=citation_records,
            evidence=[
                {
                    "document_id": h["document_id"],
                    "span": self._clip(h.get("text") or h.get("snippet") or "", 500),
                    "score": h.get("rerank_score") or h.get("final_score"),
                    "why": h.get("why_retrieved"),
                }
                for h in evidence_hits[:12]
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
