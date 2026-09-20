"""Hybrid retrieval: lexical + semantic + filters + fusion + rerank."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from sqlalchemy import select

from offline_ai.database.models import Document, DocumentChunk
from offline_ai.database.session import Database
from offline_ai.memory.embeddings import EmbeddingBackend
from offline_ai.memory.semantic import VectorStore
from offline_ai.retrieval.filters import MetadataFilter, parse_relative_date
from offline_ai.retrieval.lexical import LexicalRetriever
from offline_ai.retrieval.reranker import Reranker, create_reranker
from offline_ai.retrieval.semantic import SemanticRetriever
from offline_ai.utils.logging import get_logger
from offline_ai.utils.persian import contains_persian, content_tokens, token_overlap

logger = get_logger(__name__)


@dataclass
class RetrievalWeights:
    semantic: float = 0.45
    lexical: float = 0.30
    entity: float = 0.10
    metadata: float = 0.05
    graph: float = 0.10


@dataclass
class HybridRetrievalResult:
    query: str
    results: list[dict[str, Any]] = field(default_factory=list)
    lexical_hits: int = 0
    semantic_hits: int = 0
    filters_applied: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "results": self.results,
            "lexical_hits": self.lexical_hits,
            "semantic_hits": self.semantic_hits,
            "filters_applied": self.filters_applied,
        }


class HybridRetriever:
    def __init__(
        self,
        db: Database,
        embedder: EmbeddingBackend,
        store: VectorStore,
        *,
        weights: RetrievalWeights | None = None,
        reranker: Reranker | None = None,
    ) -> None:
        self.db = db
        self.embedder = embedder
        self.store = store
        self.semantic = SemanticRetriever(embedder, store)
        self.weights = weights or RetrievalWeights()
        self.reranker = reranker or create_reranker()

    def search(
        self,
        query: str,
        *,
        top_k: int = 20,
        rerank_top_k: int = 8,
        source: str | None = None,
        author: str | None = None,
        after: datetime | None = None,
        before: datetime | None = None,
    ) -> HybridRetrievalResult:
        rel_after, rel_before = parse_relative_date(query)
        after = after or rel_after
        before = before or rel_before

        with self.db.session() as session:
            lex = LexicalRetriever(session).search(query, top_k=top_k)
            sem = self.semantic.search(query, top_k=top_k)
            query_tokens = content_tokens(query)
            persian_query = contains_persian(query)
            w = RetrievalWeights(
                semantic=0.25 if persian_query else self.weights.semantic,
                lexical=0.50 if persian_query else self.weights.lexical,
                entity=0.15 if persian_query else self.weights.entity,
                metadata=self.weights.metadata,
                graph=self.weights.graph,
            )

            # Map chunk hits to documents
            fused: dict[str, dict[str, Any]] = {}

            for hit in lex:
                did = hit["document_id"]
                fused.setdefault(
                    did,
                    {
                        "document_id": did,
                        "semantic_score": 0.0,
                        "lexical_score": 0.0,
                        "entity_score": 0.0,
                        "metadata_score": 0.0,
                        "graph_score": 0.0,
                        "channels": [],
                    },
                )
                fused[did]["lexical_score"] = max(fused[did]["lexical_score"], hit["score"])
                fused[did]["channels"].append("lexical")
                fused[did]["snippet"] = hit.get("snippet")
                fused[did]["source"] = hit.get("source")
                fused[did]["author"] = hit.get("author")

            for hit in sem:
                did = hit.get("document_id")
                if not did:
                    continue
                fused.setdefault(
                    did,
                    {
                        "document_id": did,
                        "semantic_score": 0.0,
                        "lexical_score": 0.0,
                        "entity_score": 0.0,
                        "metadata_score": 0.0,
                        "graph_score": 0.0,
                        "channels": [],
                    },
                )
                fused[did]["semantic_score"] = max(fused[did]["semantic_score"], hit["score"])
                fused[did]["channels"].append("semantic")
                fused[did]["chunk_id"] = hit.get("chunk_id")

            # Optional metadata filter
            allowed = None
            if any([source, author, after, before]):
                allowed = set(
                    MetadataFilter(session).filter_document_ids(
                        list(fused.keys()) or None,
                        source=source,
                        author=author,
                        after=after,
                        before=before,
                    )
                )

            results = []
            for did, row in fused.items():
                if allowed is not None and did not in allowed:
                    continue
                meta_bonus = 0.0
                entity_bonus = 0.0
                doc = session.execute(
                    select(Document).where(Document.document_id == did)
                ).scalar_one_or_none()
                if doc:
                    row["text"] = doc.original_text
                    row["source"] = doc.source
                    row["author"] = doc.author
                    row["timestamp"] = doc.timestamp.isoformat() if doc.timestamp else None
                    row["source_url"] = doc.source_url
                    row["snippet"] = doc.original_text[:280]
                    q_l = query.lower()
                    if doc.source and doc.source.lower() in q_l:
                        meta_bonus += 0.5
                    if doc.author and str(doc.author).lower() in q_l:
                        meta_bonus += 0.5
                    overlap = token_overlap(query, doc.original_text)
                    if overlap:
                        entity_bonus = min(1.0, overlap / max(len(query_tokens), 1))
                    elif query_tokens and not row.get("lexical_score"):
                        # Drop semantic-only noise with no shared content tokens
                        continue
                row["metadata_score"] = meta_bonus
                row["entity_score"] = max(row.get("entity_score", 0.0), entity_bonus)
                row["final_score"] = (
                    row["semantic_score"] * w.semantic
                    + row["lexical_score"] * w.lexical
                    + row["entity_score"] * w.entity
                    + row["metadata_score"] * w.metadata
                    + row["graph_score"] * w.graph
                )
                row["why_retrieved"] = self._explain(row)
                results.append(row)

            results.sort(key=lambda x: x["final_score"], reverse=True)
            results = results[:top_k]
            results = self.reranker.rerank(query, results, top_k=rerank_top_k)

            out = HybridRetrievalResult(
                query=query,
                results=results,
                lexical_hits=len(lex),
                semantic_hits=len(sem),
                filters_applied={
                    "source": source,
                    "author": author,
                    "after": after.isoformat() if after else None,
                    "before": before.isoformat() if before else None,
                },
            )
            logger.info(
                "Hybrid search complete",
                extra={"event": "hybrid_search", "component": "retrieval"},
            )
            return out

    @staticmethod
    def _explain(row: dict[str, Any]) -> str:
        parts = []
        if row.get("lexical_score", 0) > 0:
            parts.append(f"lexical={row['lexical_score']:.3f}")
        if row.get("semantic_score", 0) > 0:
            parts.append(f"semantic={row['semantic_score']:.3f}")
        if row.get("entity_score", 0) > 0:
            parts.append(f"entity={row['entity_score']:.3f}")
        channels = ",".join(sorted(set(row.get("channels") or [])))
        return f"channels=[{channels}]; " + ", ".join(parts)
