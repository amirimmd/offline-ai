"""Lexical retrieval: FTS5 plus a Unicode substring fallback for Persian."""

from __future__ import annotations

from typing import Any

from sqlalchemy import text
from sqlalchemy.orm import Session

from offline_ai.utils.persian import search_tokens, token_overlap


def _fts_query(tokens: list[str]) -> str:
    if not tokens:
        return '""'
    return " OR ".join(f'"{t}"' for t in tokens)


class LexicalRetriever:
    def __init__(self, session: Session) -> None:
        self.session = session

    def search(self, query: str, top_k: int = 20) -> list[dict[str, Any]]:
        tokens = search_tokens(query)
        by_id: dict[str, dict[str, Any]] = {}

        fts_q = _fts_query(tokens)
        if fts_q != '""':
            try:
                rows = self.session.execute(
                    text(
                        """
                        SELECT document_id, original_text, source, author,
                               bm25(documents_fts) AS rank
                        FROM documents_fts
                        WHERE documents_fts MATCH :q
                        ORDER BY rank
                        LIMIT :k
                        """
                    ),
                    {"q": fts_q, "k": top_k},
                ).mappings().all()
                for r in rows:
                    rank = float(r["rank"])
                    by_id[r["document_id"]] = self._hit(
                        r,
                        score=1.0 / (1.0 + abs(rank)),
                        lexical_rank=rank,
                    )
            except Exception:
                pass

        # Substring match is required for Persian: FTS5 default tokenizer
        # often fails to index Arabic-script tokens usefully.
        if tokens:
            like_clauses = " OR ".join(
                f"original_text LIKE :t{i} OR source LIKE :t{i} OR COALESCE(author,'') LIKE :t{i}"
                for i in range(len(tokens))
            )
            params: dict[str, Any] = {f"t{i}": f"%{tok}%" for i, tok in enumerate(tokens)}
            params["k"] = top_k
            rows = self.session.execute(
                text(
                    f"""
                    SELECT document_id, original_text, source, author
                    FROM documents
                    WHERE {like_clauses}
                    LIMIT :k
                    """
                ),
                params,
            ).mappings().all()
            for r in rows:
                overlap = token_overlap(query, r["original_text"] or "")
                score = min(1.0, 0.35 + 0.2 * overlap)
                existing = by_id.get(r["document_id"])
                if existing is None or score > existing["score"]:
                    by_id[r["document_id"]] = self._hit(r, score=score, lexical_rank=-overlap)

        ranked = sorted(by_id.values(), key=lambda x: x["score"], reverse=True)
        return ranked[:top_k]

    @staticmethod
    def _hit(row: Any, *, score: float, lexical_rank: float) -> dict[str, Any]:
        return {
            "document_id": row["document_id"],
            "score": score,
            "lexical_rank": lexical_rank,
            "snippet": (row["original_text"] or "")[:280],
            "source": row["source"],
            "author": row["author"],
            "channel": "lexical",
        }
