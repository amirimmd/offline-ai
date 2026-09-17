"""Lexical (FTS5) retrieval."""

from __future__ import annotations

import re
from typing import Any

from sqlalchemy import text
from sqlalchemy.orm import Session


def _sanitize_fts_query(query: str) -> str:
    """Build a safe FTS5 OR query from tokens."""
    tokens = re.findall(r"[\w\u0600-\u06FF]+", query, flags=re.UNICODE)
    tokens = [t for t in tokens if len(t) > 1]
    if not tokens:
        return '""'
    # Quote each token for FTS5
    return " OR ".join(f'"{t}"' for t in tokens)


class LexicalRetriever:
    def __init__(self, session: Session) -> None:
        self.session = session

    def search(self, query: str, top_k: int = 20) -> list[dict[str, Any]]:
        fts_q = _sanitize_fts_query(query)
        if fts_q == '""':
            return []
        sql = text(
            """
            SELECT document_id, original_text, source, author,
                   bm25(documents_fts) AS rank
            FROM documents_fts
            WHERE documents_fts MATCH :q
            ORDER BY rank
            LIMIT :k
            """
        )
        rows = self.session.execute(sql, {"q": fts_q, "k": top_k}).mappings().all()
        results = []
        for r in rows:
            # bm25: lower is better; convert to 0..1-ish score
            rank = float(r["rank"])
            score = 1.0 / (1.0 + abs(rank))
            results.append(
                {
                    "document_id": r["document_id"],
                    "score": score,
                    "lexical_rank": rank,
                    "snippet": (r["original_text"] or "")[:280],
                    "source": r["source"],
                    "author": r["author"],
                    "channel": "lexical",
                }
            )
        return results
