"""Topic extraction (keyword / simple TF heuristics)."""

from __future__ import annotations

from collections import Counter

from offline_ai.utils.persian import STOPWORDS, tokenize


class TopicExtractor:
    def extract(self, text: str, top_k: int = 5) -> list[tuple[str, float]]:
        tokens = [t for t in tokenize(text) if t not in STOPWORDS and len(t) >= 3]
        counts = Counter(tokens)
        total = sum(counts.values()) or 1
        return [(w, c / total) for w, c in counts.most_common(top_k)]
