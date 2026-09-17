"""Topic extraction (keyword / simple TF heuristics)."""

from __future__ import annotations

import re
from collections import Counter

STOP = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "with", "by", "is", "was",
    "are", "were", "be", "as", "at", "from", "that", "this", "it", "its",
}


class TopicExtractor:
    def extract(self, text: str, top_k: int = 5) -> list[tuple[str, float]]:
        tokens = [t.lower() for t in re.findall(r"[A-Za-z\u0600-\u06FF]{3,}", text)]
        tokens = [t for t in tokens if t not in STOP]
        counts = Counter(tokens)
        total = sum(counts.values()) or 1
        return [(w, c / total) for w, c in counts.most_common(top_k)]
