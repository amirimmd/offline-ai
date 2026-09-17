"""Training / adapter evaluation helpers."""

from __future__ import annotations

from typing import Any


def evaluate_retrieval(predictions: list[list[str]], relevants: list[set[str]]) -> dict[str, float]:
    """Compute Precision@k, Recall@k, MRR for retrieval lists."""
    assert len(predictions) == len(relevants)
    precs = []
    recalls = []
    mrrs = []
    for preds, rel in zip(predictions, relevants, strict=True):
        if not rel:
            continue
        hits = [p for p in preds if p in rel]
        k = max(len(preds), 1)
        precs.append(len(hits) / k)
        recalls.append(len(hits) / len(rel))
        rr = 0.0
        for i, p in enumerate(preds, start=1):
            if p in rel:
                rr = 1.0 / i
                break
        mrrs.append(rr)
    n = max(len(precs), 1)
    return {
        "precision": sum(precs) / n,
        "recall": sum(recalls) / n,
        "mrr": sum(mrrs) / n,
    }


def ndcg_at_k(relevances: list[float], k: int = 10) -> float:
    import math

    rel = relevances[:k]
    dcg = sum((2**r - 1) / math.log2(i + 2) for i, r in enumerate(rel))
    ideal = sorted(relevances, reverse=True)[:k]
    idcg = sum((2**r - 1) / math.log2(i + 2) for i, r in enumerate(ideal)) or 1.0
    return dcg / idcg
