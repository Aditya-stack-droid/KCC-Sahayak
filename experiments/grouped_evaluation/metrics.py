"""Metrics for grouped evaluation: bootstrap CIs, ranking metrics, token F1."""
from __future__ import annotations

import re
from collections import Counter

import numpy as np


def bootstrap_ci(values, n_boot: int = 2000, seed: int = 0, alpha: float = 0.05):
    """Percentile bootstrap confidence interval for the mean."""
    x = np.asarray(values, dtype=float)
    if x.size == 0:
        return [None, None]
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    means = x[idx].mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return [round(float(lo), 4), round(float(hi), 4)]


def token_f1(a, b) -> float:
    ta = re.findall(r"\w+", str(a).lower())
    tb = re.findall(r"\w+", str(b).lower())
    if not ta or not tb:
        return 0.0
    common = sum((Counter(ta) & Counter(tb)).values())
    if common == 0:
        return 0.0
    p, r = common / len(ta), common / len(tb)
    return 2 * p * r / (p + r)


def recall_at_k(relevant: list[bool], k: int) -> float:
    return 1.0 if any(relevant[:k]) else 0.0


def reciprocal_rank(relevant: list[bool]) -> float:
    for rank, hit in enumerate(relevant, start=1):
        if hit:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(relevant: list[bool], k: int) -> float:
    """Binary-relevance nDCG; the ideal ranking puts one relevant item first."""
    dcg = 0.0
    for rank, hit in enumerate(relevant[:k], start=1):
        if hit:
            dcg = 1.0 / np.log2(rank + 1)
            break
    return float(dcg)  # ideal DCG = 1.0


def novelty_bucket(max_similarity: float, near_dup: float = 0.90,
                   novel: float = 0.80) -> str:
    if max_similarity >= near_dup:
        return "near_duplicate"
    if max_similarity >= novel:
        return "similar"
    return "novel"
