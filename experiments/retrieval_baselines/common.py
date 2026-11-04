"""Shared data loading, metrics and statistics for the retrieval-baseline study.

Reuses the committed Batch A grouped artifacts (no index rebuild, no
re-embedding). Relevance is the established **weak-label topic match**
(retrieved ``query_type`` == test ``query_type``) — a topical proxy, not
agricultural correctness.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / "experiments" / "grouped_evaluation" / "artifacts"
PROCESSED = ROOT / "data" / "processed"
INDEX_DIR = ART

SEED = 42


def load_corpus() -> pd.DataFrame:
    return pd.read_csv(PROCESSED / "full_labeled.csv")


def load_split() -> pd.Series:
    split = pd.read_csv(ART / "corpus_split.csv")
    corpus = load_corpus()
    if len(split) != len(corpus) or not np.array_equal(split["qid"].to_numpy(), corpus["qid"].to_numpy()):
        raise SystemExit("corpus_split.csv does not align with full_labeled.csv")
    return split["split"].reset_index(drop=True)


def load_index_metadata() -> pd.DataFrame:
    meta = pd.read_csv(ART / "grouped_retrieval_metadata.csv")
    return meta[["qid", "questions", "answers", "query_type"]].reset_index(drop=True)


def load_embeddings() -> np.ndarray:
    return np.load(ART / "corpus_embeddings.npy")


def test_frame():
    """Return the grouped test rows (aligned to the corpus) and their positions."""
    corpus = load_corpus()
    split = load_split()
    test_idx = np.flatnonzero(split.to_numpy() == "test")
    return corpus.iloc[test_idx].reset_index(drop=True), test_idx


def sample_positions(n: int, seed: int = SEED) -> np.ndarray:
    test_df, _ = test_frame()
    rng = np.random.default_rng(seed)
    pos = rng.choice(len(test_df), size=min(n, len(test_df)), replace=False)
    pos.sort()
    return pos


_NORM = re.compile(r"\s+")


def norm_text(s) -> str:
    return _NORM.sub(" ", str(s).lower()).strip()


def token_f1(a: str, b: str) -> float:
    ta, tb = norm_text(a).split(), norm_text(b).split()
    if not ta or not tb:
        return 0.0
    from collections import Counter

    common = Counter(ta) & Counter(tb)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(ta)
    recall = overlap / len(tb)
    return 2 * precision * recall / (precision + recall)


def _recall_at(ranked_labels, k: int) -> float:
    return 1.0 if any(ranked_labels[:k]) else 0.0


def _rr(ranked_labels, k: int) -> float:
    for i, hit in enumerate(ranked_labels[:k]):
        if hit:
            return 1.0 / (i + 1)
    return 0.0


def _ndcg(ranked_labels, k: int) -> float:
    gains = [1.0 if h else 0.0 for h in ranked_labels[:k]]
    dcg = sum(g / np.log2(i + 2) for i, g in enumerate(gains))
    ideal = sum(1.0 / np.log2(i + 2) for i in range(min(k, max(1, sum(ranked_labels[:k])))))
    return dcg / ideal if ideal else 0.0


def evaluate_rankings(ranked_ids: np.ndarray, test_df: pd.DataFrame, meta: pd.DataFrame,
                      n_scored: int) -> dict:
    """ranked_ids: [n, k] positional ids into ``meta`` (-1 allowed)."""
    ranked_ids = np.asarray(ranked_ids, dtype=np.int64)
    qtypes = meta["query_type"].to_numpy()
    gold_q = test_df["questions"].map(norm_text).to_numpy()
    gold_a = test_df["answers"].astype(str).to_numpy()
    gold_type = test_df["query_type"].to_numpy()

    labels = []
    top1_ids = []
    f1 = []
    exact = []
    k_max = ranked_ids.shape[1]
    for i in range(n_scored):
        row = ranked_ids[i]
        safe = np.clip(row, 0, len(meta) - 1)
        cats = qtypes[safe]
        hit = (cats == gold_type[i])
        labels.append(hit.tolist())
        top1_ids.append(int(row[0]) if row[0] >= 0 else -1)
        if row[0] >= 0:
            f1.append(token_f1(meta["answers"].iloc[int(row[0])], gold_a[i]))
            exact.append(1.0 if norm_text(meta["questions"].iloc[int(row[0])]) == gold_q[i] else 0.0)
        else:
            f1.append(0.0)
            exact.append(0.0)
    labels = np.asarray(labels, dtype=bool)
    return {
        "n": int(n_scored),
        "recall_at_1": round(float(np.mean([_recall_at(r, 1) for r in labels])), 4),
        "recall_at_3": round(float(np.mean([_recall_at(r, 3) for r in labels])), 4),
        "recall_at_5": round(float(np.mean([_recall_at(r, 5) for r in labels])), 4),
        "recall_at_10": round(float(np.mean([_recall_at(r, 10) for r in labels])), 4),
        "mrr_at_10": round(float(np.mean([_rr(r, 10) for r in labels])), 4),
        "ndcg_at_5": round(float(np.mean([_ndcg(r, 5) for r in labels])), 4),
        "topic_match_at_1": round(float(labels[:, 0].mean()), 4),
        "exact_match_at_1": round(float(np.mean(exact)), 4),
        "answer_token_f1_at_1": round(float(np.mean(f1)), 4),
        "_per_query": {"topic_hit_at_1": labels[:, 0].astype(float).tolist(),
                       "recall_at_5": np.array([_recall_at(r, 5) for r in labels]).tolist(),
                       "mrr": np.array([_rr(r, 10) for r in labels]).tolist()},
    }


def paired_bootstrap(diff: np.ndarray, n_boot: int = 2000, seed: int = SEED) -> dict:
    diff = np.asarray(diff, dtype=float)
    rng = np.random.default_rng(seed)
    n = len(diff)
    means = np.empty(n_boot)
    for b in range(n_boot):
        means[b] = diff[rng.integers(0, n, n)].mean()
    lo, hi = np.percentile(means, [2.5, 97.5])
    # two-sided bootstrap p-value: twice the smaller tail mass at zero
    p_two = min(1.0, 2.0 * min(float(np.mean(means <= 0.0)), float(np.mean(means >= 0.0))))
    return {"mean_diff": round(float(diff.mean()), 4),
            "ci95": [round(float(lo), 4), round(float(hi), 4)],
            "p_value": round(p_two, 6),
            "p_significant": bool(p_two < 0.05), "n": int(n)}


def holm(pvalues: dict[str, float]) -> dict[str, float]:
    """Holm–Bonferroni adjusted p-values."""
    items = sorted(pvalues.items(), key=lambda kv: kv[1])
    m = len(items)
    adjusted, running = {}, 0.0
    for i, (name, p) in enumerate(items):
        val = min(1.0, (m - i) * p)
        running = max(running, val)
        adjusted[name] = round(running, 6)
    return adjusted


def environment() -> dict:
    import platform

    versions = {}
    for name in ("numpy", "pandas", "sentence_transformers", "torch", "faiss", "sklearn"):
        try:
            mod = __import__(name)
            versions[name] = getattr(mod, "__version__", "unknown")
        except Exception:
            versions[name] = "not importable"
    return {"python": sys.version.split()[0], "platform": platform.platform(), "versions": versions}


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")
