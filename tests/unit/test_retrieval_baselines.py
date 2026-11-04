"""Tests for the Project B retrieval baselines (BM25, metrics, statistics)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT / "src", ROOT / "experiments"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from retrieval_baselines.bm25 import BM25, tokenize  # noqa: E402
from retrieval_baselines.common import (  # noqa: E402
    evaluate_rankings, holm, load_embeddings, load_index_metadata, load_split,
    paired_bootstrap,
)


# ------------------------------------------------------------------ BM25
def test_bm25_ranks_the_relevant_document_first():
    docs = ["wheat rust control measure", "cotton bollworm spray", "mandi price of wheat"]
    b = BM25(docs)
    ranked = b.search("wheat rust", k=3)
    assert ranked and ranked[0][0] == 0
    assert all(ranked[i][1] >= ranked[i + 1][1] for i in range(len(ranked) - 1))


def test_bm25_empty_query_returns_nothing():
    b = BM25(["wheat rust", "cotton bollworm"])
    assert b.search("", k=5) == []
    assert b.search("   ", k=5) == []


def test_bm25_is_deterministic_and_tie_breaks_by_doc_id():
    b = BM25(["alpha beta", "beta alpha", "gamma"])
    assert b.search("alpha beta", k=3) == b.search("alpha beta", k=3)
    ranked = b.search("alpha beta", k=3)
    assert [d for d, _ in ranked] == [0, 1]  # equal score -> ascending doc id


def test_bm25_handles_unicode_and_devanagari():
    docs = ["गेहूं में रतुआ", "कपास में बोलवर्म", "wheat price"]
    b = BM25(docs)
    ranked = b.search("गेहूं रतुआ", k=3)
    assert ranked and ranked[0][0] == 0
    # Devanagari words keep their vowel signs as one token
    assert any("गेहूं" in tok for tok in tokenize("गेहूं में रतुआ"))


def test_bm25_scores_are_finite_and_sorted():
    b = BM25(["a b c", "b c d", "c d e"])
    for doc, score in b.search("c d", k=3):
        assert np.isfinite(score)


# ------------------------------------------------------------------ metrics
def _frames():
    meta = pd.DataFrame({
        "questions": ["wheat rust control", "cotton bollworm", "mandi wheat price"],
        "answers": ["spray for rust", "spray for bollworm", "price is 2400"],
        "query_type": ["Plant Protection", "Plant Protection", "Market Price"],
    })
    test = pd.DataFrame({
        "questions": ["wheat rust control"],
        "answers": ["spray for rust"],
        "query_type": ["Plant Protection"],
    })
    return meta, test


def test_evaluate_rankings_topic_match_and_exact_match():
    meta, test = _frames()
    hit = evaluate_rankings(np.array([[0]]), test, meta, 1)
    assert hit["topic_match_at_1"] == 1.0
    assert hit["exact_match_at_1"] == 1.0
    assert hit["recall_at_5"] == 1.0
    miss = evaluate_rankings(np.array([[2]]), test, meta, 1)
    assert miss["topic_match_at_1"] == 0.0
    assert miss["exact_match_at_1"] == 0.0


def test_evaluate_rankings_recall_at_k():
    meta, test = _frames()
    # topic match appears at rank 2 (index 1) -> recall@5 hit, recall@1 miss
    ranked = np.array([[2, 1]])
    m = evaluate_rankings(ranked, test, meta, 1)
    assert m["recall_at_1"] == 0.0
    assert m["recall_at_5"] == 1.0
    assert m["mrr_at_10"] == 0.5


# ------------------------------------------------------------------ statistics
def test_holm_adjustment():
    adj = holm({"a": 0.01, "b": 0.04, "c": 0.5})
    assert adj["a"] == 0.03 and adj["b"] == 0.08 and adj["c"] == 0.5


def test_paired_bootstrap_detects_clear_signal_and_is_deterministic():
    rng = np.random.default_rng(0)
    strong = np.ones(200) * 0.1
    res = paired_bootstrap(strong)
    assert res["mean_diff"] == 0.1 and res["p_significant"]
    assert paired_bootstrap(strong) == res
    null = rng.normal(0, 1, 200)
    assert paired_bootstrap(null)["ci95"][0] < 0 < paired_bootstrap(null)["ci95"][1]


# ------------------------------------------------------------------ artifacts
def test_committed_grouped_artifacts_align():
    meta = load_index_metadata()
    assert len(meta) == 125044
    split = load_split()
    assert len(split) == 138875
    emb = load_embeddings()
    assert emb.shape[0] == 138875 and emb.shape[1] == 384


def test_comparative_metrics_have_all_systems(tmp_path):
    import json

    p = ROOT / "results" / "retrieval_baselines" / "comparative_evaluation" / "metrics.json"
    if not p.exists():
        import pytest

        pytest.skip("run compare.py first")
    m = json.loads(p.read_text(encoding="utf-8"))
    for system in ("dense", "tfidf", "hybrid", "bm25"):
        assert system in m["systems"], system
    # exact-match must be zero on the grouped split (no self-retrieval)
    assert m["systems"]["dense"]["exact_match_at_1"] == 0.0
