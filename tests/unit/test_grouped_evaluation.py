"""Unit tests for the grouping and metric primitives (no files written)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "experiments" / "grouped_evaluation"
sys.path.insert(0, str(EXP))

from grouping import (  # noqa: E402
    UnionFind, apply_edges, canonical_group_ids, group_exact, group_leakage,
    group_sizes, normalize_question,
)
from metrics import (  # noqa: E402
    bootstrap_ci, ndcg_at_k, novelty_bucket, recall_at_k, reciprocal_rank, token_f1,
)


def test_union_find_is_transitive():
    uf = UnionFind(5)
    apply_edges(uf, [(0, 1), (1, 2), (3, 4)])
    ids = canonical_group_ids(uf)
    assert ids.tolist() == [0, 0, 0, 3, 3]
    assert uf.n_components == 2


def test_union_find_counts_only_successful_merges():
    uf = UnionFind(3)
    assert apply_edges(uf, [(0, 1), (1, 0)]) == 1


def test_normalise_question_collapses_case_and_whitespace():
    assert normalize_question("  Pest   Control ") == "pest control"


def test_exact_grouping_merges_identical_questions():
    uf = group_exact(["Pest control", "pest   control", "market price"])
    ids = canonical_group_ids(uf)
    assert ids[0] == ids[1] != ids[2]
    assert group_sizes(ids).tolist() == [2, 1]


def test_group_leakage_counts_shared_groups():
    groups = np.array([1, 1, 2, 3, 3])
    split = np.array(["train", "test", "train", "test", "test"], dtype=object)
    out = group_leakage(groups, split)
    assert out["n_test"] == 3
    assert out["n_test_group_leaking"] == 1
    assert out["share_test_group_leaking"] == round(1 / 3, 6)


def test_bootstrap_ci_bounds_and_constant_input():
    lo, hi = bootstrap_ci([1, 1, 1, 1], n_boot=200)
    assert lo == hi == 1.0
    lo, hi = bootstrap_ci([0, 1] * 50, n_boot=300, seed=1)
    assert 0.0 <= lo <= hi <= 1.0


def test_token_f1_edge_cases():
    assert token_f1("spray urea", "spray urea") == 1.0
    assert token_f1("spray urea", "market price") == 0.0
    assert token_f1("", "anything") == 0.0


def test_ranking_metrics():
    relevant = [False, True, False]
    assert recall_at_k(relevant, 1) == 0.0
    assert recall_at_k(relevant, 2) == 1.0
    assert reciprocal_rank(relevant) == 0.5
    assert ndcg_at_k(relevant, 5) == pytest.approx(1 / np.log2(3))
    assert reciprocal_rank([False, False]) == 0.0


def test_novelty_bucket_boundaries():
    assert novelty_bucket(0.95) == "near_duplicate"
    assert novelty_bucket(0.90) == "near_duplicate"
    assert novelty_bucket(0.85) == "similar"
    assert novelty_bucket(0.80) == "similar"
    assert novelty_bucket(0.79) == "novel"


def test_stratified_group_split_has_no_group_overlap():
    from sklearn.model_selection import StratifiedGroupKFold

    rng = np.random.default_rng(0)
    n = 300
    groups = np.repeat(np.arange(60), 5)
    labels = rng.choice(["A", "B", "C"], size=n)
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0)
    for tr, te in sgkf.split(np.zeros(n), labels, groups):
        assert not set(groups[tr].tolist()) & set(groups[te].tolist())
        assert len(tr) + len(te) == n


def test_split_statistics_frame_shape():
    df = pd.DataFrame({"label": ["A", "A", "B", "B"]})
    assert df["label"].value_counts().sum() == 4
