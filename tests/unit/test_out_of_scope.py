"""Tests for the standalone out-of-scope detection experiment (Improvement 4).

Covers the authored synthetic dataset, the family-level split guards, the metric
helpers, the validation-only threshold policy, both detectors (with a stub FAISS
index so the suite needs neither the real index nor a model download) and the
report renderer.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "out_of_scope"))

import oos_data  # noqa: E402
import oos_detectors as det  # noqa: E402
import oos_metrics as om  # noqa: E402
import oos_report as report  # noqa: E402


# --------------------------------------------------------------------------- data


def test_synthetic_dataset_structure_and_counts():
    df = oos_data.build_synthetic_frame()
    assert len(df) == 180
    assert df["family_id"].nunique() == 60
    assert df["category"].nunique() == 12
    assert set(df["register"]) == {"en", "hi", "hinglish"}
    assert set(df["label"]) == {"out_of_scope"}
    assert df["synthetic"].all()
    assert df["text"].is_unique
    assert df["query_id"].is_unique
    assert (df.groupby("family_id")["register"].nunique() == 3).all()


def test_synthetic_frames_are_deterministic():
    pd.testing.assert_frame_equal(oos_data.build_synthetic_frame(),
                                  oos_data.build_synthetic_frame())


def test_split_uses_five_families_per_category_three_one_one():
    split = oos_data.split_synthetic(oos_data.build_synthetic_frame())
    per_cat = split.drop_duplicates("family_id").groupby(["category", "split"]).size().unstack()
    for _, row in per_cat.iterrows():
        assert row["train"] == 3
        assert row["val"] == 1
        assert row["test"] == 1


def test_split_counts_and_family_disjointness():
    split = oos_data.split_synthetic(oos_data.build_synthetic_frame())
    fam = split.drop_duplicates("family_id")
    assert fam["split"].value_counts().to_dict() == {"train": 36, "val": 12, "test": 12}
    assert (split.groupby("family_id")["split"].nunique() == 1).all()
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        assert not (set(split.loc[split["split"] == left, "family_id"])
                    & set(split.loc[split["split"] == right, "family_id"]))
    # Every register of a family lands in the same split.
    assert (split.groupby(["family_id", "split"])["register"].nunique() == 3).all()


def test_split_refuses_incomplete_category():
    frame = oos_data.build_synthetic_frame()
    dropped = frame[frame["family_id"] != "sports-13"]
    with pytest.raises(ValueError, match="expected 5 families"):
        oos_data.split_synthetic(dropped)


def test_ambiguous_and_mixed_frames_are_flagged_synthetic():
    amb = oos_data.build_ambiguous_frame()
    mixed = oos_data.build_mixed_frame()
    assert len(amb) == len(oos_data.AMBIGUOUS)
    assert len(mixed) == len(oos_data.MIXED_INTENT)
    assert list(amb["label"].unique()) == ["ambiguous"]
    assert list(mixed["label"].unique()) == ["mixed_intent"]
    for frame in (amb, mixed):
        assert frame["synthetic"].all()
        assert frame["text"].is_unique
        assert set(frame["register"]) <= {"en", "hi", "hinglish"}


def test_detect_register_matches_project_heuristics():
    assert oos_data.detect_register("जापान की राजधानी क्या है?") == "hi"
    assert oos_data.detect_register("mandi me gehu ka bhav kya hai") == "hinglish"
    assert oos_data.detect_register("who won the cricket match yesterday") == "en"
    assert oos_data.detect_register("") == "en"


# ------------------------------------------------------------------------ metrics


def test_binary_metrics_on_known_arrays():
    y = np.array([0, 0, 1, 1, 0, 0, 1, 1])
    rejected = np.array([False, False, True, True, True, True, False, False])
    m = om.binary_metrics(y, rejected)
    assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (2, 2, 2, 2)
    assert m["precision"] == pytest.approx(0.5)
    assert m["recall"] == pytest.approx(0.5)
    assert m["f1"] == pytest.approx(0.5)
    assert m["false_acceptance_rate"] == pytest.approx(0.5)
    assert m["false_rejection_rate"] == pytest.approx(0.5)
    assert m["balanced_accuracy"] == pytest.approx(0.5)
    assert m["n_in_scope"] == 4 and m["n_out_of_scope"] == 4


def test_binary_metrics_reports_none_for_absent_class():
    m = om.binary_metrics(np.zeros(3, dtype=int), np.zeros(3, dtype=bool))
    assert m["false_acceptance_rate"] is None
    assert m["false_rejection_rate"] == 0.0
    assert m["recall"] == 0.0


def test_binary_metrics_rejects_length_mismatch():
    with pytest.raises(ValueError):
        om.binary_metrics(np.array([0, 1]), np.array([True]))


def test_f1_ci_is_bounded_and_ordered():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, size=200)
    rejected = np.where(rng.random(200) < 0.8, y.astype(bool), ~y.astype(bool))
    lo, hi = om.f1_ci(y, rejected, n_boot=200, seed=1)
    m = om.binary_metrics(y, rejected)
    assert 0.0 <= lo <= hi <= 1.0
    assert lo <= m["f1"] <= hi


def test_threshold_policy_maximises_recall_within_frr_budget():
    in_scores = np.array([0.90, 0.88, 0.85, 0.80, 0.78, 0.74, 0.70, 0.66])
    out_scores = np.array([0.72, 0.60, 0.55, 0.40, 0.30, 0.20])
    policy = om.select_threshold(in_scores, out_scores, target_frr=0.15)
    assert policy["threshold"] == pytest.approx(0.70)
    assert policy["val_false_rejection_rate"] == pytest.approx(0.125)
    assert policy["val_recall"] == pytest.approx(5 / 6)
    # No strictly higher candidate may also satisfy the budget.
    for t in np.unique(np.concatenate([in_scores, out_scores])):
        if t > policy["threshold"] + 1e-9:
            assert np.mean(in_scores < t) > 0.15


def test_threshold_policy_rejects_unusable_input():
    with pytest.raises(ValueError):
        om.select_threshold(np.array([]), np.array([0.1]), target_frr=0.02)
    with pytest.raises(ValueError):
        om.select_threshold(np.array([0.9]), np.array([]), target_frr=0.02)


def test_threshold_policy_refuses_impossible_budget():
    with pytest.raises(ValueError):
        om.select_threshold(np.array([0.5, 0.6]), np.array([0.5]), target_frr=-0.1)


def test_threshold_sweep_trades_recall_for_false_rejections():
    y = np.array([0, 0, 0, 0, 1, 1, 1])
    scores = np.array([0.9, 0.8, 0.7, 0.6, 0.65, 0.5, 0.4])
    rows = om.threshold_sweep(y, scores, [0.5, 0.6, 0.7])
    assert [r["threshold"] for r in rows] == [0.5, 0.6, 0.7]
    assert rows[0]["false_rejection_rate"] <= rows[-1]["false_rejection_rate"]
    assert rows[0]["recall"] <= rows[-1]["recall"]
    assert rows[2]["false_rejection_rate"] == pytest.approx(0.25)


# ---------------------------------------------------------------------- detectors


def test_agricultural_evidence_finds_glossary_terms():
    hits = det.agricultural_evidence("Which pesticide should I use for tomato?")
    assert "pesticide" in hits and "tomato" in hits
    assert "urea" in det.agricultural_evidence("how much urea for paddy")
    assert det.agricultural_evidence("who won the cricket match") == []
    assert det.agricultural_evidence("") == []
    assert det.agricultural_evidence(None) == []


def test_agricultural_evidence_is_capped():
    text = "wheat paddy cotton tomato potato onion aphid rust blight urea zinc scheme mandi"
    assert len(det.agricultural_evidence(text)) <= 8


def test_decide_three_way_bands():
    top = det.decide(0.90, 0.50, 0.05, "pest control in wheat")
    assert (top.decision, top.accepted) == ("in_scope", True)
    inside = det.decide(0.52, 0.50, 0.05, "pest control in wheat")
    assert (inside.decision, inside.accepted) == ("ambiguous", True)
    just_below = det.decide(0.48, 0.50, 0.05, "who won the match")
    assert (just_below.decision, just_below.accepted) == ("ambiguous", False)
    far = det.decide(0.10, 0.50, 0.05, "who won the cricket match")
    assert (far.decision, far.accepted, far.evidence) == ("out_of_scope", False, [])


def test_decide_escalates_low_score_with_evidence_instead_of_rejecting():
    d = det.decide(0.10, 0.50, 0.05, "pesticide dose for tomato")
    assert d.decision == "ambiguous"
    assert d.accepted is False
    assert d.evidence
    assert "agricultural terms" in d.reason


class _StubIndex:
    """Minimal FAISS-like index returning descending inner products."""

    def __init__(self, vectors):
        self.vectors = np.asarray(vectors, dtype="float32")

    def search(self, queries, k):
        sims = self.vectors @ np.asarray(queries, dtype="float32").T  # (refs, queries)
        top = np.sort(sims, axis=0)[-k:][::-1].T.copy()
        return top.astype("float32"), np.zeros(top.shape, dtype="int64")


def test_knn_rejector_scores_mean_of_top_k():
    index = _StubIndex([[1.0, 0.0], [0.9, 0.1], [0.8, 0.2], [0.0, 1.0]])
    rejector = det.KNNRejector(index=index, k=2)
    scores = rejector.scores(np.array([[1.0, 0.0]], dtype="float32"))
    assert scores.shape == (1,)
    assert scores[0] == pytest.approx((1.0 + np.dot([1, 0], [0.9, 0.1])) / 2, abs=1e-5)


def test_embedding_lr_separates_toy_clusters():
    rng = np.random.default_rng(3)
    X_in = rng.normal(1.0, 0.1, (40, 4))
    X_out = rng.normal(-1.0, 0.1, (40, 4))
    detector = det.EmbeddingLRDetector(C=1.0, seed=0).fit(X_in, X_out)
    p_in = detector.scores(X_in)
    p_out = detector.scores(X_out)
    assert p_in.min() >= 0.0 and p_in.max() <= 1.0
    assert p_in.mean() > 0.9 and p_out.mean() < 0.1
    assert (p_in > 0.5).all() and (p_out < 0.5).all()


def test_embedding_lr_requires_two_classes():
    with pytest.raises(ValueError):
        det.EmbeddingLRDetector().fit(np.zeros((5, 3)), np.zeros((0, 3)))


# ------------------------------------------------------------------------ report


def _fake_metrics():
    def bench():
        return {
            "n": 174, "n_in_scope": 144, "n_out_of_scope": 30,
            "tp": 27, "fp": 5, "fn": 3, "tn": 139,
            "confusion": {"tp": 27, "fp": 5, "fn": 3, "tn": 139},
            "precision": 0.8438, "recall": 0.9, "f1": 0.8709,
            "f1_ci95": [0.7321, 0.9556], "balanced_accuracy": 0.87,
            "auroc": 0.91, "false_acceptance_rate": 0.1,
            "false_rejection_rate": 0.0347,
            "mean_score_in_scope": 0.64, "mean_score_out_of_scope": 0.31,
        }

    approach = {"threshold": 0.42, "val_false_rejection_rate": 0.02,
                "val_recall": 0.8, "policy": "max recall subject to FRR <= 0.02",
                "target_frr": 0.02, "n_val_in_scope": 5000, "n_val_out_of_scope": 12}
    return {
        "generated_utc": "2026-11-04T00:00:00+00:00",
        "seeds": {"dataset": 42, "bootstrap": 0},
        "protocol": {"benchmark": "144 + 30"},
        "data": {
            "corpus_rows": 1000,
            "corpus_split_counts": {"train": 700, "val": 150, "test": 150},
            "corpus_registers_test_sample": {"en": 150},
            "synthetic": {
                "n_queries": 180, "n_families": 60, "n_categories": 12,
                "per_split_queries": {"train": 108, "val": 36, "test": 36},
                "per_split_families": {"train": 36, "val": 12, "test": 12},
                "per_register": {"en": 60, "hi": 60, "hinglish": 60},
                "synthetic": True,
            },
            "ambiguous_queries": 16, "mixed_intent_queries": 16,
            "split_checks": {"synthetic_families_spanning_splits": 0,
                             "corpus_groups_spanning_splits": 0},
        },
        "thresholds": {"knn_distance": dict(approach), "embedding_lr": dict(approach)},
        "benchmark": {"knn_distance": bench(), "embedding_lr": bench()},
        "full_in_scope_test": {
            "knn_distance": {"n": 5000, "n_rejected": 150, "false_rejection_rate": 0.03},
            "embedding_lr": {"n": 5000, "n_rejected": 100, "false_rejection_rate": 0.02},
        },
        "by_register_out_of_scope": {
            r: {"n": 10, "knn_distance": {"n_rejected": 9, "rejection_rate": 0.9},
                "embedding_lr": {"n_rejected": 8, "rejection_rate": 0.8}}
            for r in ("en", "hi", "hinglish")
        },
        "ambiguous": {a: {"n": 16, "decision_counts": {"ambiguous": 10, "in_scope": 6},
                          "n_rejected": 4, "rejection_rate": 0.25}
                      for a in ("knn_distance", "embedding_lr")},
        "mixed_intent": {a: {"n": 16, "decision_counts": {"in_scope": 14, "ambiguous": 2},
                             "n_rejected": 0, "rejection_rate": 0.0}
                         for a in ("knn_distance", "embedding_lr")},
        "evidence": {
            "corpus_val_sample": {"n": 5000, "with_evidence": 4000},
            "synthetic_test_out_of_scope": {"n": 36, "with_evidence": 2},
            "ambiguous": {"n": 16, "with_evidence": 12},
            "mixed_intent": {"n": 16, "with_evidence": 16},
        },
        "threshold_sensitivity": {
            a: [{"threshold": 0.4, "false_acceptance_rate": 0.1,
                 "false_rejection_rate": 0.03, "f1": 0.87}]
            for a in ("knn_distance", "embedding_lr")
        },
        "limitations": ["synthetic negatives only"],
    }


def test_report_renders_required_sections_and_numbers():
    text = report.render(_fake_metrics())
    for heading in ("# Improvement 4", "## 2. Data and split integrity",
                    "## 3. Threshold selection", "## 4. Benchmark results",
                    "## 5. Full agricultural test set", "## 6. By register",
                    "## 7. Ambiguous and mixed-intent", "## 8. Threshold sensitivity",
                    "## 9. Interpretation", "## 10. Limitations", "## 11. Reproduce"):
        assert heading in text
    assert "0.8438" in text
    assert "0.8709" in text
    assert "[0.7321, 0.9556]" in text
    assert "synthetic" in text.lower()


def test_report_refuses_incomplete_metrics():
    metrics = _fake_metrics()
    del metrics["threshold_sensitivity"]
    with pytest.raises(KeyError, match="threshold_sensitivity"):
        report.render(metrics)


def test_report_states_no_multilingual_in_scope_claim():
    text = report.render(_fake_metrics())
    assert "No in-scope Hindi or Hinglish" in text or "no multilingual" in text.lower()
