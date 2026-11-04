"""Unit tests for Improvement 3 (multi-reference retrieval evaluation).

These tests use hand-built fixtures. They verify the definitions (pool
construction, own-answer exclusion, chance control, paired statistics) rather
than any corpus result, and they never load the retrieval index or encode text.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "multi_reference"))

import mref_metrics as mref  # noqa: E402
import mref_ranking as ranking  # noqa: E402
import mref_refdata as refdata  # noqa: E402
import mref_report as report  # noqa: E402


def tiny_frame() -> pd.DataFrame:
    rows = [
        dict(qid=1, questions="q a", answers="spray rogor 2 ml per litre of water",
             query_type="Plant Protection", group_id=0, split="test"),
        dict(qid=2, questions="q a", answers="apply imidacloprid 0.5 ml per litre water",
             query_type="Plant Protection", group_id=0, split="test"),
        dict(qid=3, questions="q a", answers="explained in details",
             query_type="Plant Protection", group_id=0, split="test"),
        dict(qid=4, questions="q b", answers="use urea 50 kg per acre",
             query_type="Nutrient Management", group_id=1, split="test"),
        dict(qid=5, questions="q c", answers="spray rogor 2 ml per litre of water",
             query_type="Plant Protection", group_id=2, split="train"),
    ]
    return refdata.prepare_corpus_frame(pd.DataFrame(rows))


# --------------------------------------------------------------------- pools

def test_normalise_answer_lowercases_and_collapses_whitespace():
    assert refdata.normalise_answer("  My  Name IS Rahul ") == "my name is rahul"


def test_prepare_corpus_frame_rejects_missing_columns():
    with pytest.raises(ValueError):
        refdata.prepare_corpus_frame(pd.DataFrame([{"qid": 1}]))


def test_pool_excludes_non_informative_answers():
    pools = refdata.build_reference_pools(tiny_frame())
    assert len(pools.pools[0]) == 2
    assert len(pools.pools[1]) == 1


def test_pool_only_contains_test_side_references():
    pools = refdata.build_reference_pools(tiny_frame())
    assert 2 not in pools.pools
    assert set(pools.rows["split"]) == {"test"}


def test_pool_deduplicates_identical_answers():
    df = tiny_frame()
    dup = df.iloc[[0]].copy()
    dup["qid"] = 99
    dup["row_id"] = 99
    df = pd.concat([df, dup], ignore_index=True)
    pools = refdata.build_reference_pools(refdata.prepare_corpus_frame(df))
    assert len(pools.pools[0]) == 2


def test_split_leakage_raises():
    df = tiny_frame()
    df.loc[df["qid"] == 3, "split"] = "train"
    with pytest.raises(ValueError, match="span more than one split"):
        refdata.build_reference_pools(df)


def test_own_answer_id_is_none_for_call_log_notes():
    pools = refdata.build_reference_pools(tiny_frame())
    note_row = int(pools.rows.loc[pools.rows["qid"] == 3, "row_id"].iloc[0])
    assert pools.own_id[note_row] is None


def test_own_answer_id_points_at_the_representative_answer():
    pools = refdata.build_reference_pools(tiny_frame())
    row = int(pools.rows.loc[pools.rows["qid"] == 2, "row_id"].iloc[0])
    assert pools.own_id[row] in pools.pools[0]
    assert pools.norm[pools.own_id[row]] == pools.rows.set_index("qid").loc[2, "answer_norm"]


# ------------------------------------------------------------------- metrics

def test_pool_max_f1_can_exclude_the_own_answer():
    pool = ["spray rogor 2 ml per litre", "use urea 50 kg per acre"]
    answer = "spray rogor 2 ml per litre"
    assert mref.pool_max_f1(answer, pool) == pytest.approx(1.0)
    other = mref.pool_max_f1(answer, pool, exclude_norm=answer)
    assert other is not None and other < 1.0


def test_pool_max_f1_returns_none_for_empty_pool():
    assert mref.pool_max_f1("anything", []) is None


def test_max_reference_f1_includes_own_answer_when_informative():
    answer = "spray rogor 2 ml per litre"
    pool = ["use urea 50 kg per acre"]
    value = mref.max_reference_f1(answer, pool, answer, own_is_reference=True)
    assert value == pytest.approx(1.0)


def test_max_reference_f1_ignores_own_answer_when_not_informative():
    answer = "spray rogor 2 ml per litre"
    pool = ["use urea 50 kg per acre"]
    value = mref.max_reference_f1(answer, pool, "explained in details",
                                  own_is_reference=False)
    assert value is not None and value < 1.0


def test_random_other_reference_is_seeded_and_reproducible():
    pool = ["spray rogor 2 ml per litre", "use urea 50 kg per acre",
            "apply imidacloprid 0.5 ml per litre water"]
    answer = "spray rogor 2 ml per litre"
    a = mref.random_other_reference_f1(answer, pool, answer, np.random.default_rng(0))
    b = mref.random_other_reference_f1(answer, pool, answer, np.random.default_rng(0))
    c = mref.random_other_reference_f1(answer, pool, answer, np.random.default_rng(7))
    assert a == pytest.approx(b)
    assert isinstance(c, float)


def test_random_other_reference_is_none_without_alternatives():
    pool = ["spray rogor 2 ml per litre"]
    assert mref.random_other_reference_f1(
        "spray rogor 2 ml per litre", pool, "spray rogor 2 ml per litre",
        np.random.default_rng(0)) is None


def test_token_f1_handles_hindi_text():
    assert refdata.token_f1("यूरिया 50 किलो प्रति एकड़", "यूरिया 50 किलो प्रति एकड़") == 1.0
    assert refdata.token_f1("यूरिया 50 किलो", "यूरिया 20 किलो") < 1.0


# --------------------------------------------------------------- statistics

def test_paired_bootstrap_on_constant_delta():
    result = mref.paired_bootstrap([1.0, 1.0, 1.0], [0.0, 0.0, 0.0], n_boot=200)
    assert result["n"] == 3
    assert result["mean_delta"] == pytest.approx(1.0)
    assert result["ci95"] == [1.0, 1.0]
    assert result["share_improved"] == 1.0
    assert result["significant_at_0_05"] is True


def test_paired_bootstrap_drops_nan_pairs_and_reports_n():
    result = mref.paired_bootstrap([1.0, np.nan, 0.5], [0.0, 0.0, 0.0], n_boot=100)
    assert result["n"] == 2


def test_paired_bootstrap_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        mref.paired_bootstrap([1.0, 2.0], [1.0])


def test_cohens_dz_sign_follows_the_delta():
    assert mref.cohens_dz([0.6, 0.7, 0.8], [0.2, 0.3, 0.4]) > 0
    assert mref.cohens_dz([0.2, 0.3, 0.4], [0.6, 0.7, 0.8]) < 0


def test_summarise_handles_all_missing_values():
    out = mref.summarise([np.nan, np.nan])
    assert out["n"] == 0 and out["mean"] is None


def test_recovered_misses_counts_rows_an_alternative_would_have_matched():
    frame = pd.DataFrame({"own_f1": [0.2, 0.6], "max_other_f1": [0.7, 0.8]})
    out = mref.recovered_misses(frame)
    assert out["n_single_reference_miss"] == 1
    assert out["n_recovered_by_another_reference"] == 1
    assert out["share_of_misses_recovered"] == 1.0


# --------------------------------------------------------------- pool stats

def test_pool_stats_reports_multi_reference_support_on_fixture():
    pools = refdata.build_reference_pools(tiny_frame())
    stats = refdata.pool_stats(pools)
    assert stats["n_unique_references"] == 3
    assert stats["n_rows_with_2plus_references"] == 3
    assert stats["supports_multi_reference"] is True
    assert stats["reference_diversity"]["n_groups_sampled"] == 1


def test_split_checks_show_no_group_spans_splits():
    checks = refdata.split_checks(tiny_frame())
    assert checks["groups_spanning_splits"] == 0


# ------------------------------------------------------------------ ranking

def test_screened_top_k_skips_uninformative_candidates():
    ids = np.array([[0, 1, 2, 3], [4, 5, -1, -1]])
    valid = np.array([False, True, False, True, True, False])
    out = ranking.screened_top_k(ids, valid, k=2)
    assert out[0].tolist() == [1, 3]
    assert out[1].tolist() == [4, -1]


def test_fixed_sample_is_deterministic_and_matches_seed():
    a = ranking.fixed_sample(10_000, size=100, seed=42)
    b = ranking.fixed_sample(10_000, size=100, seed=42)
    assert np.array_equal(a, b)
    assert np.all(np.diff(a) >= 0)


# ------------------------------------------------------------------- report

def _metric_block(mean: float) -> dict:
    return {"n": 3, "n_missing": 0, "mean": mean,
            "ci95": [round(mean - 0.01, 4), round(mean + 0.01, 4)], "median": mean}


def _subset(n: int = 3) -> dict:
    return {
        "n_rows": n,
        "own_reference_f1": _metric_block(0.3),
        "max_reference_f1": _metric_block(0.4),
        "max_other_reference_f1": _metric_block(0.35),
        "random_other_reference_f1": _metric_block(0.25),
        "max_reference_semantic": _metric_block(0.7),
        "recovered_misses": {
            "threshold": 0.5, "n_comparable": 3, "n_single_reference_miss": 2,
            "n_recovered_by_another_reference": 1, "share_of_misses_recovered": 0.5,
            "note": "",
        },
    }


def fake_metrics() -> dict:
    names = ("all", "gold_informative", "pool_ge_2", "pool_eq_1", "pool_ge_20",
             "pool_eq_0", "novel", "similar", "near_duplicate")
    systems = {}
    for system in ("P1_dense_all", "P1_dense_sample", "P1_hybrid_sample",
                   "P2_dense_sample", "P2_hybrid_sample"):
        systems[system] = {"system": "dense", "rows": "x",
                           "candidate_condition": "unscreened",
                           "n_rows": 3,
                           "subsets": {name: _subset() for name in names}}
    return {
        "generated_utc": "2026-11-04T00:00:00+00:00",
        "protocol": {"hybrid_sample": 3000},
        "seeds": {"bootstrap": 0},
        "split_checks": {"groups_spanning_splits": 0, "n_groups": 5},
        "reference_pools": {
            "supports_multi_reference": True,
            "criteria": {"min_share_rows_with_2plus_references": 0.5,
                         "min_share_diverse_multi_ref_groups": 0.5},
            "n_test_rows": 4, "n_test_groups": 2, "n_groups_with_references": 2,
            "n_unique_references": 3, "n_rows_with_reference": 4,
            "share_rows_with_reference": 1.0, "n_rows_with_2plus_references": 3,
            "share_rows_with_2plus_references": 0.75, "n_rows_with_5plus_references": 0,
            "share_rows_with_5plus_references": 0.0, "n_rows_without_reference": 0,
            "share_rows_without_reference": 0.0,
            "references_per_test_row": {"mean": 1.75, "median": 2.0, "p90": 2.0, "max": 2},
            "references_per_group": {"n_groups": 2, "mean": 1.5, "median": 1.5,
                                     "p90": 2.0, "max": 2},
            "skew": {"n_rows_in_groups_with_ge_100_references": 0,
                     "share_rows_in_groups_with_ge_100_references": 0.0},
            "reference_diversity": {"mean_max_pairwise_token_f1": 0.55,
                                    "share_groups_below_0_80": 1.0},
        },
        "reproduction_gate": {
            "ok": True, "sample_seed": 42, "n_sample_queries": 3000,
            "checks": [{"check": "dense all test | own-reference token-F1",
                        "batch_a_value": 0.3213, "recomputed_value": 0.3213,
                        "abs_diff": 0.0, "ok": True}],
        },
        "systems": systems,
        "paired_comparisons": {
            "P1_sample | hybrid vs dense | own-reference": {
                "n": 3, "mean_delta": 0.058, "ci95": [0.01, 0.1],
                "share_improved": 0.66, "share_unchanged": 0.34, "share_worse": 0.0,
                "cohens_dz": 0.5, "significant_at_0_05": True,
            }
        },
        "limitations": ["References are historical advisor answers, not gold."],
        "environment": {"python": "3.12", "platform": "test",
                        "versions": {"numpy": "2", "sklearn": "1", "faiss": "1",
                                     "sentence_transformers": "3"}},
    }


def test_report_renders_gate_metrics_and_the_correctness_caveat():
    text = report.render(fake_metrics())
    assert "Improvement 3" in text
    assert "PASS" in text
    assert "does **not** measure agricultural correctness" in text
    assert "0.3000" in text
    assert "Batch A reproduction gate" in text


def test_report_raises_when_metrics_structure_is_missing():
    with pytest.raises(KeyError):
        report.render({"protocol": {}})
