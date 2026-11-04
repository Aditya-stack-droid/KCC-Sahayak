"""Detection metrics and the validation-only threshold policy.

Positive class = OUT_OF_SCOPE. A query is *rejected* when its in-scope score
falls below the threshold.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    import sys
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# Reuse Batch A's bootstrap implementation rather than duplicating it.
_grouped = _load("kcc_grouped_metrics_oos",
                 ROOT / "experiments" / "grouped_evaluation" / "metrics.py")
bootstrap_ci = _grouped.bootstrap_ci


def binary_metrics(y_true, rejected) -> dict:
    """Precision/recall/F1 for out-of-scope, plus false accept/reject rates."""
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(rejected, dtype=bool)
    if y.shape != p.shape:
        raise ValueError("y_true and rejected must have the same length")
    tp = int(np.sum((y == 1) & p))
    fp = int(np.sum((y == 0) & p))
    fn = int(np.sum((y == 1) & ~p))
    tn = int(np.sum((y == 0) & ~p))
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    far = fn / (tp + fn) if (tp + fn) else None      # out-of-scope accepted
    frr = fp / (fp + tn) if (fp + tn) else None      # in-scope rejected
    tpr = recall
    tnr = tn / (tn + fp) if (tn + fp) else 0.0
    return {
        "n": int(len(y)),
        "n_in_scope": int((y == 0).sum()),
        "n_out_of_scope": int((y == 1).sum()),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "precision": round(float(precision), 6),
        "recall": round(float(recall), 6),
        "f1": round(float(f1), 6),
        "balanced_accuracy": round(float((tpr + tnr) / 2), 6),
        "false_acceptance_rate": None if far is None else round(float(far), 6),
        "false_rejection_rate": None if frr is None else round(float(frr), 6),
    }


def f1_ci(y_true, rejected, n_boot: int = 2000, seed: int = 0) -> list:
    """Query-level bootstrap CI for out-of-scope F1."""
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(rejected, dtype=bool)
    if len(y) == 0:
        return [None, None]
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), size=len(y))
        m = binary_metrics(y[idx], p[idx])
        values.append(m["f1"])
    lo, hi = np.percentile(values, [2.5, 97.5])
    return [round(float(lo), 4), round(float(hi), 4)]


def select_threshold(scores_in, scores_out, target_frr: float = 0.02) -> dict:
    """Highest threshold whose validation in-scope rejection rate is within target.

    Operationally: "reject at most ``target_frr`` of real agricultural queries,
    then reject as many out-of-scope queries as possible". Selection uses
    validation rows only. Raises if no candidate meets the constraint, rather
    than silently relaxing the policy.
    """
    scores_in = np.asarray(scores_in, dtype=float)
    scores_out = np.asarray(scores_out, dtype=float)
    if scores_in.size == 0 or scores_out.size == 0:
        raise ValueError("threshold selection needs both in-scope and out-of-scope scores")
    candidates = np.unique(np.round(np.concatenate([scores_in, scores_out]), 6))
    best = None
    for t in candidates:
        frr = float(np.mean(scores_in < t))
        if frr <= target_frr + 1e-12:
            recall = float(np.mean(scores_out < t))
            if (best is None or recall > best["val_recall"]
                    or (recall == best["val_recall"] and t > best["threshold"])):
                best = {
                    "threshold": float(t),
                    "val_false_rejection_rate": round(frr, 6),
                    "val_recall": round(recall, 6),
                    "policy": f"max out-of-scope recall subject to validation FRR <= {target_frr}",
                    "target_frr": target_frr,
                    "n_val_in_scope": int(scores_in.size),
                    "n_val_out_of_scope": int(scores_out.size),
                }
    if best is None:
        raise ValueError(f"no threshold achieves validation FRR <= {target_frr}")
    return best


def threshold_sweep(y_true, scores, grid) -> list:
    """Description of the threshold trade-off; never used for selection."""
    rows = []
    for t in grid:
        m = binary_metrics(y_true, np.asarray(scores) < t)
        rows.append({
            "threshold": float(t),
            "false_acceptance_rate": m["false_acceptance_rate"],
            "false_rejection_rate": m["false_rejection_rate"],
            "f1": m["f1"],
            "precision": m["precision"],
            "recall": m["recall"],
        })
    return rows
