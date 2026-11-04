"""Multiple-reference and paired-comparison metrics (Improvement 3).

All functions are pure and deterministic. Token-F1 is Batch A's implementation
(imported from ``refdata``), so numbers stay comparable with the original
single-reference metric.
"""
from __future__ import annotations

import numpy as np

from mref_refdata import bootstrap_ci, token_f1


def pool_max_f1(answer_norm: str, pool_norms, exclude_norm: str | None = None):
    """Highest token-F1 between an answer and any reference; None if no reference."""
    best = None
    for ref in pool_norms:
        if exclude_norm is not None and ref == exclude_norm:
            continue
        score = token_f1(answer_norm, ref)
        if best is None or score > best:
            best = score
    return best


def max_reference_f1(answer_norm: str, pool_norms, own_norm: str | None,
                     own_is_reference: bool):
    """Best F1 over the pool plus the query's own answer when that is a reference.

    The own answer is a member of the pool when it is informative, so
    ``max_reference_f1 >= own_reference_f1`` holds by construction. This is
    reported for completeness, never as the headline comparison.
    """
    best = pool_max_f1(answer_norm, pool_norms) if pool_norms else None
    if own_is_reference and own_norm:
        own = token_f1(answer_norm, own_norm)
        best = own if best is None else max(best, own)
    return best


def random_other_reference_f1(answer_norm: str, pool_norms, exclude_norm: str | None,
                              rng, draws: int = 5):
    """Mean F1 against ``draws`` random non-own references; chance-level control.

    Answers the question "how much of the max-reference gain is just having
    more chances to match?" If max >> random, the retrieved answer genuinely
    agrees with several advisors; if max ~ random, the pool is broad enough
    that almost anything matches something.
    """
    candidates = [p for p in pool_norms if exclude_norm is None or p != exclude_norm]
    if not candidates:
        return None
    idx = rng.integers(0, len(candidates), size=draws)
    return float(np.mean([token_f1(answer_norm, candidates[int(i)]) for i in idx]))


def paired_bootstrap(a, b, n_boot: int = 2000, seed: int = 0, alpha: float = 0.05) -> dict:
    """Paired bootstrap over per-query deltas ``a - b``.

    NaNs are dropped pairwise (a metric is undefined for some rows, e.g. no
    non-own reference), and the number of usable pairs is reported so the
    comparison is never read as covering more queries than it does.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError("paired arrays must have the same length")
    mask = ~(np.isnan(a) | np.isnan(b))
    d = a[mask] - b[mask]
    n = int(d.size)
    if n == 0:
        return {"n": 0, "mean_delta": None, "ci95": [None, None],
                "share_improved": None, "share_unchanged": None, "share_worse": None,
                "cohens_dz": None}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    means = d[idx].mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    sd = float(d.std(ddof=1)) if n > 1 else 0.0
    return {
        "n": n,
        "mean_delta": round(float(d.mean()), 6),
        "ci95": [round(float(lo), 6), round(float(hi), 6)],
        "share_improved": round(float((d > 0).mean()), 4),
        "share_unchanged": round(float((d == 0).mean()), 4),
        "share_worse": round(float((d < 0).mean()), 4),
        "cohens_dz": round(float(d.mean() / sd), 4) if sd > 0 else None,
        "significant_at_0_05": bool(lo > 0 or hi < 0),
    }


def cohens_dz(a, b) -> float | None:
    """Standardised mean paired difference (mean delta / sd of deltas)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    d = a - b
    d = d[~np.isnan(d)]
    if d.size < 2:
        return None
    sd = float(d.std(ddof=1))
    return round(float(d.mean() / sd), 4) if sd > 0 else None


def summarise(values, n_boot: int = 2000, seed: int = 0) -> dict:
    """Mean, CI and coverage for a per-query metric array."""
    x = np.asarray(values, dtype=float)
    valid = x[~np.isnan(x)]
    if valid.size == 0:
        return {"n": 0, "n_missing": int(x.size), "mean": None, "ci95": [None, None],
                "median": None}
    return {
        "n": int(valid.size),
        "n_missing": int(x.size - valid.size),
        "mean": round(float(valid.mean()), 6),
        "ci95": bootstrap_ci(valid, n_boot=n_boot, seed=seed),
        "median": round(float(np.median(valid)), 6),
    }


def subset_summary(frame, mask) -> dict:
    """Summarise every scored column over one row subset."""
    sub = frame[mask]
    return {
        "n_rows": int(len(sub)),
        "own_reference_f1": summarise(sub["own_f1"]),
        "max_reference_f1": summarise(sub["max_f1"]),
        "max_other_reference_f1": summarise(sub["max_other_f1"]),
        "random_other_reference_f1": summarise(sub["random_other_f1"]),
        "max_reference_semantic": summarise(sub["max_semantic"]),
    }


def recovered_misses(frame, threshold: float = 0.5) -> dict:
    """Rows a single reference scored below threshold but an alternative matched.

    This is the quantity multi-reference evaluation exists to expose: retrieval
    that looks wrong against one logged answer but agrees with another advisor
    answer to the same question.
    """
    own = frame["own_f1"].to_numpy()
    other = frame["max_other_f1"].to_numpy()
    mask = ~(np.isnan(own) | np.isnan(other))
    own, other = own[mask], other[mask]
    missed = own < threshold
    return {
        "threshold": threshold,
        "n_comparable": int(mask.sum()),
        "n_single_reference_miss": int(missed.sum()),
        "n_recovered_by_another_reference": int((missed & (other >= threshold)).sum()),
        "share_of_misses_recovered": round(
            float((other[missed] >= threshold).mean()), 4) if missed.any() else None,
        "note": ("Agreement with another historical advisor answer, not evidence of "
                 "agricultural correctness."),
    }
