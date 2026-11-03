#!/usr/bin/env python
"""Evaluate pairwise conflict detection on the authored synthetic pair set.

Metrics are strict pair-level labels: a pair is a conflict when at least one
kind is detected.  Per-kind metrics treat that kind as a binary label over all
pairs.  Everything here is synthetic and authored for this evaluation; the
numbers do not estimate how often real KCC answers conflict.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.conflict_detection.conflicts import (  # noqa: E402
    DOSE_GAP_TOLERANCE,
    MAX_ANSWERS,
    detect_pair,
)

DATA = HERE / "data" / "conflict_pairs.csv"
DEFAULT_OUT = ROOT / "results" / "conflict_detection"
ALLOWED = ("dosage", "product", "basis", "dimension", "interval", "polarity")


def load_pairs(path: Path = DATA) -> list[dict]:
    with Path(path).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _prf(tp: int, fp: int, fn: int) -> dict:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def _bootstrap_f1(flags: list[tuple[bool, bool]], n_boot: int = 2000, seed: int = 0) -> dict:
    if not flags:
        return {"low": None, "high": None, "mean": None, "n_boot": n_boot, "seed": seed}
    rng = random.Random(seed)
    values: list[float] = []
    n = len(flags)
    for _ in range(n_boot):
        tp = fp = fn = 0
        for _ in range(n):
            gold, predicted = flags[rng.randrange(n)]
            if gold and predicted:
                tp += 1
            elif predicted:
                fp += 1
            elif gold:
                fn += 1
        values.append(_prf(tp, fp, fn)["f1"])
    values.sort()
    return {
        "low": round(values[max(0, int(0.025 * n_boot) - 1)], 4),
        "high": round(values[min(n_boot - 1, int(0.975 * n_boot))], 4),
        "mean": round(sum(values) / len(values), 4),
        "n_boot": n_boot,
        "seed": seed,
    }


def evaluate(pairs: list[dict]) -> tuple[dict, list[dict]]:
    predictions: list[dict] = []
    for pair in pairs:
        gold = str(pair["gold_conflict"]).strip().lower() == "true"
        gold_kinds = {kind for kind in str(pair.get("gold_kinds", "")).split(";") if kind}
        conflicts = detect_pair(pair["text_a"], pair["text_b"])
        predicted_kinds = {conflict.kind for conflict in conflicts}
        severity = conflicts[0].severity if conflicts else None
        predictions.append({
            "pair_id": pair["pair_id"],
            "subset": pair.get("subset", "core"),
            "gold_conflict": gold,
            "predicted_conflict": bool(predicted_kinds),
            "gold_kinds": sorted(gold_kinds),
            "predicted_kinds": sorted(predicted_kinds),
            "max_severity": severity,
            "explanations": [conflict.explanation for conflict in conflicts],
            "text_a": pair["text_a"],
            "text_b": pair["text_b"],
        })

    tp = sum(1 for row in predictions if row["gold_conflict"] and row["predicted_conflict"])
    fp = sum(1 for row in predictions if not row["gold_conflict"] and row["predicted_conflict"])
    fn = sum(1 for row in predictions if row["gold_conflict"] and not row["predicted_conflict"])
    tn = sum(1 for row in predictions if not row["gold_conflict"] and not row["predicted_conflict"])
    binary = _prf(tp, fp, fn)
    binary["tn"] = tn
    binary["n"] = len(predictions)
    binary["accuracy"] = round((tp + tn) / len(predictions), 4) if predictions else 0.0
    binary["f1_95ci"] = _bootstrap_f1([(row["gold_conflict"], row["predicted_conflict"])
                                       for row in predictions])

    def subset_metrics(rows: list[dict]) -> dict:
        subset_tp = sum(1 for row in rows if row["gold_conflict"] and row["predicted_conflict"])
        subset_fp = sum(1 for row in rows if not row["gold_conflict"] and row["predicted_conflict"])
        subset_fn = sum(1 for row in rows if row["gold_conflict"] and not row["predicted_conflict"])
        subset_tn = sum(1 for row in rows if not row["gold_conflict"] and not row["predicted_conflict"])
        stats = _prf(subset_tp, subset_fp, subset_fn)
        stats["tn"] = subset_tn
        stats["n"] = len(rows)
        stats["accuracy"] = round((subset_tp + subset_tn) / len(rows), 4) if rows else 0.0
        stats["f1_95ci"] = _bootstrap_f1(
            [(row["gold_conflict"], row["predicted_conflict"]) for row in rows]
        )
        return stats

    subsets = {
        subset: subset_metrics([row for row in predictions if row["subset"] == subset])
        for subset in sorted({row["subset"] for row in predictions})
    }

    per_kind: dict[str, dict] = {}
    for kind in ALLOWED:
        kind_tp = sum(1 for row in predictions if kind in row["gold_kinds"] and kind in row["predicted_kinds"])
        kind_fp = sum(1 for row in predictions if kind not in row["gold_kinds"] and kind in row["predicted_kinds"])
        kind_fn = sum(1 for row in predictions if kind in row["gold_kinds"] and kind not in row["predicted_kinds"])
        stats = _prf(kind_tp, kind_fp, kind_fn)
        stats["support"] = sum(1 for row in predictions if kind in row["gold_kinds"])
        per_kind[kind] = stats

    severity_counts = Counter(row["max_severity"] for row in predictions if row["max_severity"])
    errors = [
        {
            "pair_id": row["pair_id"],
            "gold_conflict": row["gold_conflict"],
            "predicted_conflict": row["predicted_conflict"],
            "gold_kinds": row["gold_kinds"],
            "predicted_kinds": row["predicted_kinds"],
            "explanations": row["explanations"],
        }
        for row in predictions
        if row["gold_conflict"] != row["predicted_conflict"]
        or set(row["gold_kinds"]) != set(row["predicted_kinds"])
    ]

    metrics = {
        "protocol": {
            "dataset": "experiments/conflict_detection/data/conflict_pairs.csv",
            "synthetic": True,
            "provenance": "hand-authored for this evaluation; not real advisor data; not expert-verified",
            "matching": "pair-level any-conflict; per-kind binary over all pairs",
            "dose_gap_tolerance": DOSE_GAP_TOLERANCE,
            "max_answers_per_scan": MAX_ANSWERS,
            "bootstrap": {"n_boot": 2000, "seed": 0, "percentile": 95},
            "no_model_training": True,
        },
        "n_pairs": len(predictions),
        "n_gold_conflict": sum(1 for row in predictions if row["gold_conflict"]),
        "n_gold_no_conflict": sum(1 for row in predictions if not row["gold_conflict"]),
        "binary": binary,
        "subsets": subsets,
        "per_kind": per_kind,
        "severity_counts": dict(severity_counts),
        "errors": errors,
        "limitations": [
            "The pair set is small and hand-authored by the same person who wrote the rules.",
            "Per-kind support for dimension/basis is too small for stable estimates; wide uncertainty is expected.",
            "A detected conflict means the rules fired; it is not proof that either answer is agronomically wrong.",
            "Negation is clause-local and heuristic; complex conditional advice is out of scope.",
        ],
    }
    return metrics, predictions


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", default=str(DATA))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    metrics_path = out / "metrics.json"
    predictions_path = out / "pair_predictions.csv"
    for path in (metrics_path, predictions_path):
        if path.exists() and not args.overwrite:
            print(f"refusing to overwrite {path} (pass --overwrite)")
            return 1

    metrics, predictions = evaluate(load_pairs(Path(args.pairs)))
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    with predictions_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "pair_id", "subset", "gold_conflict", "predicted_conflict", "gold_kinds",
            "predicted_kinds", "max_severity", "explanations", "text_a", "text_b",
        ])
        writer.writeheader()
        for row in predictions:
            writer.writerow({
                **row,
                "gold_kinds": ";".join(row["gold_kinds"]),
                "predicted_kinds": ";".join(row["predicted_kinds"]),
                "explanations": " | ".join(row["explanations"]),
            })
    print(json.dumps({
        "n_pairs": metrics["n_pairs"],
        "gold_conflict": metrics["n_gold_conflict"],
        "binary": metrics["binary"],
        "subsets": metrics["subsets"],
        "per_kind": {kind: {"support": stats["support"], "f1": stats["f1"]}
                     for kind, stats in metrics["per_kind"].items()},
        "written": [str(metrics_path), str(predictions_path)],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
