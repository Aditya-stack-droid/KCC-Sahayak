#!/usr/bin/env python
"""Score the production safety gate on the authored adversarial case set.

Two views are scored separately:

``rule``
    the production content rule (``answer_risk``) alone;
``gate``
    the full gate on a content-only slice (non-high-stakes, high confidence), so
    the only available escalation reasons are answer-screening ones.

``ambiguous`` rows are excluded from precision/recall/F1 and reported separately.
No rule is tuned here: the production module is imported read-only and its
SHA-256 is recorded in the metrics.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.safety_evaluation.rules import (  # noqa: E402
    GATE_INTENT_CONFIDENCE,
    GATE_QUERY_TYPE,
    GATE_RETRIEVAL_SIMILARITY,
    gate_verdict,
    production_source_path,
    rule_verdict,
)

DATA = HERE / "data" / "adversarial_cases.csv"
DEFAULT_OUT = ROOT / "results" / "safety_evaluation"
N_BOOT = 2000
SEED = 0
PREDICTION_FIELDS = [
    "case_id", "language", "subset", "category", "gold", "severity",
    "rule_fired", "rule_reason", "gate_action", "gate_code", "gate_escalated",
    "text",
]


def load_cases(path: Path = DATA) -> list[dict]:
    with Path(path).open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        if row["gold"] not in {"unsafe", "safe", "ambiguous"}:
            raise ValueError(f"bad gold label: {row['case_id']}={row['gold']}")
    return rows


def _prf(tp: int, fp: int, fn: int) -> dict:
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = (2 * precision * recall / (precision + recall)) if precision and recall else None
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": None if precision is None else round(precision, 4),
        "recall": None if recall is None else round(recall, 4),
        "f1": None if f1 is None else round(f1, 4),
    }


def _bootstrap(flags: list[tuple[bool, bool]], metric: str, n_boot: int = N_BOOT,
               seed: int = SEED) -> dict:
    """Percentile bootstrap CI for precision/recall/F1 over scored cases."""
    values: list[float] = []
    rng = random.Random(seed)
    n = len(flags)
    if not n:
        return {"low": None, "high": None, "mean": None, "n_boot": n_boot, "seed": seed}
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
        value = _prf(tp, fp, fn)[metric]
        if value is not None:
            values.append(value)
    if not values:
        return {"low": None, "high": None, "mean": None, "n_boot": n_boot, "seed": seed}
    values.sort()
    return {
        "low": round(values[max(0, int(0.025 * len(values)) - 1)], 4),
        "high": round(values[min(len(values) - 1, int(0.975 * len(values)))], 4),
        "mean": round(sum(values) / len(values), 4),
        "n_boot": n_boot,
        "seed": seed,
    }


def _block(rows: list[dict]) -> dict:
    scored = [row for row in rows if row["gold"] in {"unsafe", "safe"}]
    tp = sum(1 for row in scored if row["gold"] == "unsafe" and row["predicted"])
    fp = sum(1 for row in scored if row["gold"] == "safe" and row["predicted"])
    fn = sum(1 for row in scored if row["gold"] == "unsafe" and not row["predicted"])
    tn = sum(1 for row in scored if row["gold"] == "safe" and not row["predicted"])
    stats = _prf(tp, fp, fn)
    stats.update({
        "tn": tn,
        "n": len(scored),
        "accuracy": round((tp + tn) / len(scored), 4) if scored else None,
        "false_escalation_rate": round(fp / (fp + tn), 4) if (fp + tn) else None,
        "false_safe_rate": round(fn / (tp + fn), 4) if (tp + fn) else None,
        # The gate escalates many things besides content; this is the honest
        # denominator: unsafe cases the gate still auto-answers.
        "gate_autosafe": (tp + fn) - sum(
            1 for row in scored if row["gold"] == "unsafe" and row.get("gate_escalated")
        ),
    })
    flags = [(row["gold"] == "unsafe", row["predicted"]) for row in scored]
    stats["f1_95ci"] = _bootstrap(flags, "f1")
    stats["recall_95ci"] = _bootstrap(flags, "recall")
    return stats


def evaluate(cases: list[dict]) -> tuple[dict, list[dict]]:
    predictions: list[dict] = []
    for case in cases:
        rule = rule_verdict(case["text"])
        gate = gate_verdict(case["text"])
        predictions.append({
            "case_id": case["case_id"],
            "language": case["language"],
            "subset": case["subset"],
            "category": case["category"],
            "gold": case["gold"],
            "severity": case["severity"],
            "rule_fired": rule["fired"],
            "rule_reason": rule["reason"] or "",
            "gate_action": gate["action"],
            "gate_code": gate["code"],
            "gate_escalated": gate["escalated"],
            "text": case["text"],
            "gold_unsafe": case["gold"] == "unsafe",
        })

    for row in predictions:
        row["predicted"] = row["rule_fired"]

    scored = [row for row in predictions if row["gold"] in {"unsafe", "safe"}]
    ambiguous = [row for row in predictions if row["gold"] == "ambiguous"]

    # Gate-level predictions are a separate view over the same scored rows.
    for row in scored:
        row["predicted_gate"] = row["gate_escalated"]

    rule_rows = scored
    gate_rows = [dict(row, predicted=row["predicted_gate"]) for row in scored]

    def breakdown(rows: list[dict], key: str) -> dict:
        values = sorted({row[key] for row in rows})
        return {value: _block([row for row in rows if row[key] == value]) for value in values}

    rule_overall = _block(rule_rows)
    gate_overall = _block(gate_rows)

    false_negatives = [
        {
            "case_id": row["case_id"],
            "language": row["language"],
            "subset": row["subset"],
            "category": row["category"],
            "severity": row["severity"],
            "gate_action": row["gate_action"],
            "text": row["text"],
        }
        for row in scored if row["gold"] == "unsafe" and not row["predicted"]
    ]
    false_positives = [
        {
            "case_id": row["case_id"],
            "language": row["language"],
            "subset": row["subset"],
            "category": row["category"],
            "rule_reason": row["rule_reason"],
            "text": row["text"],
        }
        for row in scored if row["gold"] == "safe" and row["predicted"]
    ]

    gate_misses = [row["case_id"] for row in scored
                   if row["gold"] == "unsafe" and not row["gate_escalated"]]

    gate_source = production_source_path()
    metrics = {
        "protocol": {
            "dataset": "experiments/safety_evaluation/data/adversarial_cases.csv",
            "synthetic": True,
            "provenance": (
                "authored for this evaluation by the person who wrote the evaluator; "
                "no farmer, advisor or expert was involved; not adjudicated"
            ),
            "gold_definition": (
                "unsafe = specific actionable agrochemical instruction (product, dose, "
                "concentration or application); safe = no such instruction; "
                "ambiguous = excluded from P/R/F1"
            ),
            "production_gate": {
                "path": str(gate_source.relative_to(ROOT)),
                "sha256": hashlib.sha256(gate_source.read_bytes()).hexdigest(),
                "modified_for_this_evaluation": False,
            },
            "gate_slice": {
                "query_type": GATE_QUERY_TYPE,
                "intent_confidence": GATE_INTENT_CONFIDENCE,
                "retrieval_similarity": GATE_RETRIEVAL_SIMILARITY,
                "reason": "non-high-stakes, non-time-sensitive, high confidence: only answer screening can escalate",
            },
            "bootstrap": {"n_boot": N_BOOT, "seed": SEED, "percentile": 0.95},
            "subsets": "core = plain English cases; challenge = Hindi, Hinglish, obfuscated, indirect, negation, unit variation; authorship-based, not a held-out split",
        },
        "counts": {
            "total": len(predictions),
            "scored": len(scored),
            "unsafe": sum(1 for row in scored if row["gold"] == "unsafe"),
            "safe": sum(1 for row in scored if row["gold"] == "safe"),
            "ambiguous_excluded": len(ambiguous),
            "synthetic": len(predictions),
        },
        "composition": {
            "language": dict(Counter(row["language"] for row in predictions)),
            "subset": dict(Counter(row["subset"] for row in predictions)),
            "category": dict(Counter(row["category"] for row in predictions)),
            "category_gold": {
                category: dict(Counter(row["gold"] for row in predictions if row["category"] == category))
                for category in sorted({row["category"] for row in predictions})
            },
            "language_gold": {
                language: dict(Counter(row["gold"] for row in predictions if row["language"] == language))
                for language in sorted({row["language"] for row in predictions})
            },
            "gold_by_subset": {
                subset: dict(Counter(row["gold"] for row in predictions if row["subset"] == subset))
                for subset in sorted({row["subset"] for row in predictions})
            },
        },
        "rule": {
            "overall": rule_overall,
            "subsets": breakdown(rule_rows, "subset"),
            "languages": breakdown(rule_rows, "language"),
            "categories": breakdown(rule_rows, "category"),
        },
        "gate": {
            "overall": gate_overall,
            "subsets": breakdown(gate_rows, "subset"),
            "languages": breakdown(gate_rows, "language"),
            "categories": breakdown(gate_rows, "category"),
            "unsafe_still_auto_answered": gate_misses,
        },
        "ambiguity": {
            "n": len(ambiguous),
            "rule_fired": sum(1 for row in ambiguous if row["predicted"]),
            "gate_escalated": sum(1 for row in ambiguous if row["gate_escalated"]),
            "rows": [
                {
                    "case_id": row["case_id"],
                    "language": row["language"],
                    "rule_fired": row["predicted"],
                    "gate_action": row["gate_action"],
                    "text": row["text"],
                }
                for row in ambiguous
            ],
        },
        "errors": {
            "false_negatives": false_negatives,
            "false_positives": false_positives,
            "false_negatives_by_severity": dict(Counter(row["severity"] for row in false_negatives)),
            "false_negatives_by_category": dict(Counter(row["category"] for row in false_negatives)),
            "false_negatives_by_language": dict(Counter(row["language"] for row in false_negatives)),
            "false_positives_by_category": dict(Counter(row["category"] for row in false_positives)),
        },
        "limitations": [
            "All cases are synthetic and authored by the evaluator; they are not a sample of real KCC queries or answers.",
            "Gold labels are authored expectations about escalation, not expert-adjudicated agronomic harm.",
            "The dataset was written by someone who had read the production regexes, which biases the 'core' subset towards detectable phrasing.",
            "The 'challenge' subset was written to probe known regex boundaries, so its low scores are not an unbiased estimate of field performance either.",
            "Rules were not tuned after seeing these results; the measured behaviour is the shipped behaviour.",
            "Detecting a pattern is not agronomic correctness: a rule firing does not mean the advice is wrong, and a miss does not prove the advice is safe.",
            "No real farmer speech, OCR noise or code-mixed conversational transcript is included.",
        ],
    }
    return metrics, predictions


def _write_predictions(rows: list[dict], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=PREDICTION_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({
                **row,
                "rule_fired": "true" if row["rule_fired"] else "false",
                "gate_escalated": "true" if row["gate_escalated"] else "false",
            })


def _write_errors(rows: list[dict], path: Path) -> None:
    fields = ["case_id", "language", "subset", "category", "severity", "text"] if rows and "severity" in rows[0] \
        else ["case_id", "language", "subset", "category", "text"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cases", default=str(DATA))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    metrics_path = out / "metrics.json"
    outputs = [
        metrics_path,
        out / "predictions.csv",
        out / "false_negatives.csv",
        out / "false_positives.csv",
    ]
    for path in outputs:
        if path.exists() and not args.overwrite:
            print(f"refusing to overwrite {path} (pass --overwrite)")
            return 1

    metrics, predictions = evaluate(load_cases(Path(args.cases)))
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    _write_predictions(predictions, out / "predictions.csv")
    _write_errors(metrics["errors"]["false_negatives"], out / "false_negatives.csv")
    _write_errors(metrics["errors"]["false_positives"], out / "false_positives.csv")

    print(json.dumps({
        "counts": metrics["counts"],
        "rule": metrics["rule"]["overall"],
        "gate": metrics["gate"]["overall"],
        "rule_by_subset": metrics["rule"]["subsets"],
        "written": [str(path) for path in outputs],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
