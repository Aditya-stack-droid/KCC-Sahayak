#!/usr/bin/env python
"""Score the shipped rule and the new screener on the same authored cases.

Compares ``confidence_gate.answer_risk`` (v1, frozen production rule) with
``answer_screening.screening.risk_reason`` (v2, negation- and measurement-aware)
on an identical case set, and reports the standard confusion metrics plus the
two quantities that matter for this change:

* **negation false-positive rate** — of the safe cases whose situation is a
  negation/warning, the share wrongly flagged;
* **unsafe-answer acceptance rate** — of the unsafe cases, the share the screen
  misses (``FN / (TP+FN)``). This must not increase.

``ambiguous`` rows are excluded from precision/recall/F1 and reported separately.
No rule is tuned here.
"""
from __future__ import annotations

import argparse
import csv
import random
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for path in (ROOT, ROOT / "src", ROOT / "experiments"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from answer_screening import cases as cases_module  # noqa: E402
from answer_screening import report as report_module  # noqa: E402

try:
    from confidence_gate import answer_risk as _risk_v1
except Exception:  # pragma: no cover
    from src.confidence_gate import answer_risk as _risk_v1  # type: ignore

from answer_screening.screening import risk_reason as _risk_v2  # noqa: E402

DATA = HERE / "data" / "screening_cases.csv"
DEFAULT_OUT = ROOT / "results" / "answer_screening"
N_BOOT = 2000
SEED = 0
NEGATION_SITUATIONS = {
    "direct_negation", "negation_before_chemical", "negation_after_chemical",
    "hindi_negation", "roman_hindi_negation", "hinglish_negation",
    "spelling_variation", "punctuation", "negated_dosage",
    "dangerous_combination",
}


def load_cases(path: Path = DATA) -> list[dict]:
    with Path(path).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _prf(tp: int, fp: int, fn: int) -> dict:
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = (2 * precision * recall / (precision + recall)) if precision and recall else None
    return {
        "tp": tp, "fp": fp, "fn": fn,
        "precision": None if precision is None else round(precision, 4),
        "recall": None if recall is None else round(recall, 4),
        "f1": None if f1 is None else round(f1, 4),
    }


def _block(rows: list[dict]) -> dict:
    scored = [r for r in rows if r["gold"] in {"unsafe", "safe"}]
    tp = sum(1 for r in scored if r["gold"] == "unsafe" and r["predicted"])
    fp = sum(1 for r in scored if r["gold"] == "safe" and r["predicted"])
    fn = sum(1 for r in scored if r["gold"] == "unsafe" and not r["predicted"])
    tn = sum(1 for r in scored if r["gold"] == "safe" and not r["predicted"])
    stats = _prf(tp, fp, fn)
    stats.update({
        "tn": tn,
        "n": len(scored),
        "accuracy": round((tp + tn) / len(scored), 4) if scored else None,
        "false_positive_rate": round(fp / (fp + tn), 4) if (fp + tn) else None,
        "unsafe_acceptance_rate": round(fn / (tp + fn), 4) if (tp + fn) else None,
        "confusion_matrix": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
    })
    flags = [(r["gold"] == "unsafe", r["predicted"]) for r in scored]
    stats["f1_95ci"] = _bootstrap(flags, "f1")
    return stats


def _bootstrap(flags: list[tuple[bool, bool]], metric: str, n_boot: int = N_BOOT, seed: int = SEED) -> dict:
    rng = random.Random(seed)
    values: list[float] = []
    if not flags:
        return {"low": None, "high": None, "mean": None, "n_boot": n_boot, "seed": seed}
    n = len(flags)
    for _ in range(n_boot):
        tp = fp = fn = 0
        for _ in range(n):
            gold, pred = flags[rng.randrange(n)]
            if gold and pred:
                tp += 1
            elif pred:
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
        "n_boot": n_boot, "seed": seed,
    }


def _breakdown(rows: list[dict], key: str) -> dict:
    return {value: _block([r for r in rows if r[key] == value]) for value in sorted({r[key] for r in rows})}


def evaluate(cases: list[dict]) -> tuple[dict, list[dict]]:
    predictions: list[dict] = []
    for case in cases:
        v1 = _risk_v1(case["text"]) is not None
        v2 = _risk_v2(case["text"]) is not None
        predictions.append({
            "case_id": case["case_id"],
            "text": case["text"],
            "language": case["language"],
            "situation": case["situation"],
            "gold": case["gold"],
            "v1_fired": v1,
            "v2_fired": v2,
        })

    def view(system: str) -> list[dict]:
        return [dict(r, predicted=r[system]) for r in predictions]

    scored = [r for r in predictions if r["gold"] in {"unsafe", "safe"}]
    ambiguous = [r for r in predictions if r["gold"] == "ambiguous"]

    v1_rows, v2_rows = view("v1_fired"), view("v2_fired")

    def neg_fp(rows: list[dict]) -> dict:
        subset = [r for r in rows if r["situation"] in NEGATION_SITUATIONS and r["gold"] == "safe"]
        flagged = sum(1 for r in subset if r["predicted"])
        return {
            "safe_negation_cases": len(subset),
            "flagged": flagged,
            "negation_false_positive_rate": round(flagged / len(subset), 4) if subset else None,
        }

    metrics = {
        "protocol": {
            "dataset": "experiments/answer_screening/data/screening_cases.csv",
            "synthetic": True,
            "provenance": "authored for this evaluation by the person who wrote the screener; not adjudicated",
            "gold_definition": (
                "unsafe = specific actionable agrochemical instruction or a question about such a "
                "product (must fire and escalate); safe = no such instruction; ambiguous excluded"
            ),
            "systems": {
                "v1": "src/confidence_gate.py::answer_risk (frozen production rule)",
                "v2": "experiments/answer_screening/screening.py::risk_reason (negation + measurement aware)",
            },
            "negation_situations": sorted(NEGATION_SITUATIONS),
            "bootstrap": {"n_boot": N_BOOT, "seed": SEED},
        },
        "counts": {
            "total": len(predictions),
            "scored": len(scored),
            "unsafe": sum(1 for r in scored if r["gold"] == "unsafe"),
            "safe": sum(1 for r in scored if r["gold"] == "safe"),
            "ambiguous_excluded": len(ambiguous),
        },
        "composition": {
            "language": dict(Counter(r["language"] for r in predictions)),
            "situation": dict(Counter(r["situation"] for r in predictions)),
            "gold": dict(Counter(r["gold"] for r in predictions)),
        },
        "v1": {
            "overall": _block(v1_rows),
            "languages": _breakdown(v1_rows, "language"),
            "situations": _breakdown(v1_rows, "situation"),
            "negation_fp": neg_fp(v1_rows),
            "false_negatives": [r["case_id"] for r in scored if r["gold"] == "unsafe" and not r["v1_fired"]],
            "false_positives": [r["case_id"] for r in scored if r["gold"] == "safe" and r["v1_fired"]],
        },
        "v2": {
            "overall": _block(v2_rows),
            "languages": _breakdown(v2_rows, "language"),
            "situations": _breakdown(v2_rows, "situation"),
            "negation_fp": neg_fp(v2_rows),
            "false_negatives": [r["case_id"] for r in scored if r["gold"] == "unsafe" and not r["v2_fired"]],
            "false_positives": [r["case_id"] for r in scored if r["gold"] == "safe" and r["v2_fired"]],
        },
        "comparison": {
            "unsafe_acceptance_rate_change": _delta(_block(v1_rows)["unsafe_acceptance_rate"], _block(v2_rows)["unsafe_acceptance_rate"]),
            "false_positives_change": _block(v2_rows)["fp"] - _block(v1_rows)["fp"],
            "negation_false_positive_rate_change": _delta(
                neg_fp(v1_rows)["negation_false_positive_rate"], neg_fp(v2_rows)["negation_false_positive_rate"]),
            "new_false_positives_vs_v1": sorted(
                set(r["case_id"] for r in scored if r["gold"] == "safe" and r["v2_fired"])
                - set(r["case_id"] for r in scored if r["gold"] == "safe" and r["v1_fired"])),
            "new_false_negatives_vs_v1": sorted(
                set(r["case_id"] for r in scored if r["gold"] == "unsafe" and not r["v2_fired"])
                - set(r["case_id"] for r in scored if r["gold"] == "unsafe" and not r["v1_fired"])),
        },
        "ambiguity": {
            "n": len(ambiguous),
            "v1_fired": sum(1 for r in ambiguous if r["v1_fired"]),
            "v2_fired": sum(1 for r in ambiguous if r["v2_fired"]),
            "rows": [{"case_id": r["case_id"], "v1_fired": r["v1_fired"], "v2_fired": r["v2_fired"], "text": r["text"]} for r in ambiguous],
        },
        "limitations": [
            "All cases are synthetic and authored by the screener's author; not a sample of real queries.",
            "Gold labels are authored expectations about escalation, not expert-adjudicated agronomic harm.",
            "The author had read both rule sets, so the 'challenge' cases are not an unbiased field estimate.",
            "v2 adds a wider vocabulary (brands, Devanagari), so part of the recall change is vocabulary, not only negation/measurement logic.",
            "Detecting a pattern is not agronomic correctness; a miss does not prove an answer is safe.",
            "The shipped production gate is unchanged; v2 is an evaluated candidate, not yet integrated.",
        ],
    }
    return metrics, predictions


def _delta(before, after):
    if before is None or after is None:
        return None
    return round(after - before, 4)


def _write_predictions(rows: list[dict], path: Path) -> None:
    fields = ["case_id", "language", "situation", "gold", "v1_fired", "v2_fired", "text"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, "v1_fired": str(row["v1_fired"]).lower(), "v2_fired": str(row["v2_fired"]).lower()})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cases", default=str(DATA))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    targets = [out / "metrics.json", out / "predictions.csv", out / "REPORT.md"]
    for path in targets:
        if path.exists() and not args.overwrite:
            print(f"refusing to overwrite {path} (pass --overwrite)")
            return 1

    import json

    metrics, predictions = evaluate(load_cases(Path(args.cases)))
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    _write_predictions(predictions, out / "predictions.csv")
    (out / "REPORT.md").write_text(report_module.render(metrics), encoding="utf-8")
    print(json.dumps({
        "counts": metrics["counts"],
        "v1": metrics["v1"]["overall"],
        "v2": metrics["v2"]["overall"],
        "comparison": metrics["comparison"],
        "written": [str(p) for p in targets],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
