#!/usr/bin/env python
"""Compare the single-intent baseline with rule-based multi-intent decomposition.

Baseline ("single"): the shipped keyword rule ``src/weak_labels.label_query``
returns one label per query. Candidate ("multi"): ``decompose`` returns a label
set. Both are rule-based and deterministic; neither loads a model.

Metrics are set-based (a query is a set of intents):
* exact-set match, micro/macro precision/recall/F1 over (query, label) pairs,
* missed-intent rate, extra-intent rate, decomposition accuracy,
* safety-escalation recall and precision (a query must escalate when any gold
  intent is high-stakes or time-sensitive).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for path in (ROOT, ROOT / "src", ROOT / "experiments"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from glossary import HIGH_STAKES_TYPES, TIME_SENSITIVE_TYPES  # noqa: E402
from weak_labels import label_query as _single_label  # noqa: E402

from multi_intent import cases as cases_module  # noqa: E402
from multi_intent import report as report_module  # noqa: E402
from multi_intent.decompose import LABELS, decompose  # noqa: E402

DATA = HERE / "data" / "multi_intent_cases.csv"
DEFAULT_OUT = ROOT / "results" / "multi_intent"
RISKY = HIGH_STAKES_TYPES | TIME_SENSITIVE_TYPES


def load_cases(path: Path = DATA) -> list[dict]:
    with Path(path).open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["gold_intents"] = [x for x in row["gold_intents"].split(";") if x]
    return rows


def _sets(predicted: list[str], gold: list[str]) -> tuple[set, set]:
    return set(predicted), set(gold)


def _micro(pairs: list[tuple[set, set]]) -> dict:
    tp = sum(len(p & g) for p, g in pairs)
    fp = sum(len(p - g) for p, g in pairs)
    fn = sum(len(g - p) for p, g in pairs)
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = 2 * precision * recall / (precision + recall) if precision and recall else None
    return {"tp": tp, "fp": fp, "fn": fn,
            "precision": None if precision is None else round(precision, 4),
            "recall": None if recall is None else round(recall, 4),
            "f1": None if f1 is None else round(f1, 4)}


def _macro(pairs: list[tuple[set, set]]) -> dict:
    per = {}
    for label in LABELS:
        tp = sum(1 for p, g in pairs if label in p and label in g)
        fp = sum(1 for p, g in pairs if label in p and label not in g)
        fn = sum(1 for p, g in pairs if label not in p and label in g)
        precision = tp / (tp + fp) if tp + fp else None
        recall = tp / (tp + fn) if tp + fn else None
        f1 = 2 * precision * recall / (precision + recall) if precision and recall else None
        per[label] = {"tp": tp, "fp": fp, "fn": fn,
                      "precision": None if precision is None else round(precision, 4),
                      "recall": None if recall is None else round(recall, 4),
                      "f1": None if f1 is None else round(f1, 4)}
    present = [v["f1"] for v in per.values() if v["f1"] is not None]
    return {"per_label": per, "macro_f1": round(sum(present) / len(present), 4) if present else None}


def evaluate(cases: list[dict]) -> tuple[dict, list[dict]]:
    records = []
    for case in cases:
        gold = sorted(set(case["gold_intents"]))
        single = [_single_label(case["text"])]
        multi = decompose(case["text"])
        records.append({
            "case_id": case["case_id"],
            "text": case["text"],
            "language": case["language"],
            "category": case["category"],
            "gold": gold,
            "single": single,
            "multi": sorted(multi.distinct_intents),
            "multi_is_multi": multi.is_multi,
            "gold_safety": any(l in RISKY for l in gold),
            "multi_safety": multi.safety_escalation_required,
            "n_sub": len(multi.sub_queries),
        })

    def score(rows, key) -> dict:
        pairs = [_sets(r[key], r["gold"]) for r in rows]
        exact = sum(1 for p, g in pairs if p == g) / len(rows) if rows else None
        missed = sum(len(g - p) for p, g in pairs) / sum(len(g) for _, g in pairs) if pairs else None
        extra = sum(len(p - g) for p, g in pairs) / sum(len(p) for p, _ in pairs) if pairs else None
        decomp = None
        multi_rows = [r for r in rows if len(set(r["gold"])) >= 2]
        if multi_rows:
            decomp = sum(1 for r in multi_rows if len(set(r[key])) >= 2) / len(multi_rows)
        return {
            "exact_set_match": round(exact, 4) if exact is not None else None,
            "micro": _micro(pairs),
            "macro": _macro(pairs),
            "missed_intent_rate": round(missed, 4) if missed is not None else None,
            "extra_intent_rate": round(extra, 4) if extra is not None else None,
            "decomposition_accuracy": round(decomp, 4) if decomp is not None else None,
        }

    def safety(rows, key) -> dict:
        gold_pos = [r for r in rows if r["gold_safety"]]
        tp = sum(1 for r in gold_pos if r[key])
        fp = sum(1 for r in rows if not r["gold_safety"] and r[key])
        tn = sum(1 for r in rows if not r["gold_safety"] and not r[key])
        fn = sum(1 for r in gold_pos if not r[key])
        precision = tp / (tp + fp) if tp + fp else None
        recall = tp / (tp + fn) if tp + fn else None
        return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
                "precision": None if precision is None else round(precision, 4),
                "recall": None if recall is None else round(recall, 4)}

    single_safety = [dict(r, single="X" if False else r["single"]) for r in records]
    # baseline safety: the single label is risky if it is high-stakes/time-sensitive
    for r in records:
        r["single_safety"] = any(l in RISKY for l in r["single"])

    metrics = {
        "protocol": {
            "dataset": "experiments/multi_intent/data/multi_intent_cases.csv",
            "synthetic": True,
            "provenance": "authored for this evaluation by the person who wrote the decomposer; not adjudicated",
            "definition": "multi-intent = the query asks for two or more distinct intent classes",
            "baseline": "src/weak_labels.label_query (existing single-label keyword rule)",
            "candidate": "experiments/multi_intent/decompose.decompose (rule-based, escalate-only)",
            "labels": LABELS,
        },
        "counts": {
            "total": len(records),
            "multi": sum(1 for r in records if len(set(r["gold"])) >= 2),
            "single": sum(1 for r in records if len(set(r["gold"])) == 1),
            "safety_positive": sum(1 for r in records if r["gold_safety"]),
            "by_language": dict(Counter(r["language"] for r in records)),
            "by_category": dict(Counter(r["category"] for r in records)),
        },
        "single": {**score(records, "single"), "safety": safety(records, "single_safety")},
        "multi": {**score(records, "multi"), "safety": safety(records, "multi_safety")},
        "by_language": {
            lang: {
                "n": len([r for r in records if r["language"] == lang]),
                "single": score([r for r in records if r["language"] == lang], "single"),
                "multi": score([r for r in records if r["language"] == lang], "multi"),
            }
            for lang in sorted({r["language"] for r in records})
        },
        "failures": {
            "multi_missed_intents": [
                {"case_id": r["case_id"], "text": r["text"], "gold": r["gold"],
                 "predicted": r["multi"], "missing": sorted(set(r["gold"]) - set(r["multi"]))}
                for r in records if set(r["gold"]) - set(r["multi"])
            ],
            "multi_extra_intents": [
                {"case_id": r["case_id"], "text": r["text"], "gold": r["gold"],
                 "predicted": r["multi"], "extra": sorted(set(r["multi"]) - set(r["gold"]))}
                for r in records if set(r["multi"]) - set(r["gold"])
            ],
            "safety_missed": [r["case_id"] for r in records if r["gold_safety"] and not r["multi_safety"]],
        },
        "limitations": [
            "Cases are synthetic and authored by the decomposer's author; not real traffic.",
            "Gold intent sets are authored judgements; multi-intent labelling is inherently subjective.",
            "The baseline is the keyword rule, not the deployed ML classifier (no model loaded here).",
            "The decomposer reuses English keyword rules plus a small bilingual cue lexicon; it is not a trained multi-label model.",
            "It is not wired into the production gate; it would only add an escalation signal.",
        ],
    }
    return metrics, records


def _write_predictions(records: list[dict], path: Path) -> None:
    fields = ["case_id", "language", "category", "gold", "single", "multi", "gold_safety", "multi_safety"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields + ["text"], extrasaction="ignore")
        writer.writeheader()
        for r in records:
            writer.writerow({**r, "gold": ";".join(r["gold"]), "single": ";".join(r["single"]), "multi": ";".join(r["multi"])})


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
    metrics, records = evaluate(load_cases(Path(args.cases)))
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    _write_predictions(records, out / "predictions.csv")
    (out / "REPORT.md").write_text(report_module.render(metrics), encoding="utf-8")
    print(json.dumps({
        "counts": metrics["counts"],
        "single_micro_f1": metrics["single"]["micro"]["f1"],
        "multi_micro_f1": metrics["multi"]["micro"]["f1"],
        "single_exact": metrics["single"]["exact_set_match"],
        "multi_exact": metrics["multi"]["exact_set_match"],
        "single_safety_recall": metrics["single"]["safety"]["recall"],
        "multi_safety_recall": metrics["multi"]["safety"]["recall"],
        "written": [str(p) for p in targets],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
