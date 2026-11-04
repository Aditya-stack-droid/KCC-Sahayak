#!/usr/bin/env python
"""Safety-focused comparison of three multi-intent systems on the same cases.

Systems
-------
``baseline``  the shipped single-label keyword rule (``src/weak_labels``).
``v1``        the original decomposer (``multi_intent.decompose``) — no risk layer.
``v2``        the risk-first detector (``multi_intent.detect``) — per-sub-query and
              whole-query risk assessment, conservative escalation.

Evaluated on the original case set (``cases.py``) and the new adversarial set
(``adversarial_cases.py``), reported separately and combined. Every high-risk
case that a system fails to escalate is listed individually.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (ROOT, ROOT / "src", ROOT / "experiments"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from glossary import HIGH_STAKES_TYPES, TIME_SENSITIVE_TYPES  # noqa: E402
from weak_labels import label_query as _single  # noqa: E402

from multi_intent import adversarial_cases as adv  # noqa: E402
from multi_intent import cases as v1_cases  # noqa: E402
from multi_intent import report_v2 as report_module  # noqa: E402
from multi_intent.decompose import decompose  # noqa: E402
from multi_intent.detect import detect  # noqa: E402

RISKY = set(HIGH_STAKES_TYPES) | set(TIME_SENSITIVE_TYPES)
DEFAULT_OUT = ROOT / "results" / "multi_intent_v2"


def load(path: Path, gold_split_on: str = ";") -> list[dict]:
    with Path(path).open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for r in rows:
        r["gold_intents"] = [x for x in r["gold_intents"].split(gold_split_on) if x]
        r["gold_safety"] = any(x in RISKY for x in r["gold_intents"])
    return rows


def _prf(pairs: list[tuple[set, set]]) -> dict:
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


def _macro(pairs: list[tuple[set, set]]) -> float | None:
    labels = sorted({l for p, g in pairs for l in (p | g)})
    f1s = []
    for lab in labels:
        tp = sum(1 for p, g in pairs if lab in p and lab in g)
        fp = sum(1 for p, g in pairs if lab in p and lab not in g)
        fn = sum(1 for p, g in pairs if lab not in p and lab in g)
        pr = tp / (tp + fp) if tp + fp else None
        rc = tp / (tp + fn) if tp + fn else None
        if pr is not None and rc is not None and pr + rc:
            f1s.append(2 * pr * rc / (pr + rc))
    return round(sum(f1s) / len(f1s), 4) if f1s else None


def _score_system(rows: list[dict], predicted_key: str, escalate_key: str | None) -> dict:
    pairs = [(set(r[predicted_key]), set(r["gold_intents"])) for r in rows]
    exact = sum(1 for p, g in pairs if p == g) / len(rows) if rows else None
    missed = sum(len(g - p) for p, g in pairs) / sum(len(g) for _, g in pairs) if pairs else None
    extra = sum(len(p - g) for p, g in pairs) / sum(len(p) for p, _ in pairs) if pairs else None
    multi_rows = [r for r in rows if len(set(r["gold_intents"])) >= 2]
    decomp = (sum(1 for r in multi_rows if len(set(r[predicted_key])) >= 2) / len(multi_rows)
              if multi_rows else None)

    def escalates(r):
        if escalate_key is None:
            return any(l in RISKY for l in r[predicted_key])
        return bool(r[escalate_key])

    risk_pos = [r for r in rows if r["gold_safety"]]
    tp = sum(1 for r in risk_pos if escalates(r))
    fn = sum(1 for r in risk_pos if not escalates(r))
    fp = sum(1 for r in rows if not r["gold_safety"] and escalates(r))
    tn = sum(1 for r in rows if not r["gold_safety"] and not escalates(r))
    prec = tp / (tp + fp) if tp + fp else None
    rec = tp / (tp + fn) if tp + fn else None

    risk_intent_hits = sum(1 for r in risk_pos if any(l in RISKY for l in r[predicted_key]))
    missed_risk = [r["case_id"] for r in risk_pos if not escalates(r)]
    return {
        "n": len(rows),
        "exact_set_match": round(exact, 4) if exact is not None else None,
        "micro": _prf(pairs),
        "macro_f1": _macro(pairs),
        "missed_intent_rate": round(missed, 4) if missed is not None else None,
        "extra_intent_rate": round(extra, 4) if extra is not None else None,
        "decomposition_accuracy": round(decomp, 4) if decomp is not None else None,
        "safety_escalation": {
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": None if prec is None else round(prec, 4),
            "recall": None if rec is None else round(rec, 4),
        },
        "high_risk_intent_recall": round(risk_intent_hits / len(risk_pos), 4) if risk_pos else None,
        "high_risk_false_negatives": missed_risk,
        "high_risk_fn_count": len(missed_risk),
    }


def _predict(rows: list[dict]) -> list[dict]:
    out = []
    for r in rows:
        text = r["text"]
        v1 = decompose(text)
        v2 = detect(text)
        out.append({
            "case_id": r["case_id"], "text": text, "language": r["language"],
            "category": r["category"], "gold_intents": r["gold_intents"], "gold_safety": r["gold_safety"],
            "baseline": [_single(text)],
            "v1": sorted(v1.distinct_intents),
            "v1_escalate": any(l in RISKY and l != "Other" for l in v1.distinct_intents),
            "v2": sorted(v2.distinct_intents),
            "v2_escalate": v2.escalate,
            "v2_reasons": v2.escalation_reasons,
        })
    return out


def evaluate(rows: list[dict]) -> dict:
    preds = _predict(rows)
    return {
        "n": len(preds),
        "baseline": _score_system(preds, "baseline", None),
        "v1": _score_system(preds, "v1", "v1_escalate"),
        "v2": _score_system(preds, "v2", "v2_escalate"),
        "predictions": preds,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    old = load(HERE / "data" / "multi_intent_cases.csv")
    new = load(HERE / "data" / "adversarial_cases_v2.csv")
    combined = old + new

    metrics = {
        "protocol": {
            "systems": {"baseline": "src/weak_labels.label_query (single label)",
                        "v1": "multi_intent.decompose (no risk layer)",
                        "v2": "multi_intent.detect (risk-first)"},
            "risk_intents": sorted(RISKY),
            "gold_safety_definition": "any gold intent in the high-stakes or time-sensitive set",
            "case_sets": ["multi_intent_cases.csv (v1 set)", "adversarial_cases_v2.csv (new set)"],
        },
        "original_set": evaluate(old),
        "adversarial_set": evaluate(new),
        "combined": evaluate(combined),
        "limitations": [
            "Cases are synthetic and authored by the module's author; not real traffic.",
            "gold_safety is derived from authored gold intents; a mislabelled gold intent would mislabel safety.",
            "v2 escalates more by design (conservative); this lowers escalation precision on ambiguous cases.",
            "Rule-based confidence is a heuristic, not a calibrated probability.",
        ],
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name in ("metrics.json", "predictions.csv", "REPORT.md"):
        if (out / name).exists() and not args.overwrite:
            print(f"refusing to overwrite {out / name} (pass --overwrite)")
            return 1
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    with (out / "predictions.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["case_id", "language", "category", "gold_intents", "gold_safety",
                         "baseline", "v1", "v1_escalate", "v2", "v2_escalate"])
        for r in metrics["combined"]["predictions"]:
            writer.writerow([r["case_id"], r["language"], r["category"], ";".join(r["gold_intents"]),
                             r["gold_safety"], ";".join(r["baseline"]), ";".join(r["v1"]),
                             r["v1_escalate"], ";".join(r["v2"]), r["v2_escalate"]])
    (out / "REPORT.md").write_text(report_module.render(metrics), encoding="utf-8")

    summary = {k: {"micro_f1": metrics[k]["v2"]["micro"]["f1"],
                   "high_risk_fn": metrics[k]["v2"]["high_risk_fn_count"],
                   "v1_high_risk_fn": metrics[k]["v1"]["high_risk_fn_count"]}
               for k in ("original_set", "adversarial_set", "combined")}
    print(json.dumps(summary, indent=2))
    print("written:", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
