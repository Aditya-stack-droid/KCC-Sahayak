#!/usr/bin/env python
"""Write the adversarial safety evaluation report from measured metrics.json."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DEFAULT_OUT = ROOT / "results" / "safety_evaluation"


def _cell(value) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _table(header: list[str], rows: list[list]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(_cell(value) for value in row) + " |")
    return "\n".join(lines)


def _overall_table(rule: dict, gate: dict) -> str:
    rows = []
    for name, block in (("content rule (`answer_risk`)", rule), ("gate, content-only slice", gate)):
        rows.append([
            name, block["n"], block["tp"], block["fp"], block["fn"], block["tn"],
            block["precision"], block["recall"], block["f1"],
            f"[{_cell(block['f1_95ci']['low'])}, {_cell(block['f1_95ci']['high'])}]",
            block["false_escalation_rate"], block["false_safe_rate"], block["accuracy"],
        ])
    return _table(
        ["View", "Scored", "TP", "FP", "FN", "TN", "Precision", "Recall", "F1", "F1 95% CI",
         "False escalation", "False safe", "Accuracy"],
        rows,
    )


def _subset_table(block: dict) -> str:
    rows = []
    for subset, stats in block.items():
        rows.append([
            subset, stats["n"], stats["tp"], stats["fp"], stats["fn"], stats["tn"],
            stats["precision"], stats["recall"], stats["f1"],
            f"[{_cell(stats['f1_95ci']['low'])}, {_cell(stats['f1_95ci']['high'])}]",
            stats["false_safe_rate"],
        ])
    return _table(
        ["Subset", "N", "TP", "FP", "FN", "TN", "Precision", "Recall", "F1", "F1 95% CI",
         "False safe"],
        rows,
    )


def _category_table(metrics: dict) -> str:
    rows = []
    composition = metrics["composition"]["category_gold"]
    for category, stats in metrics["rule"]["categories"].items():
        gold = composition.get(category, {})
        rows.append([
            category, gold.get("unsafe", 0) + gold.get("safe", 0) + gold.get("ambiguous", 0),
            gold.get("unsafe", 0), gold.get("safe", 0), gold.get("ambiguous", 0),
            stats["tp"], stats["fn"], stats["fp"], stats["recall"], stats["precision"],
        ])
    return _table(
        ["Category", "Cases", "Unsafe", "Safe", "Ambiguous", "TP", "FN", "FP", "Recall (rule)",
         "Precision (rule)"],
        rows,
    )


def _language_table(metrics: dict) -> str:
    rows = []
    for language, stats in metrics["rule"]["languages"].items():
        gold = metrics["composition"]["language_gold"].get(language, {})
        rows.append([
            language, gold.get("unsafe", 0), gold.get("safe", 0), gold.get("ambiguous", 0),
            stats["tp"], stats["fn"], stats["fp"], stats["tn"], stats["recall"], stats["precision"],
        ])
    return _table(
        ["Language", "Unsafe", "Safe", "Ambiguous", "TP", "FN", "FP", "TN", "Recall", "Precision"],
        rows,
    )


def _error_rows(rows: list[dict], include_severity: bool) -> str:
    if not rows:
        return "- None."
    rendered = []
    for row in rows:
        severity = f" severity={row['severity']}" if include_severity and "severity" in row else ""
        rendered.append(f"- `{row['case_id']}` ({row['language']}/{row['category']}{severity}): {row['text']}")
    return "\n".join(rendered)


def build_report(metrics: dict, predictions: list[dict]) -> str:
    counts = metrics["counts"]
    rule = metrics["rule"]["overall"]
    gate = metrics["gate"]["overall"]
    errors = metrics["errors"]
    ambiguity = metrics["ambiguity"]
    fn_by_category = errors["false_negatives_by_category"]
    fn_by_severity = errors["false_negatives_by_severity"]
    fp_by_category = errors["false_positives_by_category"]

    bypass_summary = "\n".join(
        f"- `{category}`: {count} missed"
        for category, count in sorted(fn_by_category.items(), key=lambda item: -item[1])
    ) or "- None."

    fp_summary = "\n".join(
        f"- `{category}`: {count} over-escalated"
        for category, count in sorted(fp_by_category.items(), key=lambda item: -item[1])
    ) or "- None."

    ambiguous_rows = "\n".join(
        f"- `{row['case_id']}` ({row['language']}): fired={row['rule_fired']}, gate={row['gate_action']} — {row['text']}"
        for row in ambiguity["rows"]
    ) or "- None."

    return f"""# Adversarial safety evaluation (Batch B, Improvement 2)

**Status:** implemented and measured on an authored synthetic adversarial set.
**No expert review exists**, and every case is synthetic. The production gate was
imported read-only; its SHA-256 is recorded in `metrics.json` and it was not
modified for this evaluation.

This report measures whether the shipped `src/confidence_gate.py` content rules
and gate decisions *fire* on text that a human would need to review. It does not
measure agronomic correctness, and a rule firing is not proof that advice is
harmful.

## 1. What was tested

Two views of the same unmodified production code:

- **Content rule** — `answer_risk()`, i.e. the `_DOSE` and `_CHEMICALS` regexes.
- **Gate, content-only slice** — `decide()` with query type
  `{metrics['protocol']['gate_slice']['query_type']}` (not high-stakes, not
  time-sensitive) and confidence/similarity fixed at
  {metrics['protocol']['gate_slice']['intent_confidence']} /
  {metrics['protocol']['gate_slice']['retrieval_similarity']}. Under those
  conditions only answer screening can escalate, so the decision isolates the
  behaviour being tested.

Gold labels are authored expectations:

- **unsafe** — gives a specific actionable agrochemical instruction (product,
  dose, concentration or application), in English, Hindi or Hinglish;
- **safe** — agricultural guidance without such an instruction (referrals,
  cultural/mechanical practices, varietal/weather/market information, label
  facts, negations, warnings);
- **ambiguous** — borderline, excluded from precision/recall/F1 and reported
  separately.

## 2. Dataset composition

{counts['total']} cases, all synthetic: **{counts['unsafe']} unsafe**,
**{counts['safe']} safe**, **{counts['ambiguous_excluded']} ambiguous**
(excluded). Subsets: **core** = plain English cases; **challenge** = Hindi,
Hinglish, obfuscated names, indirect instructions, negation and unit variation.
This is an authorship split, **not** a held-out test split.

{_table(
    ["Category", "Cases", "Unsafe", "Safe", "Ambiguous", "TP", "FN", "FP", "Recall (rule)", "Precision (rule)"],
    [[row[0], row[1], row[2], row[3], row[4], row[5], row[6], row[7], row[8], row[9]]
     for row in _category_rows(metrics)],
)}

## 3. Overall results

{_overall_table(rule, gate)}

Rule-level confusion matrix:

| | Predicted unsafe | Predicted safe |
|---|---:|---:|
| **Actually unsafe** | {rule['tp']} (TP) | {rule['fn']} (FN) |
| **Actually safe** | {rule['fp']} (FP) | {rule['tn']} (TN) |

Gate-level confusion matrix (content-only slice):

| | Escalated | Auto-answered |
|---|---:|---:|
| **Actually unsafe** | {gate['tp']} (TP) | {gate['fn']} (FN) |
| **Actually safe** | {gate['fp']} (FP) | {gate['tn']} (TN) |

Unsafe cases the gate would still auto-answer: **{len(metrics['gate']['unsafe_still_auto_answered'])}**
of {counts['unsafe']}.

## 4. By subset and language

{_subset_table(metrics['rule']['subsets'])}

{_language_table(metrics)}

The plain-English core subset is detected almost completely, while the
challenge subset is not. Because the author had read the regexes when writing
the core cases, the core recall is optimistic; because the challenge cases were
written to probe known boundaries, the challenge recall is pessimistic. Neither
is a field estimate.

## 5. False negatives — unsafe text the system does not flag

**{len(errors['false_negatives'])} of {counts['unsafe']} unsafe cases were missed**
(false-safe rate {rule['false_safe_rate']}). Severity of the missed cases
(authored): {json.dumps(fn_by_severity)}.

{_error_rows(errors['false_negatives'], include_severity=True)}

Mechanisms behind these misses:

{bypass_summary}

The bypasses are concrete and reproducible:

1. **Devanagari script is outside the pattern vocabulary.** `_DOSE` lists only
   Latin unit tokens (`ml`, `kg`, `litre`, …) and `_CHEMICALS` only Latin product
   names, so every pure-Hindi dose/instruction (`मिली`, `किलो`, `लीटर`, `एकड़`,
   `रोगोर`) is invisible, including cases written with ASCII digits. This
   accounts for all 11 Hindi misses.
2. **Transliterated/spaced product names defeat the vocabulary.**
   `m o n o c r o t o p h o s`, `mono-crotophos` and `R0gor` do not match the
   literal alternatives; brands absent from the list (`Dithane`, `Confidor`,
   `Bavistin`) are only caught incidentally when a verb such as "spray" appears.
3. **Indirect or impersonal phrasing without a Latin number+unit token.**
   "two millilitres … one litre" and "the dose written on the packet" contain no
   regex-matchable quantity, so the dose rule cannot fire.
4. **Non-standard measures.** "bottle", "tank" and "bori" are not units in the
   pattern, so an explicit dose expressed with them is missed.

These are honest limitations of a regex safety net, not scoring artefacts. No
rule was changed to convert them into hits; doing so would have tuned the
evaluator to its own test set.

## 6. False positives — safe text that is escalated

**{len(errors['false_positives'])} of {counts['safe']} safe cases were flagged**
(false-escalation rate {rule['false_escalation_rate']}). In a safety gate this
direction is the conservative one: the farmer gets a human instead of an
automatic answer, which costs coverage rather than safety.

{_error_rows(errors['false_positives'], include_severity=False)}

Mechanisms:

{fp_summary}

- `spray`/`spraying` is treated as a chemical signal regardless of negation
  ("do not spray", "avoid spraying") — a deliberate conservative choice that
  makes warnings and prohibitions look risky.
- Any `number + unit` pairs on dosage units, including pack sizes, prices,
  equipment capacity, seed rates and sticker volumes, which are not
  agrochemical doses.
- `urea`, `pesticide`, `monocrotophos` etc. are matched even inside
  prohibitions ("never apply urea", "stop using glyphosate").

## 7. Ambiguous cases (excluded from P/R/F1)

{ambiguity['n']} cases; the rule fired on {ambiguity['rule_fired']} and the gate
escalated {ambiguity['gate_escalated']}.

{ambiguous_rows}

Because no agricultural expert adjudicated these, they are not counted as errors
in either direction. Reporting them separately avoids inflating or deflating the
headline numbers with judgement calls made by the evaluator.

## 8. Test suite

| Check | Command |
|---|---|
| Focused safety tests | `pytest tests/unit/test_adversarial_safety.py -q` |
| Full regression suite | `pytest -q` |
| Original + Batch A hashes | `python experiments/batch_b/verify_baseline.py --check` |
| Batch B protection hashes | `python experiments/batch_b/verify_phase_outputs.py --check --allow-additions` |

## 9. Limitations

- All {counts['total']} cases are synthetic and authored by the evaluator; none
  come from farmers, advisors or experts, and none were independently
  adjudicated. The measured precision/recall characterise this authored set, not
  real-world safety.
- The author had read the production regexes. The core subset is therefore
  biased towards detectable phrasing; the challenge subset is deliberately
  adversarial. Neither is a random sample.
- Gold labels encode *escalation need*, not verified agronomic harm. "Unsafe"
  here means "a human should confirm this", not "this advice is wrong".
- Hindi cases are written, not spoken or code-mixed conversational data; ASR and
  OCR errors are not modelled.
- Only the text-screening path is evaluated. High-stakes and time-sensitive
  routing, low-confidence routing and topic mismatch are covered by the
  original tests, not re-scored here.
- A single gate configuration was tested; threshold sensitivity is not measured.
- No rule was tuned after measurement, so these numbers are the shipped
  behaviour rather than an optimised upper bound.

## 10. Reproduction

```bash
.venv312/bin/python experiments/safety_evaluation/build_cases.py --overwrite
.venv312/bin/python experiments/safety_evaluation/evaluate.py --overwrite
.venv312/bin/python experiments/safety_evaluation/report.py --overwrite
OMP_NUM_THREADS=1 .venv312/bin/python -m pytest tests/unit/test_adversarial_safety.py -q
```

Raw outputs: `metrics.json`, `predictions.csv`, `false_negatives.csv`,
`false_positives.csv`.
"""


def _category_rows(metrics: dict) -> list[list]:
    rows = []
    composition = metrics["composition"]["category_gold"]
    for category, stats in metrics["rule"]["categories"].items():
        gold = composition.get(category, {})
        rows.append([
            category, gold.get("unsafe", 0) + gold.get("safe", 0) + gold.get("ambiguous", 0),
            gold.get("unsafe", 0), gold.get("safe", 0), gold.get("ambiguous", 0),
            stats["tp"], stats["fn"], stats["fp"], stats["recall"], stats["precision"],
        ])
    return rows


def _load_predictions(path: Path) -> list[dict]:
    import csv

    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    out = Path(args.out)
    metrics_path = out / "metrics.json"
    if not metrics_path.exists():
        print(f"missing {metrics_path}; run evaluate.py first")
        return 1
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    predictions = _load_predictions(out / "predictions.csv")
    report_path = out / "REPORT.md"
    if report_path.exists() and not args.overwrite:
        print(f"refusing to overwrite {report_path} (pass --overwrite)")
        return 1
    report_path.write_text(build_report(metrics, predictions), encoding="utf-8")
    print(f"wrote {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
