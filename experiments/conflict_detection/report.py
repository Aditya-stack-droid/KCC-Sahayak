#!/usr/bin/env python
"""Write the combined conflict-detection report from measured JSON outputs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DEFAULT_OUT = ROOT / "results" / "conflict_detection"


def _table(header: list[str], rows: list[list]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    for row in rows:
        lines.append("| " + " | ".join("—" if value is None else str(value) for value in row) + " |")
    return "\n".join(lines)


def build_report(metrics: dict, corpus: dict | None) -> str:
    binary = metrics["binary"]
    kind_rows = []
    for kind, stats in metrics["per_kind"].items():
        kind_rows.append([
            kind, stats["support"], stats["tp"], stats["fp"], stats["fn"],
            f"{stats['precision']:.4f}", f"{stats['recall']:.4f}", f"{stats['f1']:.4f}",
        ])

    subsets_rows = [
        [subset, stats["n"], f"{stats['precision']:.4f}", f"{stats['recall']:.4f}",
         f"{stats['f1']:.4f}", f"{stats['accuracy']:.4f}"]
        for subset, stats in metrics.get("subsets", {}).items()
    ]

    corpus_section = "Not run (run `run_corpus.py`)."
    if corpus:
        bucket_rows = [
            [bucket, info["groups"], info["groups_with_conflict"], info["rule_fire_rate"]]
            for bucket, info in sorted(corpus["by_group_size"].items())
        ]
        corpus_section = "\n\n".join([
            f"- Eligible multi-answer groups: **{corpus['eligible_groups']:,}**",
            f"- Sampled groups: **{corpus['sampled_groups']:,}** (seed {corpus['protocol']['seed']})",
            f"- Answer pairs compared: **{corpus['pairs_evaluated']:,}**",
            f"- Pairs where a rule fired: **{corpus['pairs_with_conflict']:,}** "
            f"({corpus['pair_rule_fire_rate']:.2%})",
            f"- Groups with at least one flagged pair: **{corpus['groups_with_conflict']:,}** "
            f"({corpus['group_rule_fire_rate']:.2%})",
            "Kinds (pairs with at least one such conflict): "
            + ", ".join(f"`{kind}` {count}" for kind, count in sorted(corpus["pairs_by_kind"].items())),
            "",
            _table(["Informative answers in group", "Groups", "Flagged groups", "Fire rate"], bucket_rows),
            "",
            f"The redacted review queue (`group_review.csv`) contains {corpus['review_rows']} rows. "
            "It is a candidate list for a human reviewer, not a set of confirmed errors.",
        ])

    return f"""# Answer conflict detection (Batch B, Improvement 1)

**Status:** implemented and measured on an authored synthetic pair set plus a
descriptive corpus scan. **No expert adjudication exists**, and a rule firing is
not proof that an answer is wrong.

The module never merges, averages or resolves advice. When any conflict is
found, the optional wrapper turns an auto-answer into **ESCALATE** for a human
advisor; the production gate is imported read-only and remains unchanged.

## 1. What is detected

| Kind | Severity | Rule |
|---|---|---|
| `product` | high | both answers name products and the product sets are disjoint |
| `polarity` | high | a product is negated in one answer and recommended in the other (clause-local negation) |
| `dosage` | high / medium | same product dimension and basis, but even the closest pair of amounts differs by ≥ 50% |
| `basis` | medium | doses are expressed on different non-convertible bases (acre vs bigha; plant vs acre) |
| `dimension` | medium | the same product is dosed in incompatible units (kg vs litre) |
| `interval` | high / medium | application intervals differ by ≥ 50% |

Unit handling: g/kg/quintal/tonne and ml/litre are normalised; hectare is
converted to acre (1 ha = 2.4710538 acre); bigha/kanal/guntha are **not**
converted because their size varies by district. `per N litre` and `per N acre`
are normalised to per-one-unit. Prices, plot/token ids and pack sizes are
excluded from quantity extraction.

Product handling: spelling and abbreviation variants only (`D.A.P.`, `2,4-D`,
`MOP`/`potash`). Brands are **not** mapped to active ingredients, so `rogor` and
`dimethoate` count as different products. This is a documented limitation, not a
claim of chemical equivalence.

## 2. Synthetic pair evaluation

- Pairs: **{metrics['n_pairs']}** ({metrics['n_gold_conflict']} conflict,
  {metrics['n_gold_no_conflict']} no-conflict), hand-authored and **synthetic**.
- Matching: strict pair-level label; per-kind metrics treat each kind as a binary
  label over all pairs. Dose tolerance: {metrics['protocol']['dose_gap_tolerance']}.
- Bootstrap: {metrics['protocol']['bootstrap']['n_boot']} resamples, seed
  {metrics['protocol']['bootstrap']['seed']}.

| TP | FP | FN | TN | Precision | Recall | F1 | F1 95% CI | Accuracy |
|---:|---:|---:|---:|---:|---:|---:|---|---:|
| {binary['tp']} | {binary['fp']} | {binary['fn']} | {binary['tn']} | {binary['precision']:.4f} |
{binary['recall']:.4f} | {binary['f1']:.4f} | [{binary['f1_95ci']['low']}, {binary['f1_95ci']['high']}] |
{binary['accuracy']:.4f} |

Per-kind:

{_table(["Kind", "Support", "TP", "FP", "FN", "Precision", "Recall", "F1"], kind_rows)}

By authored subset (the challenge cases were written after the core rules
existed, to expose likely failure modes; they are still authored, not a held-out
benchmark):

{_table(["Subset", "Pairs", "Precision", "Recall", "F1", "Accuracy"], subsets_rows)}

Small support (especially dimension/basis) means wide uncertainty; these
per-kind values are not stable estimates.

## 3. Errors found

{chr(10).join(f"- `{row['pair_id']}` gold={row['gold_kinds']} predicted={row['predicted_kinds']}" for row in metrics['errors'][:20]) or "- None."}

## 4. Descriptive corpus scan

Reuses Batch A's grouped metadata and informative mask. It reports how often the
rules fire, not how often advice is actually wrong.

{corpus_section}

## 5. Escalation behaviour

`decide_with_conflicts` runs the production gate on the top-ranked answer and
adds the conflict rule:

- no conflict: the production decision is returned unchanged;
- conflict: the decision becomes `ESCALATE`, with machine-readable reasons
  (`ANSWER_CONFLICT`, `DOSAGE_CONFLICT`, …), even for a category the production
  gate would auto-answer;
- an existing production ESCALATE keeps its original code as `base_code`.

The wrapper returns no suggested or merged answer. Focused tests cover these
three cases.

## 6. Limitations

- The synthetic pair set is small, authored by the same person who wrote the
  rules, and not expert-verified. Precision/recall are not real-world estimates.
- Corpus flags are rule-fire rates. Group members may discuss different crops or
  stages, which the current rules do not model as context.
- Brand/active-ingredient equivalence is not inferred; `rogor` vs `dimethoate`
  will be reported as a product difference.
- Negation is clause-local and pattern-based; conditional advice (“if aphids
  persist, then spray…”) is only partially handled.
- Dosages within 50% are treated as compatible. This tolerance is a deliberate
  trade-off to limit false positives and is not agronomically validated.
- Only the first five answers are compared, and the corpus scan covers the
  grouped train+val side only.
- No integration into the backend or production gate.

## 7. Reproduction

```bash
.venv312/bin/python experiments/conflict_detection/build_pairs.py
.venv312/bin/python experiments/conflict_detection/evaluate.py --overwrite
.venv312/bin/python experiments/conflict_detection/run_corpus.py --overwrite
.venv312/bin/python experiments/conflict_detection/report.py --overwrite
OMP_NUM_THREADS=1 .venv312/bin/python -m pytest tests/unit/test_conflict_detection.py -q
```
"""


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
    corpus_path = out / "corpus_scan.json"
    corpus = json.loads(corpus_path.read_text(encoding="utf-8")) if corpus_path.exists() else None
    report_path = out / "REPORT.md"
    if report_path.exists() and not args.overwrite:
        print(f"refusing to overwrite {report_path} (pass --overwrite)")
        return 1
    report_path.write_text(build_report(metrics, corpus), encoding="utf-8")
    print(f"wrote {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
