"""Run the synthetic PII evaluation and write the Batch B Phase 1 report.

Outputs (new directories only):
    results/privacy_audit/metrics.json
    results/privacy_audit/dataset_summary.json
    results/privacy_audit/errors_balanced.csv
    results/privacy_audit/REPORT.md

The evaluation is synthetic.  No real personal information is included in the
dataset, metrics, errors or report.  Error rows include only redacted text and
entity types/spans, never detected values.
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
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.batch_b.common import environment, rel, utc_now  # noqa: E402
from experiments.privacy.audit import DATA_PATH, evaluate_both_policies, load_rows  # noqa: E402
from experiments.privacy.recognizers import detect  # noqa: E402
from experiments.privacy.redact import redact_spans  # noqa: E402


def _dataset_summary(rows: list[dict]) -> dict:
    languages = Counter(row["language"] for row in rows)
    types = Counter()
    negatives = 0
    for row in rows:
        if not row["entities"]:
            negatives += 1
        for entity in row["entities"]:
            types[entity["type"]] += 1
    return {
        "path": rel(DATA_PATH),
        "n_examples": len(rows),
        "synthetic": True,
        "languages": dict(sorted(languages.items())),
        "entity_counts": dict(sorted(types.items())),
        "negative_or_ambiguous_examples": negatives,
    }


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def _table(headers: list[str], rows: list[list[str]]) -> list[str]:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return lines


def _write_errors(path: Path, errors: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "id",
                "language",
                "gold_types",
                "pred_types",
                "false_positive_spans",
                "false_negative_spans",
                "redacted_text",
                "notes",
            ],
        )
        writer.writeheader()
        for error in errors:
            writer.writerow(
                {
                    "id": error["id"],
                    "language": error["language"],
                    "gold_types": ";".join(error["gold_types"]),
                    "pred_types": ";".join(error["pred_types"]),
                    "false_positive_spans": ";".join(error["false_positive_spans"]),
                    "false_negative_spans": ";".join(error["false_negative_spans"]),
                    "redacted_text": error["redacted_text"],
                    "notes": error["notes"],
                }
            )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA_PATH)
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "privacy_audit")
    parser.add_argument("--overwrite", action="store_true", help="replace existing Batch B Phase 1 outputs")
    args = parser.parse_args()

    out = args.out
    if out.exists() and not args.overwrite:
        print(f"refusing to overwrite {rel(out)}; pass --overwrite to regenerate")
        return 1
    out.mkdir(parents=True, exist_ok=True)

    rows = load_rows(args.data)
    summary = _dataset_summary(rows)
    results = evaluate_both_policies(rows)
    results["created_at"] = utc_now()
    results["environment"] = environment()
    results["dataset"] = summary

    balanced = results["policies"]["balanced"]
    strict = results["policies"]["strict"]

    # Add redacted text to errors for the CSV only; never add raw values.
    for policy in ("balanced", "strict"):
        for error in results["policies"][policy]["errors"]:
            row = next(r for r in rows if r["id"] == error["id"])
            # Mask both detected and gold spans.  Gold masking keeps even a
            # missed entity out of the report, so no raw PII reaches logs.
            spans = [(d.start, d.end, d.type) for d in detect(row["text"], policy=policy)]
            spans += [(e["start"], e["end"], e["type"]) for e in row["entities"]]
            error["redacted_text"] = redact_spans(row["text"], spans)

    (out / "metrics.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (out / "dataset_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    _write_errors(out / "errors_balanced.csv", balanced["errors"])

    lines: list[str] = []
    lines.append("# PII detection and redaction — synthetic evaluation (Batch B, Phase 1)")
    lines.append("")
    lines.append(
        "All examples in this report are **synthetic**. They were authored for "
        "this evaluation and contain fake names, numbers and `example.com` "
        "addresses. No real farmer data, annotation or expert review was used. "
        "These numbers do **not** establish production readiness or real-world "
        "PII recall."
    )
    lines.append("")
    lines.append("## Protocol")
    lines.append("")
    lines.append(
        "- Dataset: `experiments/privacy/data/synthetic_pii_eval.csv` "
        f"({summary['n_examples']} examples; "
        f"{summary['languages']}; {summary['negative_or_ambiguous_examples']} "
        "negative/ambiguous)."
    )
    lines.append(
        "- Matching: strict exact character span **and** entity type. A partial "
        "name match is counted as one false positive and one false negative."
    )
    lines.append(
        "- Two policies: `balanced` accepts a context-free 10-digit mobile-like "
        "number; `strict` requires a phone cue, separator, +91/00 prefix or "
        "international prefix."
    )
    lines.append(
        "- Bootstrap: 2,000 document resamples, seed 0, percentile 95% CI for "
        "micro F1."
    )
    lines.append("- Random seed: 0. No model training and no new dependency.")
    lines.append("")

    lines.append("## Overall")
    lines.append("")
    overall_rows = []
    for name, result in (("balanced", balanced), ("strict", strict)):
        o = result["overall"]
        ci = o["f1_95ci"]
        overall_rows.append(
            [
                name,
                str(o["tp"]),
                str(o["fp"]),
                str(o["fn"]),
                _fmt(o["precision"]),
                _fmt(o["recall"]),
                _fmt(o["f1"]),
                f"[{ci['low']:.4f}, {ci['high']:.4f}]",
                _fmt(o["exact_document_rate"]),
                _fmt(o["false_positive_rate_document_level"]),
            ]
        )
    lines.extend(
        _table(
            [
                "Policy",
                "TP",
                "FP",
                "FN",
                "Precision",
                "Recall",
                "F1",
                "F1 95% CI",
                "Exact docs",
                "Doc-level FPR",
            ],
            overall_rows,
        )
    )
    lines.append("")
    lines.append(
        f"Balanced predicted **{balanced['overall']['predicted_entities']}** "
        f"entities against **{balanced['overall']['gold_entities']}** gold entities. "
        f"The synthetic set has {balanced['overall']['negative_documents']} "
        "documents with no PII; "
        f"{balanced['overall']['negative_documents_with_false_positive']} of them "
        "produced a false positive."
    )
    lines.append("")

    lines.append("## Entity-wise performance (balanced policy)")
    lines.append("")
    type_rows = []
    for kind, metrics in balanced["per_type"].items():
        type_rows.append(
            [
                kind,
                str(metrics["gold"]),
                str(metrics["pred"]),
                str(metrics["tp"]),
                str(metrics["fp"]),
                str(metrics["fn"]),
                _fmt(metrics["precision"]),
                _fmt(metrics["recall"]),
                _fmt(metrics["f1"]),
            ]
        )
    lines.extend(
        _table(
            ["Entity", "Gold", "Predicted", "TP", "FP", "FN", "Precision", "Recall", "F1"],
            type_rows,
        )
    )
    lines.append("")

    lines.append("## Language-wise performance (balanced policy)")
    lines.append("")
    language_rows = []
    for language, metrics in sorted(balanced["per_language"].items()):
        language_rows.append(
            [
                language,
                str(metrics["documents"]),
                str(metrics["negative_documents"]),
                _fmt(metrics["precision"]),
                _fmt(metrics["recall"]),
                _fmt(metrics["f1"]),
                _fmt(metrics["fpr_document_level"]),
            ]
        )
    lines.extend(
        _table(
            ["Language", "Documents", "Negative docs", "Precision", "Recall", "F1", "Doc-level FPR"],
            language_rows,
        )
    )
    lines.append("")
    lines.append(
        "Language labels are author-assigned (`en`, `hinglish`, `hi`) and describe "
        "the synthetic sentence, not a validated language identification result."
    )
    lines.append("")

    lines.append("## False positives")
    lines.append("")
    false_positives = [e for e in balanced["errors"] if e["false_positive_spans"]]
    if false_positives:
        lines.extend(
            _table(
                ["ID", "Language", "Gold", "Predicted", "FP spans", "Redacted text"],
                [
                    [
                        e["id"],
                        e["language"],
                        ", ".join(e["gold_types"]) or "none",
                        ", ".join(e["pred_types"]) or "none",
                        ", ".join(e["false_positive_spans"]),
                        e["redacted_text"],
                    ]
                    for e in false_positives
                ],
            )
        )
    else:
        lines.append("No false-positive spans were observed.")
    lines.append("")
    if balanced["overall"]["negative_documents_with_false_positive"] == 0:
        lines.append(
            "No negative/ambiguous document produced any detection under the "
            "balanced policy. This is a synthetic-set result; it does not mean "
            "the detector will have zero false positives on real text."
        )
    lines.append("")

    lines.append("## False negatives and known failures")
    lines.append("")
    false_negatives = [e for e in balanced["errors"] if e["false_negative_spans"]]
    if false_negatives:
        lines.extend(
            _table(
                ["ID", "Language", "Gold", "Predicted", "FN spans", "Redacted text", "Note"],
                [
                    [
                        e["id"],
                        e["language"],
                        ", ".join(e["gold_types"]) or "none",
                        ", ".join(e["pred_types"]) or "none",
                        ", ".join(e["false_negative_spans"]),
                        e["redacted_text"],
                        e["notes"],
                    ]
                    for e in false_negatives
                ],
            )
        )
    else:
        lines.append("No false negatives were observed.")
    lines.append("")
    lines.append(
        "The hard cases are intentional. They show that the name heuristics do "
        "not handle initials, lowercase honorifics or lowercase surnames, and "
        "that dotted and trunk-zero phone formats are not recognised."
    )
    lines.append("")

    lines.append("## What this establishes")
    lines.append("")
    lines.append(
        "- On **this synthetic set**, the balanced recogniser finds most declared "
        f"entities (F1 {balanced['overall']['f1']:.4f}, 95% CI "
        f"[{balanced['overall']['f1_95ci']['low']:.4f}, "
        f"{balanced['overall']['f1_95ci']['high']:.4f}])."
    )
    lines.append(
        "- The negative/ambiguous examples did not trigger phone redaction, "
        "including plot/token/order/price/quantity contexts."
    )
    lines.append(
        "- Aadhaar-like values are only accepted with a valid Verhoeff checksum "
        "or an explicit Aadhaar cue, and a bank cue takes precedence for account "
        "numbers."
    )
    lines.append("")

    lines.append("## What this does not establish")
    lines.append("")
    lines.append(
        "- It does **not** measure real-world PII prevalence, recall or precision. "
        "The dataset is small, synthetic and authored by the same person who "
        "wrote the recognisers. Perfectly covered categories reflect that shared "
        "authorship, not independent validation."
    )
    lines.append(
        "- **Names are best-effort and anchor-based.** Unanchored names (e.g. "
        "`Rahul said ...`) are deliberately not detected; this is a recall gap."
    )
    lines.append(
        "- **Phone format coverage is incomplete.** Dot-grouped numbers, trunk "
        "prefix `0`, and some other real formats are missed."
    )
    lines.append(
        "- Hindi/Hinglish handling is regex- and digit-normalisation-based. It "
        "does not use a trained NER model, gazetteer or Presidio. Devanagari "
        "digit normalisation is 1:1 and preserves offsets."
    )
    lines.append(
        "- No real corpus was scanned in this phase, so no claim is made about "
        "how much PII exists in the historical KCC-Sahayak corpus."
    )
    lines.append(
        "- Aadhaar detection is identifier-shaped, not an identity verification "
        "system. An invalid test number with an explicit cue is still redacted."
    )
    lines.append(
        "- Recall cannot be guaranteed. Redaction is a mitigation, not a "
        "guarantee that no personal information remains."
    )
    lines.append("")

    lines.append("## Reproduction")
    lines.append("")
    lines.append("```bash")
    lines.append(".venv312/bin/python experiments/privacy/build_eval_set.py")
    lines.append(".venv312/bin/python experiments/privacy/run_evaluation.py --overwrite")
    lines.append(".venv312/bin/python -m pytest tests/unit/test_privacy.py -q")
    lines.append(".venv312/bin/python experiments/batch_b/verify_baseline.py --check")
    lines.append("```")
    lines.append("")
    lines.append(
        "Metrics: `results/privacy_audit/metrics.json`. Error rows: "
        "`results/privacy_audit/errors_balanced.csv` (redacted text only)."
    )
    lines.append("")
    lines.append(f"Generated: {results['created_at']}")
    lines.append("")

    (out / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"wrote {rel(out / 'metrics.json')}")
    print(f"wrote {rel(out / 'REPORT.md')}")
    print(f"wrote {rel(out / 'errors_balanced.csv')}")
    print(
        "balanced: "
        f"P={balanced['overall']['precision']:.4f} "
        f"R={balanced['overall']['recall']:.4f} "
        f"F1={balanced['overall']['f1']:.4f} "
        f"CI=[{balanced['overall']['f1_95ci']['low']:.4f}, "
        f"{balanced['overall']['f1_95ci']['high']:.4f}]"
    )
    print(
        "strict:   "
        f"P={strict['overall']['precision']:.4f} "
        f"R={strict['overall']['recall']:.4f} "
        f"F1={strict['overall']['f1']:.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
