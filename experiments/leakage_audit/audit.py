#!/usr/bin/env python
"""Row-level dataset leakage audit.

Quantifies, from the committed splits alone (no embeddings, no model), the types
of leakage the audit flagged:

* exact-question overlap between the test split and train / validation,
* answer-text overlap (the same logged answer appears on both sides),
* full record (question + answer) overlap,
* test-internal duplicate questions (an evaluation row can retrieve its twin),
* the share of test rows whose question also exists in the retrieval index
  (train + validation) — i.e. rows that could retrieve an exact duplicate of
  themselves.

Near-duplicate / group leakage is reported by the Batch A grouped evaluation
(``results/improved_evaluation/grouped_retrieval.json``) and is quoted here rather
than recomputed, since it needs the embedding index.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

PROCESSED = ROOT / "data" / "processed"
GROUPED = ROOT / "results" / "improved_evaluation" / "grouped_retrieval.json"
DEFAULT_OUT = ROOT / "results" / "leakage_audit"


def _norm(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().str.replace(r"\s+", " ", regex=True).str.strip()


def audit() -> dict:
    train = pd.read_csv(PROCESSED / "train.csv")
    val = pd.read_csv(PROCESSED / "val.csv")
    test = pd.read_csv(PROCESSED / "test.csv")
    for frame in (train, val, test):
        frame["_q"] = _norm(frame["questions"])
        frame["_a"] = _norm(frame["answers"])
        frame["_r"] = frame["_q"] + "\u0000" + frame["_a"]

    index_q = set(train["_q"]) | set(val["_q"])
    index_r = set(train["_r"]) | set(val["_r"])
    index_a = set(train["_a"]) | set(val["_a"])

    n_test = len(test)
    test_q_in_train = int(test["_q"].isin(set(train["_q"])).sum())
    test_q_in_val = int(test["_q"].isin(set(val["_q"])).sum())
    test_q_in_index = int(test["_q"].isin(index_q).sum())
    test_a_in_index = int(test["_a"].isin(index_a).sum())
    test_r_in_index = int(test["_r"].isin(index_r).sum())

    grouped = json.loads(GROUPED.read_text(encoding="utf-8")) if GROUPED.exists() else {}
    src = GROUPED.name

    metrics = {
        "protocol": {
            "source": "data/processed/{train,val,test}.csv",
            "normalisation": "lowercase, collapse whitespace",
            "note": "row-split leakage; exact/answer/record overlap recomputed, near-duplicate quoted from Batch A",
        },
        "sizes": {"train": len(train), "val": len(val), "test": n_test,
                  "index": len(train) + len(val)},
        "exact_question_leakage": {
            "test_rows": n_test,
            "test_question_in_train": test_q_in_train,
            "test_question_in_val": test_q_in_val,
            "test_question_in_index_share": round(test_q_in_index / n_test, 4),
        },
        "answer_leakage": {
            "test_rows_whose_answer_appears_in_index": test_a_in_index,
            "share": round(test_a_in_index / n_test, 4),
        },
        "record_leakage": {
            "test_rows_whose_full_record_appears_in_index": test_r_in_index,
            "share": round(test_r_in_index / n_test, 4),
        },
        "test_internal_duplicates": {
            "duplicate_question_groups": int(test["_q"].duplicated().sum()),
        },
        "grouped_reference": {
            "source": src,
            "near_duplicate_share_ge_0.95": grouped.get("test_max_dense_similarity_to_index", {}).get("ge_0.95"),
            "near_duplicate_share_ge_0.90": grouped.get("test_max_dense_similarity_to_index", {}).get("ge_0.90"),
            "novel_share_lt_0.80": grouped.get("test_max_dense_similarity_to_index", {}).get("lt_0.80"),
            "note": "after question-grouping; the row split is ~70% near-duplicate, the grouped split ~0.1%",
        },
        "interpretation": [
            "Row-level metrics overstate generalisation because the same question (often an exact string) appears in train and test.",
            "These are descriptive counts on the shipped splits, not model measurements.",
            "The grouped split (Batch A) removes near-duplicate leakage by construction and is the leakage-resistant protocol.",
        ],
    }
    return metrics


def render(metrics: dict) -> str:
    e = metrics["exact_question_leakage"]
    lines = [
        "# Dataset leakage audit",
        "",
        "**Evidence class:** descriptive recomputation from the committed splits; no model, no training.",
        "",
        f"**Sizes:** train {metrics['sizes']['train']:,}, val {metrics['sizes']['val']:,}, "
        f"test {metrics['sizes']['test']:,}, retrieval index {metrics['sizes']['index']:,}.",
        "",
        "## Row-split leakage (this split)",
        "",
        "| Measure | Count | Share of test |",
        "|---|---:|---:|",
        f"| Test question exactly in train | {e['test_question_in_train']:,} | {e['test_question_in_train']/e['test_rows']:.2%} |",
        f"| Test question exactly in val | {e['test_question_in_val']:,} | {e['test_question_in_val']/e['test_rows']:.2%} |",
        f"| Test question in index (train+val) | {e['test_question_in_index_share']*e['test_rows']:.0f} | {e['test_question_in_index_share']:.2%} |",
        f"| Test answer appears in index | {metrics['answer_leakage']['test_rows_whose_answer_appears_in_index']:,} | {metrics['answer_leakage']['share']:.2%} |",
        f"| Full test record appears in index | {metrics['record_leakage']['test_rows_whose_full_record_appears_in_index']:,} | {metrics['record_leakage']['share']:.2%} |",
        "",
        f"Test-internal duplicate questions: {metrics['test_internal_duplicates']['duplicate_question_groups']:,}.",
        "",
        "## Grouped split (Batch A reference)",
        "",
        f"- Near-duplicate (>=0.95) share: {metrics['grouped_reference']['near_duplicate_share_ge_0.95']}",
        f"- Near-duplicate (>=0.90) share: {metrics['grouped_reference']['near_duplicate_share_ge_0.90']}",
        f"- Novel (<0.80) share: {metrics['grouped_reference']['novel_share_lt_0.80']}",
        "",
        "## Interpretation",
        "",
    ]
    lines += [f"- {x}" for x in metrics["interpretation"]]
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name in ("metrics.json", "REPORT.md"):
        if (out / name).exists() and not args.overwrite:
            print(f"refusing to overwrite {out / name} (pass --overwrite)")
            return 1
    metrics = audit()
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (out / "REPORT.md").write_text(render(metrics), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
