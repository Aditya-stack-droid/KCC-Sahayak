#!/usr/bin/env python
"""Answer-quality evaluation framework.

The project has retrieval and answer-similarity evaluation but no measure of
whether an answer is *correct*. This framework separates the dimensions and, for
each, records what is actually measured, the artefact it comes from, and — where
the required ground truth does not exist — says so instead of inventing a score.

Dimensions and status (``measured`` / ``proxy`` / ``unavailable``):

1. Retrieval relevance          measured  results/improved_evaluation/grouped_retrieval.json
2. Answer completeness          proxy     results/multi_reference/per_query_dense_all.csv
3. Semantic similarity          proxy     results/multi_reference/per_query_dense_all.csv
4. Factual consistency          proxy     results/multi_reference/per_query_dense_all.csv
5. Crop/entity consistency      unavailable  (no per-row entity ground truth)
6. Dosage consistency           proxy     results/conflict_detection/metrics.json
7. Unit consistency             proxy     results/conflict_detection/metrics.json
8. Safety compliance            measured  results/answer_screening/metrics.json
9. Escalation correctness       proxy     results/selective_eval.json
-- Agronomic correctness        unavailable  (no agronomist/expert labels)

No automated score here is agricultural correctness.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DEFAULT_OUT = ROOT / "results" / "answer_quality"

RESULTS = ROOT / "results"


def _load(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _pct(x):
    return None if x is None else round(100 * x, 2)


def build() -> dict:
    grouped = _load(RESULTS / "improved_evaluation" / "grouped_retrieval.json") or {}
    dense = (grouped.get("dense_all_test") or {})
    per_query = RESULTS / "multi_reference" / "per_query_dense_all.csv"
    conflict = _load(RESULTS / "conflict_detection" / "metrics.json") or {}
    screening = _load(RESULTS / "answer_screening" / "metrics.json") or {}
    selective = _load(RESULTS / "selective_eval.json") or {}

    # Completeness / similarity / consistency from the per-query file.
    own_f1, max_other, max_sem, low_support = [], [], [], 0
    n = 0
    if per_query.exists():
        with per_query.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                n += 1
                try:
                    f1 = float(row["own_f1"])
                except (KeyError, ValueError):
                    continue
                own_f1.append(f1)
                if f1 < 0.2:
                    low_support += 1
                if row.get("max_other_f1"):
                    max_other.append(float(row["max_other_f1"]))
                if row.get("max_semantic"):
                    max_sem.append(float(row["max_semantic"]))

    def mean(xs):
        return round(sum(xs) / len(xs), 4) if xs else None

    dims = {
        "retrieval_relevance": {
            "status": "measured",
            "source": "results/improved_evaluation/grouped_retrieval.json",
            "what": "topical retrieval quality on the grouped test split (weak-label relevance)",
            "metrics": {
                "recall_at_1": dense.get("recall_at_1"),
                "recall_at_5": dense.get("recall_at_5"),
                "mrr_at_10": dense.get("mrr_at_10"),
                "ndcg_at_5": dense.get("ndcg_at_5"),
            },
        },
        "answer_completeness": {
            "status": "proxy",
            "source": "results/multi_reference/per_query_dense_all.csv",
            "what": "token overlap between the retrieved answer and the row's own logged answer",
            "metrics": {"mean_own_f1": mean(own_f1), "rows": len(own_f1)},
        },
        "semantic_similarity": {
            "status": "proxy",
            "source": "results/multi_reference/per_query_dense_all.csv",
            "what": "max embedding similarity to any reference answer (wording closeness, NOT correctness)",
            "metrics": {"mean_max_semantic": mean(max_sem)},
        },
        "factual_consistency": {
            "status": "proxy",
            "source": "results/multi_reference/per_query_dense_all.csv",
            "what": "agreement with other historical answers to the same question (agreement, not truth)",
            "metrics": {"mean_max_other_f1": mean(max_other)},
        },
        "crop_entity_consistency": {
            "status": "unavailable",
            "reason": "no per-row crop/entity ground truth for retrieved answers",
        },
        "dosage_consistency": {
            "status": "proxy",
            "source": "results/conflict_detection/metrics.json",
            "what": "rule-fired dosage conflicts between answers to the same question (rule match, not truth)",
            "metrics": {"pair_precision": (conflict.get("binary") or {}).get("precision"),
                        "pair_recall": (conflict.get("binary") or {}).get("recall")},
        },
        "unit_consistency": {
            "status": "proxy",
            "source": "results/conflict_detection/metrics.json",
            "what": "basis/dimension conflicts detected by the same rules",
            "metrics": {"basis_f1": (conflict.get("per_kind") or {}).get("basis", {}).get("f1"),
                        "dimension_f1": (conflict.get("per_kind") or {}).get("dimension", {}).get("f1")},
        },
        "safety_compliance": {
            "status": "measured",
            "source": "results/answer_screening/metrics.json",
            "what": "content safety rule agreement on authored cases (v1 shipped vs v2 candidate)",
            "metrics": {"v1_f1": (screening.get("v1") or {}).get("overall", {}).get("f1"),
                        "v2_f1": (screening.get("v2") or {}).get("overall", {}).get("f1")},
        },
        "escalation_correctness": {
            "status": "proxy",
            "source": "results/selective_eval.json",
            "what": "coverage / selective intent accuracy / rule-defined violation rate of the gate",
            "metrics": {"coverage": (selective.get("silver_default") or {}).get("coverage"),
                        "selective_accuracy": (selective.get("silver_default") or {}).get("selective_accuracy"),
                        "violation_rate": (selective.get("gate_ablation_silver") or {}).get("5_+topic_match(full)", {}).get("violation_rate_among_auto")},
        },
        "unsupported_claim_rate": {
            "status": "proxy",
            "source": "results/multi_reference/per_query_dense_all.csv",
            "what": "share of rows whose retrieved answer barely overlaps its own logged answer (own_f1 < 0.2)",
            "metrics": {"share": round(low_support / len(own_f1), 4) if own_f1 else None,
                        "definition": "own_f1 < 0.2; a low-support proxy, not a factual check"},
        },
        "agronomic_correctness": {
            "status": "unavailable",
            "reason": "no agronomist or expert labels exist; correctness cannot be measured from the corpus",
            "next_step": "add verified reference labels and re-run; the framework slots them in as 'measured'",
        },
    }

    return {
        "protocol": {
            "sources": "committed result artefacts only; no new model run",
            "no_fabrication": "dimensions without ground truth are marked unavailable rather than scored",
        },
        "dimensions": dims,
        "headline": {
            "retrieval_topic_match_at_1": dense.get("category_match_top1"),
            "answer_own_token_f1": mean(own_f1),
            "answer_max_other_token_f1": mean(max_other),
            "unsupported_claim_rate_proxy": round(low_support / len(own_f1), 4) if own_f1 else None,
            "agronomic_correctness": "unavailable",
        },
        "warnings": [
            "Similarity, agreement and token overlap are NOT agricultural correctness.",
            "No agronomist has reviewed any answer; no expert labels exist.",
            "Proxy metrics use historical advisor answers as references, which may be contradictory or outdated.",
        ],
    }


def render(report: dict) -> str:
    lines = [
        "# Answer-quality evaluation framework",
        "",
        "**Evidence class:** aggregation of committed artefacts; no new model run, no expert labels.",
        "Each dimension is labelled **measured**, **proxy**, or **unavailable**; unavailable",
        "dimensions are not scored.",
        "",
        "| Dimension | Status | What it measures | Key metric | Source |",
        "|---|---|---|---|---|",
    ]
    for name, d in report["dimensions"].items():
        metric = "; ".join(f"{k}={v}" for k, v in d.get("metrics", {}).items() if v is not None) or "—"
        what = d.get("what") or d.get("reason", "")
        lines.append(f"| {name} | {d['status']} | {what} | {metric} | {d.get('source', '—')} |")
    lines += ["", "## Headline", ""]
    for k, v in report["headline"].items():
        lines.append(f"- **{k}**: {v}")
    lines += ["", "## Warnings", ""]
    lines += [f"- {w}" for w in report["warnings"]]
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name in ("framework.json", "REPORT.md"):
        if (out / name).exists() and not args.overwrite:
            print(f"refusing to overwrite {out / name} (pass --overwrite)")
            return 1
    report = build()
    (out / "framework.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (out / "REPORT.md").write_text(render(report), encoding="utf-8")
    print(json.dumps(report["headline"], indent=2))
    print("written:", out / "framework.json", out / "REPORT.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
