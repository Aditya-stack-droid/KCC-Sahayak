#!/usr/bin/env python
"""Render the Project B comparative report from the saved metrics."""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BASE = ROOT / "results" / "retrieval_baselines"


def _load(p: Path):
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def _f(v, nd=4):
    return "n/a" if v is None else f"{v:.{nd}f}"


def _system_rows(cmp_metrics: dict) -> list[str]:
    rows = ["| System | topic@1 | Recall@5 | MRR@10 | nDCG@5 | exact@1 | answer F1@1 |",
            "|---|---:|---:|---:|---:|---:|---:|"]
    order = ["dense", "tfidf", "hybrid", "bm25", "dense_pool20", "dense_ce",
             "bm25_pool20", "bm25_ce", "hybrid_pool20", "hybrid_ce"]
    systems = cmp_metrics["systems"]
    for name in order:
        if name not in systems:
            continue
        s = systems[name]
        rows.append(f"| {name} | {s['topic_match_at_1']:.4f} | {s['recall_at_5']:.4f} | "
                    f"{s['mrr_at_10']:.4f} | {s['ndcg_at_5']:.4f} | {s['exact_match_at_1']:.4f} | "
                    f"{s['answer_token_f1_at_1']:.4f} |")
    return rows


def main() -> int:
    cmp_metrics = _load(BASE / "comparative_evaluation" / "metrics.json")
    bm25 = _load(BASE / "bm25" / "metrics.json")
    ce = _load(BASE / "cross_encoder" / "metrics.json")
    ft = _load(BASE / "fine_tuning" / "metrics.json")
    if not cmp_metrics:
        print("run compare.py first")
        return 1

    lines = [
        "# Project B — advanced retrieval baselines",
        "",
        "**Evidence class:** evaluation on the Batch A grouped split (leakage-resistant),",
        "relevance = weak-label topic match (a topical proxy, **not** agricultural correctness).",
        "No production index, model or threshold was modified. Nothing is integrated.",
        "",
        f"Sample: **{cmp_metrics['protocol']['sample']}** grouped-test queries (seed "
        f"{cmp_metrics['protocol']['seed']}); candidate pool for fusion/reranking: "
        f"{cmp_metrics['protocol']['candidate_pool']}.",
        "",
        "## 1. Comparative results",
        "",
    ]
    lines += _system_rows(cmp_metrics)
    lines += [
        "",
        "`*_pool20` is the un-reranked candidate pool (same 20 candidates); `*_ce` is the same",
        "pool reranked by the cross-encoder. Exact-match@1 is 0.0 by construction (grouped split",
        "removes exact duplicates).",
        "",
        "## 2. Paired statistical comparison (vs dense)",
        "",
        "| System | Δ topic@1 [95% CI] | Δ Recall@5 [95% CI] | Δ MRR [95% CI] | Holm-adj p (R@5) |",
        "|---|---|---|---|---:|",
    ]
    holm = cmp_metrics.get("holm_adjusted_p_recall_at_5", {})
    comps = cmp_metrics.get("paired_vs_dense", {})
    for name in ["tfidf", "hybrid", "bm25", "dense_ce", "bm25_ce", "hybrid_ce"]:
        if name not in comps:
            continue
        c = comps[name]
        cell = lambda m: f"{c[m]['mean_diff']:+.4f} {c[m]['ci95']}"
        lines.append(f"| {name} | {cell('topic_hit_at_1')} | {cell('recall_at_5')} | "
                     f"{cell('mrr')} | {holm.get(name, 'n/a')} |")
    lines += [
        "",
        "Paired bootstrap, 2,000 resamples, same queries for both systems; Holm across the",
        "comparisons on Recall@5.",
        "",
        "## 3. Cross-encoder reranking (top-20 pool)",
        "",
    ]
    if ce:
        t = ce["timing_seconds"]
        lines += [
            f"- Model: `{ce['protocol']['model']}` (Apache-2.0), pool {ce['protocol']['pool']}.",
            f"- Model load {t['model_load']}s; reranking {t['rerank_per_query_ms']} ms per query per pool.",
            "",
            "| System | topic@1 | Recall@5 | MRR@10 | nDCG@5 |",
            "|---|---:|---:|---:|---:|",
        ]
        for name in ("dense_ce", "bm25_ce", "hybrid_ce"):
            s = ce["systems"][name]["metrics"]
            lines.append(f"| {name} | {s['topic_match_at_1']:.4f} | {s['recall_at_5']:.4f} | "
                         f"{s['mrr_at_10']:.4f} | {s['ndcg_at_5']:.4f} |")
    else:
        lines.append("_Cross-encoder run not found._")
    lines += ["", "## 4. Fine-tuned encoder", ""]
    if ft:
        feas = ft["feasibility"]
        lines += [
            f"- Status: **{ft['status']}**.",
            f"- Supervision: {feas['supervision']}.",
            f"- Grouped-train rows: {feas['grouped_train_rows']:,}; informative: "
            f"{feas['informative_train_rows']:,}; call-log-note share: {feas['call_log_note_share']:.2%}.",
            f"- Suitability: {feas['suitability']}.",
        ]
        if ft["status"] == "failed":
            lines += [f"- Blocker: {ft.get('blocker', ft.get('error'))}"]
        elif ft["status"] == "completed":
            b = ft["systems"]["base"]
            f = ft["systems"]["fine_tuned"]
            lines += ["", "| System | topic@1 | Recall@5 | Recall@10 | MRR@10 |", "|---|---:|---:|---:|---:|",
                      f"| base | {b['topic_match_at_1']:.4f} | {b['recall_at_5']:.4f} | {b['recall_at_10']:.4f} | {b['mrr_at_10']:.4f} |",
                      f"| fine-tuned | {f['topic_match_at_1']:.4f} | {f['recall_at_5']:.4f} | {f['recall_at_10']:.4f} | {f['mrr_at_10']:.4f} |",
                      "", f"Paired Δ topic@1: {ft['paired_vs_base_topic_at_1']}"]
    lines += ["", "## 5. Computational cost", ""]
    if bm25:
        lines += [f"- BM25: build {bm25['build']['build_seconds']}s over "
                  f"{bm25['build']['n_docs']:,} docs, vocab {bm25['build']['vocab']:,}; "
                  f"query p50 {bm25['latency_ms']['p50']} ms, p95 {bm25['latency_ms']['p95']} ms (CPU)."]
    if ce:
        t = ce["timing_seconds"]
        lines += [f"- Cross-encoder: CPU, load {t['model_load']}s, "
                  f"{t['rerank_per_query_ms']} ms/query/pool (top-20)."]
    lat = cmp_metrics.get("latency_seconds", {})
    lines += [f"- Batch retrieval timings (2,000 queries): dense {lat.get('dense_seconds')}s, "
              f"TF-IDF {lat.get('tfidf_seconds')}s, hybrid fusion {lat.get('hybrid_seconds')}s (CPU).",
              "- No GPU was used; no GPU timings are reported.",
              "",
              "## 6. Limitations", ""]
    lines += [f"- {x}" for x in cmp_metrics.get("limitations", [])]
    if ce:
        lines += [f"- {x}" for x in ce.get("limitations", [])]
    lines.append("")

    out = BASE / "REPORT.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
