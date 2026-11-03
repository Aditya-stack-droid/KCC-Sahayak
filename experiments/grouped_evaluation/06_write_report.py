"""Combine the measured Batch A outputs into one report.

Reads only files produced by steps 02-05. Missing inputs are reported as
pending; nothing is estimated or filled in.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import ART, OUT  # noqa: E402


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def main() -> None:
    groups = load(ART / "groups_stats.json")
    intent = load(OUT / "grouped_intent.json")
    retrieval = load(OUT / "grouped_retrieval.json")

    lines = [
        "# Batch A — leakage-resistant evaluation: combined report", "",
        "All numbers below are read from the JSON files produced by this pipeline. "
        "No original result file was modified.", "",
        "## 1. Question grouping and original-split leakage", "",
    ]
    if groups:
        ls = groups["original_split_leakage"]
        lines += [
            f"- Corpus rows: **{groups['n_rows']:,}** in **{groups['n_groups']:,}** groups "
            f"(threshold {groups['near_dup_threshold']}, k={groups['knn_k']}).",
            f"- Largest group: **{groups['largest_group']:,}** rows; p95 size {groups['group_size_p95']}, "
            f"p99 {groups['group_size_p99']}.",
            f"- Exact test question present in train: **{ls['exact_test_question_in_train']:.2%}**.",
            f"- Test query shares a group with a train query: "
            f"**{ls['test_in_group_with_train']['share_test_group_leaking']:.2%}**",
            f"- Test query has a >=0.95 dense neighbour in the shipped index: "
            f"**{ls['test_max_dense_similarity_to_train_val']['ge_0.95']:.2%}**.",
            f"- Test query has no >=0.80 neighbour (novel): "
            f"**{ls['test_max_dense_similarity_to_train_val']['lt_0.80']:.2%}**.", "",
        ]
    else:
        lines += ["_Pending: run `02_build_groups.py`._", ""]

    lines += ["## 2. Grouped intent evaluation", ""]
    if intent:
        a = intent["aggregate"]
        lines += [
            "| Model | Accuracy (mean ± sd, 5 seeds) | Macro F1 | Weighted F1 |",
            "|---|---:|---:|---:|",
            f"| Refit, embedding + char TF-IDF + LR (augmentation) | "
            f"{a['accuracy']['mean']:.4f} ± {a['accuracy']['sd']:.4f} | "
            f"{a['macro_f1']['mean']:.4f} ± {a['macro_f1']['sd']:.4f} | "
            f"{a['weighted_f1']['mean']:.4f} ± {a['weighted_f1']['sd']:.4f} |",
            f"| Refit, same recipe without paraphrases | "
            f"{a['no_augmentation_accuracy']['mean']:.4f} ± "
            f"{a['no_augmentation_accuracy']['sd']:.4f} | — | — |", "",
            "Novelty buckets (refit with augmentation, mean over seeds):", "",
            "| Bucket | Accuracy |", "|---|---:|",
        ]
        for b in ("near_duplicate", "similar", "novel"):
            v = a["novelty_bucket_accuracy_with_augmentation"][b]
            lines.append(f"| {b} | {v['mean']:.4f} ± {v['sd']:.4f} |" if v["mean"] is not None
                         else f"| {b} | n/a |")
        lines += ["", "Deployed artefact on the same grouped test sets (contaminated share is disclosed):", "",
                  "| Seed | Deployed accuracy | Share of test rows seen in its training pool |",
                  "|---:|---:|---:|"]
        for s, r in intent["per_seed"].items():
            d = r["deployed_artifact_on_grouped_test"]
            lines.append(f"| {s} | {d['accuracy']:.4f} | "
                         f"{d['share_test_seen_in_deployed_training_pool']:.2%} |")
        lines += ["", f"Full detail: `{OUT / 'GROUPED_INTENT_REPORT.md'}`", ""]
    else:
        lines += ["_Pending: run `04_evaluate_grouped_intent.py`._", ""]

    lines += ["## 3. Grouped retrieval evaluation", ""]
    if retrieval:
        d = retrieval["dense_all_test"]
        h = retrieval["hybrid_sample"]
        lines += [
            f"- Index rows: {retrieval['n_index']:,}; grouped test rows: {retrieval['n_test']:,}.",
            f"- Test rows with a >=0.95 neighbour in the grouped index: "
            f"**{retrieval['test_max_dense_similarity_to_index']['ge_0.95']:.2%}**.",
            f"- Dense topic match @1: **{d['category_match_top1']:.4f}** "
            f"(95% CI {d['category_match_top1_ci95']}); recall@5 {d['recall_at_5']:.4f}; "
            f"MRR@10 {d['mrr_at_10']:.4f}; nDCG@5 {d['ndcg_at_5']:.4f}.",
            f"- Hybrid (alpha=0.4, n={retrieval['protocol']['hybrid_sample']}) topic match @1: "
            f"{h['category_match_top1']:.4f}; recall@5 {h['recall_at_5']:.4f}; "
            f"answer token-F1 {h['answer_token_f1_top1_all']:.4f}.",
            "", f"Full detail: `{OUT / 'GROUPED_RETRIEVAL_REPORT.md'}`", "",
        ]
    else:
        lines += ["_Pending: run `05_evaluate_grouped_retrieval.py`._", ""]

    lines += [
        "## 4. What changed and what did not", "",
        "- Split leakage is removed by construction: no question group appears on both sides.",
        "- Annotator bias is **not** removed: labels, gold queries and paraphrases remain LLM-generated.",
        "- Retrieval relevance remains a weak-label topical proxy, not agronomic correctness.",
        "- Human verification, real speech and expert safety review remain PENDING.", "",
    ]
    (OUT / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print("wrote", OUT / "REPORT.md")


if __name__ == "__main__":
    main()
