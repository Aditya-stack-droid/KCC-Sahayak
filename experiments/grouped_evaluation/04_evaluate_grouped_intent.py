"""Evaluate intent classification on the group-aware annotated splits.

For each of five split seeds, a fresh model is trained with the documented
deployed recipe (MiniLM embedding + char n-gram TF-IDF -> logistic regression,
C=10, embedding weight 1.0), on the grouped training side only. Spoken-style
paraphrases are included only when their source query is on the training side.
The same recipe is also fit without paraphrases so the augmentation effect can
be checked under the stricter split.

The deployed artefact is evaluated separately on the same grouped test sets.
It was trained on the original 3,000-query pool, so some grouped-test rows were
seen during its training; the contaminated share is reported rather than hidden.

Outputs:
  results/improved_evaluation/grouped_intent.json
  results/improved_evaluation/grouped_intent_per_class.csv
  results/improved_evaluation/grouped_confusion_seed0.csv
  results/improved_evaluation/GROUPED_INTENT_REPORT.md
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "src"))
from common import ART, LABELS, OUT, SEEDS, encode_cached, ensure_dirs, load_annotations, load_gold  # noqa: E402
from intent_model import IntentHead, fit_head  # noqa: E402
from metrics import bootstrap_ci, novelty_bucket  # noqa: E402


def scores(y_true, y_pred) -> dict:
    p, r, f, s = precision_recall_fscore_support(
        y_true, y_pred, labels=LABELS, average=None, zero_division=0)
    macro = precision_recall_fscore_support(y_true, y_pred, labels=LABELS,
                                            average="macro", zero_division=0)
    weighted = precision_recall_fscore_support(y_true, y_pred, labels=LABELS,
                                               average="weighted", zero_division=0)
    ok = np.asarray(y_true) == np.asarray(y_pred)
    return {
        "n": int(len(y_true)),
        "accuracy": round(float(ok.mean()), 4),
        "accuracy_ci95": bootstrap_ci(ok),
        "macro_precision": round(float(macro[0]), 4),
        "macro_recall": round(float(macro[1]), 4),
        "macro_f1": round(float(macro[2]), 4),
        "weighted_precision": round(float(weighted[0]), 4),
        "weighted_recall": round(float(weighted[1]), 4),
        "weighted_f1": round(float(weighted[2]), 4),
        "per_class": {
            lab: {"precision": round(float(p[i]), 4), "recall": round(float(r[i]), 4),
                  "f1": round(float(f[i]), 4), "support": int(s[i])}
            for i, lab in enumerate(LABELS)
        },
    }


def predict(head, emb, texts):
    return np.array(head.classes)[head.decision(emb, list(texts)).argmax(1)]


def main() -> None:
    ensure_dirs()
    ann, aug = load_annotations()
    gold = load_gold()

    E_ann = encode_cached(ann["text"], "annotated_texts")
    E_aug = encode_cached(aug["text"], "augmented_texts")
    E_gold = encode_cached(gold["text"], "gold_texts")
    ann_pos = {str(i): k for k, i in enumerate(ann["id"])}
    aug_pos = {str(i): k for k, i in enumerate(aug["source_id"])}

    deployed = IntentHead.load(HERE.parents[1] / "models" / "intent_embed_lr" / "model.joblib")
    deployed_train_pool = set(ann.loc[ann["pool"] == "train", "qid"].tolist())

    per_seed = {}
    per_class_rows = []
    confusion_written = False
    bucket_acc = {"novel": [], "similar": [], "near_duplicate": []}
    bucket_counts = {}

    for seed in SEEDS:
        split_df = pd.read_csv(ART / f"annotated_split_seed{seed}.csv")
        if not np.array_equal(split_df["id"].astype(str).to_numpy(), ann["id"].astype(str).to_numpy()):
            raise SystemExit(f"split seed {seed} is not aligned with annotated.csv")
        aug_split = pd.read_csv(ART / f"augmented_split_seed{seed}.csv")
        s = split_df["split"].to_numpy()
        tr, te = s == "train", s == "test"

        aug_tr = aug_split["split"].to_numpy() == "train"
        if len(aug_split) != len(aug):
            # split file preserves augmented.csv row order; guard anyway
            aug_split = aug_split.set_index("source_id").loc[aug["source_id"]].reset_index()
            aug_tr = aug_split["split"].to_numpy() == "train"

        e_tr = np.vstack([E_ann[tr], E_aug[aug_tr]])
        t_tr = pd.concat([ann.loc[tr, "text"], aug.loc[aug_tr, "text"]], ignore_index=True)
        y_tr = pd.concat([ann.loc[tr, "label"], aug.loc[aug_tr, "label"]], ignore_index=True).to_numpy()

        results = {}
        for variant, use_aug in (("with_augmentation", True), ("no_augmentation", False)):
            if use_aug:
                head = fit_head(e_tr, t_tr, y_tr, C=10, use_tfidf=True, emb_weight=1.0)
            else:
                head = fit_head(E_ann[tr], ann.loc[tr, "text"], ann.loc[tr, "label"].to_numpy(),
                                C=10, use_tfidf=True, emb_weight=1.0)
            pred = predict(head, E_ann[te], ann.loc[te, "text"])
            res = scores(ann.loc[te, "label"].to_numpy(), pred)
            res["n_train_annotated"] = int(tr.sum())
            res["n_train_paraphrases"] = int(aug_tr.sum()) if use_aug else 0
            res["n_test"] = int(te.sum())
            results[variant] = res

            if variant == "with_augmentation" and not confusion_written:
                cm = confusion_matrix(ann.loc[te, "label"], pred, labels=LABELS)
                pd.DataFrame(cm, index=LABELS, columns=LABELS).to_csv(
                    OUT / "grouped_confusion_seed0.csv")
                confusion_written = True

            if variant == "with_augmentation":
                # per-class rows (seed 0 only, kept compact)
                if seed == 0:
                    for i, lab in enumerate(LABELS):
                        per_class_rows.append({
                            "seed": seed, "label": lab,
                            **{k: v for k, v in res["per_class"][lab].items()},
                        })

        # novelty buckets relative to the grouped training side
        sims = (E_ann[te] @ np.vstack([E_ann[tr], E_aug[aug_tr]]).T).max(axis=1)
        buckets = np.array([novelty_bucket(float(x)) for x in sims])
        if seed == 0:
            bucket_counts["with_augmentation"] = {
                b: int((buckets == b).sum()) for b in ("novel", "similar", "near_duplicate")}
        for b in ("novel", "similar", "near_duplicate"):
            m = buckets == b
            if m.any():
                pred = predict(
                    fit_head(e_tr, t_tr, y_tr, C=10, use_tfidf=True, emb_weight=1.0),
                    E_ann[te][m], ann.loc[te, "text"].iloc[np.flatnonzero(m)])
                bucket_acc[b].append(float((pred == ann.loc[te, "label"].to_numpy()[m]).mean()))

        # gold evaluation for the refit model
        gp = predict(fit_head(e_tr, t_tr, y_tr, C=10, use_tfidf=True, emb_weight=1.0),
                     E_gold, gold["text"])
        gold_res = scores(gold["label"].to_numpy(), gp)
        gold_res["by_style"] = {
            st: round(float((gp[(gold["style"] == st).to_numpy()] ==
                             gold.loc[gold["style"] == st, "label"].to_numpy()).mean()), 4)
            for st in ("en", "hinglish", "hi")
        }

        # deployed artefact on the same grouped test rows
        dp = predict(deployed, E_ann[te], ann.loc[te, "text"])
        dep = scores(ann.loc[te, "label"].to_numpy(), dp)
        dep["share_test_seen_in_deployed_training_pool"] = round(
            float(ann.loc[te, "qid"].isin(deployed_train_pool).mean()), 4)
        dep["note"] = ("The deployed head was trained on the original 3,000-row pool; "
                       "rows counted above were seen during its training and make its "
                       "grouped-test score an upper bound. The refit rows are the primary "
                       "leakage-resistant estimate.")

        per_seed[str(seed)] = {
            "train": int(tr.sum()), "test": int(te.sum()),
            "train_paraphrases_included": int(aug_tr.sum()),
            "test_paraphrases_excluded": int((~aug_tr).sum()),
            "max_similarity_to_training": {
                "ge_0.95": round(float((sims >= 0.95).mean()), 4),
                "ge_0.90": round(float((sims >= 0.90).mean()), 4),
                "lt_0.80": round(float((sims < 0.80).mean()), 4),
            },
            "models": results,
            "gold_with_augmentation": gold_res,
            "deployed_artifact_on_grouped_test": dep,
        }
        print(f"seed {seed}: refit with aug={results['with_augmentation']['accuracy']:.4f} "
              f"no aug={results['no_augmentation']['accuracy']:.4f} "
              f"gold={gold_res['accuracy']:.4f} "
              f"deployed={dep['accuracy']:.4f} (seen {dep['share_test_seen_in_deployed_training_pool']:.2f})")

    def agg(key):
        vals = [per_seed[str(s)]["models"]["with_augmentation"][key] for s in SEEDS]
        return {"mean": round(float(np.mean(vals)), 4), "sd": round(float(np.std(vals)), 4),
                "values": vals}

    aggregate = {
        "accuracy": agg("accuracy"),
        "macro_f1": agg("macro_f1"),
        "weighted_f1": agg("weighted_f1"),
        "no_augmentation_accuracy": {
            "mean": round(float(np.mean([per_seed[str(s)]["models"]["no_augmentation"]["accuracy"]
                                         for s in SEEDS])), 4),
            "sd": round(float(np.std([per_seed[str(s)]["models"]["no_augmentation"]["accuracy"]
                                      for s in SEEDS])), 4),
        },
        "novelty_bucket_accuracy_with_augmentation": {
            b: {"mean": round(float(np.mean(v)), 4) if v else None,
                "sd": round(float(np.std(v)), 4) if v else None,
                "n_seeds": len(v)}
            for b, v in bucket_acc.items()
        },
        "novelty_bucket_counts_seed0": bucket_counts.get("with_augmentation"),
    }

    original = {}
    p = HERE.parents[1] / "results" / "research_eval.json"
    if p.exists():
        stored = json.loads(p.read_text())
        production = stored.get("main", {}).get("production (emb+char, aug)", {})
        original = {
            "source": "results/research_eval.json (original row-level split, LLM labels)",
            "silver_accuracy": production.get("silver", {}).get("acc"),
            "silver_ci95": production.get("silver", {}).get("ci95"),
            "gold_accuracy": production.get("gold_all", {}).get("acc"),
            "note": "Reference only; not comparable one-for-one because the split differs.",
        }

    out = {
        "protocol": {
            "grouping": "exact normalised question + MiniLM cosine >= 0.95, transitive union-find",
            "split": "StratifiedGroupKFold on meaning labels; 7 folds; fold 0 test; 5 seeds",
            "model": "MiniLM embedding + char_wb 2-5 TF-IDF -> LogisticRegression C=10, emb_weight=1.0",
            "buckets": "max cosine to grouped training side: <0.80 novel, 0.80-0.90 similar, >=0.90 near-duplicate",
            "seeds": SEEDS,
            "primary_estimate": "refit model on grouped train (with augmentation restricted to train groups)",
        },
        "annotation_counts": {
            "annotated_total": int(len(ann)),
            "paraphrases_total": int(len(aug)),
            "gold_total": int(len(gold)),
        },
        "per_seed": per_seed,
        "aggregate": aggregate,
        "original_split_reference": original,
        "limitations": [
            "Meaning labels are LLM-generated, not human-verified; this pipeline removes split leakage, not annotator bias.",
            "The annotated pool is 3,496 queries; stratified grouping can only redistribute these rows.",
            "Grouping uses a k=64 candidate cap and a 0.95 threshold; residual high-similarity pairs are reported per seed.",
            "The gold set shares its author with the paraphrases, so its numbers remain optimistic.",
        ],
    }
    (OUT / "grouped_intent.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    pd.DataFrame(per_class_rows).to_csv(OUT / "grouped_intent_per_class.csv", index=False)

    lines = [
        "# Grouped intent evaluation (Batch A)", "",
        "Protocol: exact-duplicate + MiniLM cosine >= 0.95 question groups; "
        "StratifiedGroupKFold (7 folds, fold 0 = test) over the 3,496 LLM-labelled queries; "
        "five split seeds; refit with the documented deployed recipe.", "",
        "**Labels are still LLM-generated. This removes split leakage, not annotator bias.**", "",
        "## Aggregate (mean ± sd over 5 seeds)", "",
        "| Model | Accuracy | Macro F1 | Weighted F1 |", "|---|---:|---:|---:|",
        f"| Refit, embedding + char TF-IDF + LR (with augmentation) | "
        f"{aggregate['accuracy']['mean']:.4f} ± {aggregate['accuracy']['sd']:.4f} | "
        f"{aggregate['macro_f1']['mean']:.4f} ± {aggregate['macro_f1']['sd']:.4f} | "
        f"{aggregate['weighted_f1']['mean']:.4f} ± {aggregate['weighted_f1']['sd']:.4f} |",
        f"| Refit, same recipe without paraphrases | "
        f"{aggregate['no_augmentation_accuracy']['mean']:.4f} ± "
        f"{aggregate['no_augmentation_accuracy']['sd']:.4f} | — | — |", "",
        "## Per seed", "",
        "| Seed | Train | Test | Acc (with aug) | Acc (no aug) | Acc (deployed artefact) | Test rows seen in deployed training |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for s in SEEDS:
        r = per_seed[str(s)]
        lines.append(
            f"| {s} | {r['train']} | {r['test']} | "
            f"{r['models']['with_augmentation']['accuracy']:.4f} | "
            f"{r['models']['no_augmentation']['accuracy']:.4f} | "
            f"{r['deployed_artifact_on_grouped_test']['accuracy']:.4f} | "
            f"{r['deployed_artifact_on_grouped_test']['share_test_seen_in_deployed_training_pool']:.2%} |")
    lines += ["", "## Novelty buckets (refit, mean over seeds)", "",
              "| Bucket | N (seed 0) | Accuracy |", "|---|---:|---:|"]
    for b in ("near_duplicate", "similar", "novel"):
        v = aggregate["novelty_bucket_accuracy_with_augmentation"][b]
        n = (aggregate.get("novelty_bucket_counts_seed0") or {}).get(b, "—")
        lines.append(f"| {b} | {n} | {v['mean']:.4f} ± {v['sd']:.4f} |" if v["mean"] is not None
                     else f"| {b} | {n} | n/a |")
    lines += ["", "## Gold set (independent phrasing, refit model)", "",
              "| Seed | Accuracy | English | Hinglish | Hindi |", "|---:|---:|---:|---:|---:|"]
    for s in SEEDS:
        g = per_seed[str(s)]["gold_with_augmentation"]
        lines.append(f"| {s} | {g['accuracy']:.4f} | {g['by_style']['en']:.4f} | "
                     f"{g['by_style']['hinglish']:.4f} | {g['by_style']['hi']:.4f} |")
    if original:
        lines += ["", "## Original row-level reference (not directly comparable)", "",
                  f"- Silver accuracy: {original.get('silver_accuracy')} "
                  f"(CI {original.get('silver_ci95')})",
                  f"- Gold accuracy: {original.get('gold_accuracy')}",
                  f"- Source: `{original.get('source')}`"]
    lines += ["", "## Limitations", ""] + [f"- {x}" for x in out["limitations"]] + [""]
    (OUT / "GROUPED_INTENT_REPORT.md").write_text("\n".join(lines), encoding="utf-8")

    print("\nAggregate:", json.dumps(aggregate, indent=2))
    print("wrote", OUT / "grouped_intent.json")


if __name__ == "__main__":
    main()
