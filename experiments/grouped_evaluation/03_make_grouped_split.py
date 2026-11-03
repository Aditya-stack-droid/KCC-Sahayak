"""Create group-aware splits for the corpus and the annotated evaluation set.

Corpus split (seed 0, 10-fold StratifiedGroupKFold): fold 0 -> test,
fold 1 -> validation, everything else -> train. Used for retrieval
evaluation and leakage checks.

Annotated split (seeds 0-4, 7-fold StratifiedGroupKFold): one fold -> test,
the rest -> train. Stratification uses the meaning-based labels. Parallel
spoken-style paraphrases are attached to the split of their source query so
they can never leak into the test side.

Outputs:
  artifacts/corpus_split.csv
  artifacts/annotated_split_seed{k}.csv
  artifacts/augmented_split_seed{k}.csv
  artifacts/split_stats.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ART, SEEDS, ensure_dirs, load_annotations, load_corpus  # noqa: E402


def check_disjoint(groups: np.ndarray, split: np.ndarray) -> dict:
    train = set(groups[split == "train"].tolist())
    test = set(groups[split == "test"].tolist())
    val = set(groups[split == "val"].tolist()) if (split == "val").any() else set()
    return {
        "train_test_group_overlap": len(train & test),
        "train_val_group_overlap": len(train & val),
        "val_test_group_overlap": len(val & test),
    }


def distribution(frame: pd.DataFrame, split: np.ndarray) -> dict:
    out = {}
    for name in ("train", "val", "test"):
        m = split == name
        if m.any():
            out[name] = {str(k): int(v) for k, v in frame.loc[m, "label"].value_counts().sort_index().items()}
    return out


def main() -> None:
    ensure_dirs()
    df = load_corpus()
    groups_df = pd.read_csv(ART / "groups.csv")
    if len(groups_df) != len(df) or not np.array_equal(groups_df["qid"].to_numpy(), df["qid"].to_numpy()):
        raise SystemExit("groups.csv does not align with the corpus; rerun 02_build_groups.py")
    group_ids = groups_df["group_id"].to_numpy()

    # ---------------- corpus split (one seed; retrieval is expensive to repeat)
    n_splits = 10
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=0)
    folds = list(sgkf.split(np.zeros(len(df)), df["query_type"], group_ids))
    test_idx, val_idx = folds[0][1], folds[1][1]
    split = np.full(len(df), "train", dtype=object)
    split[val_idx] = "val"
    split[test_idx] = "test"
    corpus_split = pd.DataFrame({
        "qid": df["qid"], "group_id": group_ids, "weak_label": df["query_type"], "split": split,
    })
    corpus_split.to_csv(ART / "corpus_split.csv", index=False)
    checks = check_disjoint(group_ids, split)
    if any(checks.values()):
        raise SystemExit(f"group leakage in corpus split: {checks}")
    dist = {name: int((split == name).sum()) for name in ("train", "val", "test")}
    print("corpus split:", dist, "groups:", len(np.unique(group_ids)))

    # ---------------- annotated splits (5 seeds)
    ann, aug = load_annotations()
    ann_groups = ann["qid"].map(dict(zip(groups_df["qid"], groups_df["group_id"]))).to_numpy()
    if pd.isna(ann_groups).any():
        raise SystemExit("some annotated qids are absent from groups.csv")
    ann_groups = ann_groups.astype(np.int64)
    stats = {
        "corpus": {
            "n_splits": n_splits, "counts": dist, "group_checks": checks,
            "n_groups": int(len(np.unique(group_ids))),
            "class_share": {k: round(float(v), 4) for k, v in
                            (df["query_type"].value_counts(normalize=True)).items()},
        },
        "annotated": {},
    }

    for seed in SEEDS:
        sgkf = StratifiedGroupKFold(n_splits=7, shuffle=True, random_state=seed)
        folds = list(sgkf.split(np.zeros(len(ann)), ann["label"], ann_groups))
        # choose the minority of the two folds' test membership? fold 0 is ~1/7 by construction
        test_idx = folds[0][1]
        s = np.full(len(ann), "train", dtype=object)
        s[test_idx] = "test"
        s_out = pd.DataFrame({
            "id": ann["id"], "qid": ann["qid"], "group_id": ann_groups,
            "label": ann["label"], "split": s,
        })
        s_out.to_csv(ART / f"annotated_split_seed{seed}.csv", index=False)
        c = check_disjoint(ann_groups, s)
        if c["train_test_group_overlap"]:
            raise SystemExit(f"annotated split seed {seed} leaks groups: {c}")
        # attach paraphrases to their source query's side
        src_group = dict(zip(ann["id"], ann_groups))
        train_groups = set(ann_groups[s == "train"].tolist())
        aug_s = pd.DataFrame({
            "source_id": aug["source_id"], "text": aug["text"], "label": aug["label"],
            "style": aug["style"],
            "group_id": aug["source_id"].map(src_group),
        })
        aug_s["split"] = np.where(aug_s["group_id"].isin(train_groups), "train", "test")
        aug_s = aug_s.dropna(subset=["group_id"])
        aug_s["group_id"] = aug_s["group_id"].astype(np.int64)
        aug_s.to_csv(ART / f"augmented_split_seed{seed}.csv", index=False)

        stats["annotated"][str(seed)] = {
            "n_train": int((s == "train").sum()), "n_test": int(test_idx.size),
            "n_train_aug": int((aug_s["split"] == "train").sum()),
            "n_test_aug_excluded": int((aug_s["split"] == "test").sum()),
            "group_checks": c,
            "class_distribution": {
                "train": {k: int(v) for k, v in ann.loc[s == "train", "label"].value_counts().sort_index().items()},
                "test": {k: int(v) for k, v in ann.loc[s == "test", "label"].value_counts().sort_index().items()},
            },
        }
        print(f"annotated seed {seed}: train={int((s == 'train').sum())} test={int(test_idx.size)} "
              f"aug_train={int((aug_s['split'] == 'train').sum())} "
              f"aug_excluded={int((aug_s['split'] == 'test').sum())}")

    # compare test class distribution to the full annotated distribution
    for seed in SEEDS:
        s = pd.read_csv(ART / f"annotated_split_seed{seed}.csv")
        full = ann["label"].value_counts(normalize=True)
        te = s.loc[s["split"] == "test", "label"].value_counts(normalize=True)
        max_dev = max(abs(te.get(k, 0.0) - full.get(k, 0.0)) for k in full.index)
        stats["annotated"][str(seed)]["test_class_max_abs_share_deviation"] = round(float(max_dev), 4)

    (ART / "split_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print("wrote", ART / "split_stats.json")


if __name__ == "__main__":
    main()
