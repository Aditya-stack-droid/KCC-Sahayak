"""Group the full cleaned corpus by question identity.

Two rows are in the same group when their normalised question strings are
identical, or when the cosine similarity of their MiniLM embeddings is at
least ``NEAR_DUP`` (0.95). Candidates come from a FAISS k-NN search
(k=64); edges are merged transitively with union-find.

Also measures leakage in the *original* row-level split:
  * share of test questions exactly present in train;
  * share of test questions in a group that also contains a train query;
  * share of test questions with a >=0.95 dense neighbour in train+val
    (the index used by the shipped system).

Outputs:
  artifacts/groups.csv
  artifacts/groups_stats.json
"""
from __future__ import annotations

import json
import sys
import time

import faiss
import numpy as np
import pandas as pd

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from common import ART, DATA, NEAR_DUP, ensure_dirs, load_corpus  # noqa: E402
from grouping import apply_edges, canonical_group_ids, group_exact, group_leakage, group_sizes  # noqa: E402


def main() -> None:
    ensure_dirs()
    df = load_corpus()
    qids = np.load(ART / "corpus_qids.npy")
    emb = np.load(ART / "corpus_embeddings.npy")
    if len(df) != len(qids) or len(qids) != len(emb):
        raise SystemExit("corpus, qid list and embeddings disagree; rerun 01_prepare_embeddings.py")
    if not np.array_equal(df["qid"].to_numpy(), qids):
        raise SystemExit("qid order mismatch between corpus and embedding cache")

    # ---- exact duplicates
    t0 = time.time()
    uf = group_exact(df["questions"].tolist())
    exact_components = uf.n_components
    print(f"exact grouping: {len(df)} rows -> {exact_components} components in {time.time()-t0:.1f}s")

    # ---- near duplicates via FAISS k-NN over the whole corpus
    t0 = time.time()
    k = 64
    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(np.ascontiguousarray(emb))
    n = len(emb)
    n_edges = 0
    pair_count = 0
    for start in range(0, n, 4096):
        stop = min(start + 4096, n)
        sims, idxs = index.search(np.ascontiguousarray(emb[start:stop]), min(k, n))
        rows = np.repeat(np.arange(start, stop), sims.shape[1])
        cols = idxs.ravel()
        vals = sims.ravel()
        keep = (rows != cols) & (vals >= NEAR_DUP)
        pair_count += int(keep.sum())
        edges = zip(rows[keep].tolist(), cols[keep].tolist())
        n_edges += apply_edges(uf, edges)
        if start % 40960 == 0:
            print(f"  scanned {stop}/{n} rows, merges so far {n_edges}")
    print(f"near-duplicate scan: {pair_count} directed pairs >= {NEAR_DUP}, "
          f"{n_edges} union merges, {time.time()-t0:.1f}s")

    group_ids = canonical_group_ids(uf)
    sizes = group_sizes(group_ids)
    counts = pd.Series(group_ids).value_counts()
    groups = pd.DataFrame({"qid": qids, "group_id": group_ids})
    groups["group_size"] = groups["group_id"].map(counts).astype(int)
    groups.to_csv(ART / "groups.csv", index=False)

    # ---- leakage in the original row-level split
    train = pd.read_csv(DATA / "train.csv", usecols=["qid"])
    val = pd.read_csv(DATA / "val.csv", usecols=["qid"])
    test = pd.read_csv(DATA / "test.csv", usecols=["qid"])
    split = np.full(len(df), "other", dtype=object)
    qid_pos = {int(q): i for i, q in enumerate(qids.tolist())}
    for name, frame in (("train", train), ("val", val), ("test", test)):
        for q in frame["qid"].tolist():
            pos = qid_pos.get(int(q))
            if pos is not None:
                split[pos] = name
    mask = split != "other"
    if not mask.all():
        raise SystemExit(f"{int((~mask).sum())} corpus rows are not in any original split")

    train_q = set(train["qid"].tolist())
    test_df = df[df["qid"].isin(test["qid"])]
    exact_in_train = test_df["questions"].isin(
        set(df[df["qid"].isin(train_q)]["questions"])).mean()
    gal = group_leakage(group_ids, split)
    val_groups = set(group_ids[split == "val"].tolist())
    test_groups = group_ids[split == "test"]
    leak_tv = np.array([g in (set(group_ids[split == "train"].tolist()) | val_groups)
                        for g in test_groups.tolist()])

    # reproduce the shipped near-duplicate statistic against train+val
    tv_mask = (split == "train") | (split == "val")
    tv_emb = np.ascontiguousarray(emb[tv_mask])
    tv_index = faiss.IndexFlatIP(emb.shape[1])
    tv_index.add(tv_emb)
    test_emb = np.ascontiguousarray(emb[split == "test"])
    sims, _ = tv_index.search(test_emb, 1)
    top1 = sims[:, 0]

    stats = {
        "n_rows": int(len(df)),
        "n_groups": int(len(counts)),
        "n_singleton_groups": int((counts == 1).sum()),
        "largest_group": int(counts.max()),
        "group_size_p50": int(np.percentile(sizes, 50)),
        "group_size_p95": int(np.percentile(sizes, 95)),
        "group_size_p99": int(np.percentile(sizes, 99)),
        "mean_group_size": round(float(sizes.mean()), 3),
        "near_dup_threshold": NEAR_DUP,
        "knn_k": k,
        "directed_pairs_ge_threshold": pair_count,
        "union_edges_applied": n_edges,
        "exact_components_before_near_dup": exact_components,
        "original_split_leakage": {
            "exact_test_question_in_train": round(float(exact_in_train), 6),
            "test_in_group_with_train": gal,
            "test_in_group_with_train_or_val": {
                "n_test": int((split == "test").sum()),
                "n_leaking": int(leak_tv.sum()),
                "share": round(float(leak_tv.mean()), 6),
            },
            "test_max_dense_similarity_to_train_val": {
                "ge_0.95": round(float((top1 >= 0.95).mean()), 6),
                "ge_0.90": round(float((top1 >= 0.90).mean()), 6),
                "lt_0.80": round(float((top1 < 0.80).mean()), 6),
            },
        },
        "note": ("Group membership implies at least one >=threshold link, but the k-NN "
                 "candidate cap (k=64) means very large duplicate clusters could in "
                 "principle be split; the post-split maximum-similarity check in "
                 "03/05 verifies the actual grouped split."),
    }
    (ART / "groups_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
