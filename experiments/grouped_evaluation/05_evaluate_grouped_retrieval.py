"""Retrieval evaluation on the group-aware corpus split.

Builds a fresh dense index and TF-IDF matrix over the grouped train+val side,
then evaluates dense and hybrid retrieval on the grouped test side. The
original single-reference token-F1 metric is kept so the grouped and original
numbers can be compared; ranking metrics are added.

Relevance is the weak-label match (retrieved topic == test topic), exactly the
proxy used by the original evaluation. It is a topical proxy, not factual
correctness, and is reported as such.

Outputs:
  artifacts/grouped_retrieval_index.faiss
  results/improved_evaluation/grouped_retrieval.json
  results/improved_evaluation/GROUPED_RETRIEVAL_REPORT.md
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import faiss
import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "src"))
from answer_quality import informative_mask  # noqa: E402
from common import ART, INDEX_DIR, OUT, ensure_dirs, load_corpus  # noqa: E402
from metrics import bootstrap_ci, ndcg_at_k, novelty_bucket, reciprocal_rank, recall_at_k, token_f1  # noqa: E402

ALPHA = 0.4  # deployed fusion weight (models/retrieval_index/hybrid.json)
HYBRID_SAMPLE = 3000


def rank_metrics(ranks_labels: list[list[bool]], k_max: int = 10) -> dict:
    n = len(ranks_labels)
    return {
        "recall_at_1": round(float(np.mean([recall_at_k(r, 1) for r in ranks_labels])), 4),
        "recall_at_3": round(float(np.mean([recall_at_k(r, 3) for r in ranks_labels])), 4),
        "recall_at_5": round(float(np.mean([recall_at_k(r, 5) for r in ranks_labels])), 4),
        "mrr_at_10": round(float(np.mean([reciprocal_rank(r[:k_max]) for r in ranks_labels])), 4),
        "ndcg_at_5": round(float(np.mean([ndcg_at_k(r, 5) for r in ranks_labels])), 4),
        "n": int(n),
    }


def main() -> None:
    ensure_dirs()
    df = load_corpus()
    split = pd.read_csv(ART / "corpus_split.csv")
    if len(split) != len(df) or not np.array_equal(split["qid"].to_numpy(), df["qid"].to_numpy()):
        raise SystemExit("corpus_split.csv does not align with the corpus")
    emb = np.load(ART / "corpus_embeddings.npy")
    s = split["split"].to_numpy()
    indexable = (s == "train") | (s == "val")
    test_mask = s == "test"
    corpus_idx = np.flatnonzero(indexable)
    test_idx = np.flatnonzero(test_mask)
    print(f"index rows {len(corpus_idx)}, test rows {len(test_idx)}")

    # ---------------- dense index over grouped train+val only
    t0 = time.time()
    index = faiss.IndexFlatIP(emb.shape[1])
    for start in range(0, len(corpus_idx), 20000):
        chunk = corpus_idx[start:start + 20000]
        index.add(np.ascontiguousarray(emb[chunk]))
    faiss.write_index(index, str(ART / "grouped_retrieval_index.faiss"))
    print(f"built dense index in {time.time()-t0:.1f}s")

    meta = df.iloc[corpus_idx].reset_index(drop=True)
    meta[["qid", "questions", "answers", "query_type"]].to_csv(
        ART / "grouped_retrieval_metadata.csv", index=False)
    valid = informative_mask(meta["answers"])
    np.save(ART / "grouped_retrieval_informative_mask.npy", valid)
    print(f"informative answers in index: {int(valid.sum())}/{len(meta)}")

    # ---------------- TF-IDF over the grouped train+val questions
    t0 = time.time()
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True, dtype=np.float32)
    C = vec.fit_transform(meta["questions"].astype(str))
    joblib.dump(vec, ART / "grouped_tfidf.joblib")
    sp.save_npz(ART / "grouped_tfidf_matrix.npz", C)
    print(f"TF-IDF: {C.shape} nnz={C.nnz} in {time.time()-t0:.1f}s")

    Q_emb = np.ascontiguousarray(emb[test_idx])
    Q_text = df.iloc[test_idx]["questions"].astype(str).tolist()
    Q_tf = vec.transform(Q_text)
    y_test = df.iloc[test_idx]["query_type"].to_numpy()
    answers_test = df.iloc[test_idx]["answers"].astype(str).to_numpy()
    gold_informative = informative_mask(pd.Series(answers_test))

    dense_sims, dense_ids = index.search(Q_emb, 200)
    dense_top1_sim = dense_sims[:, 0]
    dense_top1_id = np.where(dense_ids[:, 0] >= 0, dense_ids[:, 0], 0)

    def top1_category(ids):
        return np.where(ids >= 0, meta["query_type"].to_numpy()[np.clip(ids, 0, len(meta) - 1)], "")

    def evaluate_ids(ids: np.ndarray) -> dict:
        """Metrics from a [n, 10] matrix of index ids (dense ranking)."""
        cats = np.where(ids >= 0, meta["query_type"].to_numpy()[np.clip(ids, 0, len(meta) - 1)], "")
        match = cats == y_test[:, None]
        ranks = [list(row) for row in match]
        f1 = np.array([
            token_f1(meta["answers"].iloc[int(i)] if i >= 0 else "", a)
            for i, a in zip(ids[:, 0], answers_test)
        ])
        return {
            "category_match_top1": round(float(match[:, 0].mean()), 4),
            "category_match_top1_ci95": bootstrap_ci(match[:, 0]),
            "answer_token_f1_top1_all": round(float(f1.mean()), 4),
            "answer_token_f1_top1_informative_gold": round(float(f1[gold_informative].mean()), 4)
            if gold_informative.any() else None,
            **rank_metrics(ranks),
        }

    dense_metrics = evaluate_ids(dense_ids[:, :10])
    print("dense (all grouped test):", json.dumps(dense_metrics))

    # ---------------- hybrid on a fixed sample for tractability
    rng = np.random.default_rng(42)
    sample = rng.choice(len(test_idx), size=min(HYBRID_SAMPLE, len(test_idx)), replace=False)
    sample.sort()
    Qs_tf = Q_tf[sample]
    Qs_ids = dense_ids[sample]
    Qs_sims = dense_sims[sample]
    Qs_emb = Q_emb[sample]
    y_s = y_test[sample]
    ans_s = answers_test[sample]
    gold_s = gold_informative[sample]

    tf_ids = np.zeros((len(sample), 200), dtype=np.int64)
    tf_scores = np.zeros((len(sample), 200), dtype=np.float32)
    t0 = time.time()
    block = 128
    for start in range(0, len(sample), block):
        stop = min(start + block, len(sample))
        S = (Qs_tf[start:stop] @ C.T).toarray()
        part = np.argpartition(-S, 199, axis=1)[:, :200]
        tf_ids[start:stop] = part
        tf_scores[start:stop] = np.take_along_axis(S, part, axis=1)
    print(f"TF-IDF candidate search for {len(sample)} queries in {time.time()-t0:.1f}s")

    hybrid_ids = np.zeros((len(sample), 10), dtype=np.int64)
    hybrid_dense_sims = np.zeros((len(sample), 10), dtype=np.float32)
    hybrid_tf_scores = np.zeros((len(sample), 10), dtype=np.float32)
    t0 = time.time()
    for i in range(len(sample)):
        cand = np.union1d(Qs_ids[i], tf_ids[i])
        cand = cand[cand >= 0]
        if len(cand) == 0:
            continue
        dsim = emb[corpus_idx[cand]] @ Qs_emb[i]
        score_map = dict(zip(tf_ids[i].tolist(), tf_scores[i].tolist()))
        missing = [int(c) for c in cand if int(c) not in score_map]
        if missing:
            extra = np.asarray((C[missing] @ Qs_tf[i].T).todense()).ravel()
            score_map.update({int(c): float(v) for c, v in zip(missing, extra)})
        ts = np.array([score_map[int(c)] for c in cand], dtype="float32")
        fused = ALPHA * dsim + (1.0 - ALPHA) * ts
        order = np.argsort(-fused)[:10]
        hybrid_ids[i] = cand[order]
        hybrid_dense_sims[i] = dsim[order]
        hybrid_tf_scores[i] = ts[order]
    print(f"hybrid ranking for {len(sample)} queries in {time.time()-t0:.1f}s")

    # hybrid metrics (same-sample dense baseline for a paired comparison)
    def sample_metrics(ids, label):
        cats = np.where(ids >= 0, meta["query_type"].to_numpy()[np.clip(ids, 0, len(meta) - 1)], "")
        match = cats == y_s[:, None]
        f1 = np.array([token_f1(meta["answers"].iloc[int(j)] if j >= 0 else "", a)
                       for j, a in zip(ids[:, 0], ans_s)])
        return {
            "category_match_top1": round(float(match[:, 0].mean()), 4),
            "category_match_top1_ci95": bootstrap_ci(match[:, 0]),
            "answer_token_f1_top1_all": round(float(f1.mean()), 4),
            "answer_token_f1_top1_informative_gold": round(float(f1[gold_s].mean()), 4)
            if gold_s.any() else None,
            **rank_metrics([list(r) for r in match]),
        }

    hybrid_metrics = sample_metrics(hybrid_ids, "hybrid")
    dense_sample_metrics = sample_metrics(Qs_ids[:, :10], "dense")
    print("dense (sample):", json.dumps(dense_sample_metrics))
    print("hybrid (sample):", json.dumps(hybrid_metrics))

    # ---------------- buckets by similarity to the grouped index
    buckets = np.array([novelty_bucket(float(x)) for x in dense_top1_sim])
    bucket_stats = {}
    for b in ("novel", "similar", "near_duplicate"):
        m = buckets == b
        if not m.any():
            bucket_stats[b] = {"n": 0}
            continue
        ids = dense_ids[m][:, :10]
        cats = np.where(ids >= 0, meta["query_type"].to_numpy()[np.clip(ids, 0, len(meta) - 1)], "")
        match = cats == y_test[m][:, None]
        f1 = np.array([token_f1(meta["answers"].iloc[int(j)] if j >= 0 else "", a)
                       for j, a in zip(ids[:, 0], answers_test[m])])
        bucket_stats[b] = {
            "n": int(m.sum()),
            "category_match_top1": round(float(match[:, 0].mean()), 4),
            "recall_at_5": round(float(np.mean([recall_at_k(list(r), 5) for r in match])), 4),
            "answer_token_f1_top1_all": round(float(f1.mean()), 4),
        }

    out = {
        "protocol": {
            "split": "grouped corpus split (StratifiedGroupKFold, 10 folds, seed 0; fold 0 test, fold 1 val)",
            "index": "FAISS IndexFlatIP over grouped train+val questions",
            "fusion": f"alpha={ALPHA} (dense) + {1 - ALPHA:.1f} (word TF-IDF), same as deployed weight",
            "relevance": "retrieved weak keyword label == test weak keyword label (topical proxy, not correctness)",
            "hybrid_sample": int(len(sample)),
        },
        "n_index": int(len(meta)),
        "n_test": int(len(test_idx)),
        "informative_index_share": round(float(valid.mean()), 4),
        "test_max_dense_similarity_to_index": {
            "ge_0.95": round(float((dense_top1_sim >= 0.95).mean()), 6),
            "ge_0.90": round(float((dense_top1_sim >= 0.90).mean()), 6),
            "lt_0.80": round(float((dense_top1_sim < 0.80).mean()), 6),
        },
        "dense_all_test": dense_metrics,
        "dense_sample": dense_sample_metrics,
        "hybrid_sample": hybrid_metrics,
        "by_novelty_bucket": bucket_stats,
        "original_split_reference": {},
        "limitations": [
            "Relevance is a weak-label topical proxy, not human relevance or agronomic correctness.",
            "Only one grouped split seed was run for retrieval because the index rebuild is expensive.",
            "The hybrid evaluation uses a fixed 3,000-query sample; dense metrics use all grouped test rows.",
            "Token-F1 is still single-reference; multi-reference evaluation is Batch B.",
            "The corpus metadata has no district/date field, so temporal or district-held-out evaluation is not possible.",
        ],
    }
    stored_p = ROOT / "results" / "research_eval.json"
    if stored_p.exists():
        stored = json.loads(stored_p.read_text()).get("corpus_audit", {})
        out["original_split_reference"] = {
            "source": "results/research_eval.json (original row-level split)",
            "near_duplicate_share_ge_0.95": stored.get("weak_test_near_duplicate_share_ge_0.95"),
            "retrieval_category_match_near_duplicate": stored.get("retrieval_category_match_near_duplicate"),
            "retrieval_category_match_novel": stored.get("retrieval_category_match_novel"),
            "note": "The original near-duplicate share is expected to fall to ~0 under the grouped split.",
        }

    (OUT / "grouped_retrieval.json").write_text(json.dumps(out, indent=2), encoding="utf-8")

    lines = [
        "# Grouped retrieval evaluation (Batch A)", "",
        f"Grouped corpus split: {len(meta):,} index rows (train+val), {len(test_idx):,} test rows. "
        f"Relevance = weak-label topic match (proxy, not correctness).", "",
        "## Leakage check after grouping", "",
        f"- Test queries with a >=0.95 neighbour in the grouped index: "
        f"**{out['test_max_dense_similarity_to_index']['ge_0.95']:.2%}**",
        f"- Test queries with a >=0.90 neighbour: "
        f"{out['test_max_dense_similarity_to_index']['ge_0.90']:.2%}",
        f"- Test queries with max similarity <0.80 (novel): "
        f"{out['test_max_dense_similarity_to_index']['lt_0.80']:.2%}", "",
        "## Dense retrieval (all grouped test rows)", "",
        "| Metric | Value |", "|---|---:|",
        f"| Topic match @1 | {dense_metrics['category_match_top1']:.4f} "
        f"(95% CI {dense_metrics['category_match_top1_ci95']}) |",
        f"| Recall @5 | {dense_metrics['recall_at_5']:.4f} |",
        f"| MRR @10 | {dense_metrics['mrr_at_10']:.4f} |",
        f"| nDCG @5 | {dense_metrics['ndcg_at_5']:.4f} |",
        f"| Answer token-F1 @1 (all) | {dense_metrics['answer_token_f1_top1_all']:.4f} |",
        f"| Answer token-F1 @1 (informative gold) | "
        f"{dense_metrics['answer_token_f1_top1_informative_gold']} |", "",
        "## Dense vs hybrid (same 3,000-query sample)", "",
        "| System | Topic match @1 | Recall @5 | MRR @10 | Answer F1 |",
        "|---|---:|---:|---:|---:|",
        f"| Dense | {dense_sample_metrics['category_match_top1']:.4f} | "
        f"{dense_sample_metrics['recall_at_5']:.4f} | {dense_sample_metrics['mrr_at_10']:.4f} | "
        f"{dense_sample_metrics['answer_token_f1_top1_all']:.4f} |",
        f"| Hybrid (alpha={ALPHA}) | {hybrid_metrics['category_match_top1']:.4f} | "
        f"{hybrid_metrics['recall_at_5']:.4f} | {hybrid_metrics['mrr_at_10']:.4f} | "
        f"{hybrid_metrics['answer_token_f1_top1_all']:.4f} |", "",
        "## By novelty bucket (dense, all test rows)", "",
        "| Bucket | N | Topic match @1 | Recall @5 | Answer F1 |", "|---|---:|---:|---:|---:|",
    ]
    for b in ("near_duplicate", "similar", "novel"):
        v = bucket_stats[b]
        if v.get("n"):
            lines.append(f"| {b} | {v['n']} | {v['category_match_top1']:.4f} | "
                         f"{v['recall_at_5']:.4f} | {v['answer_token_f1_top1_all']:.4f} |")
        else:
            lines.append(f"| {b} | 0 | — | — | — |")
    if out["original_split_reference"]:
        r = out["original_split_reference"]
        lines += ["", "## Original row-level reference (not directly comparable)", "",
                  f"- Near-duplicate share (>=0.95): {r.get('near_duplicate_share_ge_0.95')}",
                  f"- Topic match on near-duplicates: {r.get('retrieval_category_match_near_duplicate')}",
                  f"- Topic match on novel queries: {r.get('retrieval_category_match_novel')}"]
    lines += ["", "## Limitations", ""] + [f"- {x}" for x in out["limitations"]] + [""]
    (OUT / "GROUPED_RETRIEVAL_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print("wrote", OUT / "grouped_retrieval.json")


if __name__ == "__main__":
    main()
