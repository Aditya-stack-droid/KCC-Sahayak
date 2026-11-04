#!/usr/bin/env python
"""Comparative retrieval evaluation on the grouped test split.

Runs, on the **same sampled grouped-test queries** (seed 42):

* dense (reuses the committed grouped FAISS index),
* TF-IDF (reuses the committed grouped TF-IDF matrix),
* hybrid (alpha=0.4 fusion over the union of dense and TF-IDF candidates),
* BM25 (from ``bm25.py`` results, same sample),
* cross-encoder reranking (from ``cross_encoder.py`` results, if present).

Reports ranking metrics (relevance = weak-label topic match), paired bootstrap
confidence intervals for each difference vs dense, and Holm-corrected
significance. No index is rebuilt and no production artefact is modified.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (HERE, ROOT, ROOT / "experiments"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from common import (  # noqa: E402
    ART, SEED, environment, evaluate_rankings, holm, load_embeddings, load_index_metadata,
    load_split, paired_bootstrap, sample_positions, test_frame, write_json,
)

ALPHA = 0.4
POOL = 50
DEFAULT_OUT = ROOT / "results" / "retrieval_baselines" / "comparative_evaluation"


def _dense_rankings(q_emb: np.ndarray, k: int, pool: int = POOL):
    import faiss

    index = faiss.read_index(str(ART / "grouped_retrieval_index.faiss"))
    sims, ids = index.search(np.ascontiguousarray(q_emb, dtype="float32"), pool)
    return ids[:, :k], ids, sims


def _tfidf(q_texts, k: int, pool: int = POOL):
    import joblib
    import scipy.sparse as sp

    vec = joblib.load(ART / "grouped_tfidf.joblib")
    C = sp.load_npz(ART / "grouped_tfidf_matrix.npz").tocsr()
    Q = vec.transform(q_texts)
    ids = np.full((len(q_texts), pool), -1, dtype=np.int64)
    scores = np.zeros((len(q_texts), pool), dtype=np.float32)
    for start in range(0, len(q_texts), 128):
        stop = min(start + 128, len(q_texts))
        S = (Q[start:stop] @ C.T).toarray()
        part = np.argpartition(-S, pool - 1, axis=1)[:, :pool]
        ids[start:stop] = part
        scores[start:stop] = np.take_along_axis(S, part, axis=1)
    return ids[:, :k], ids, scores


def _hybrid(q_emb, dense_ids, dense_sims, tf_ids, tf_scores, emb, corpus_idx, k: int):
    out = np.full((len(dense_ids), k), -1, dtype=np.int64)
    for i in range(len(dense_ids)):
        cand = np.union1d(dense_ids[i], tf_ids[i])
        cand = cand[cand >= 0]
        if not len(cand):
            continue
        d = emb[corpus_idx[cand]] @ q_emb[i]
        smap = dict(zip(tf_ids[i].tolist(), tf_scores[i].tolist()))
        ts = np.array([smap.get(int(c), 0.0) for c in cand], dtype="float32")
        fused = ALPHA * d + (1.0 - ALPHA) * ts
        order = np.lexsort((cand, -fused))[:k]
        out[i] = cand[order]
    return out


def _load_bm25(sample: int, out_path: Path):
    p = ROOT / "results" / "retrieval_baselines" / "bm25" / "metrics.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text(encoding="utf-8"))
    if d["protocol"]["sample"] != sample:
        return None
    return d


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sample", type=int, default=2000)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "metrics.json").exists() and not args.overwrite:
        print(f"refusing to overwrite {out / 'metrics.json'} (pass --overwrite)")
        return 1

    meta = load_index_metadata()
    test_df, test_idx = test_frame()
    pos = sample_positions(args.sample)
    T = test_df.iloc[pos].reset_index(drop=True)
    q_texts = T["questions"].astype(str).tolist()
    emb = load_embeddings()
    q_emb = np.ascontiguousarray(emb[test_idx[pos]], dtype="float32")
    # grouped-index position j maps to corpus row corpus_positions[j] (train+val)
    split = load_split().to_numpy()
    corpus_positions = np.flatnonzero((split == "train") | (split == "val"))

    timings = {}
    t0 = time.perf_counter()
    dense_ids, d_ids, d_sims = _dense_rankings(q_emb, args.k)
    timings["dense_seconds"] = round(time.perf_counter() - t0, 2)
    dense_metrics = evaluate_rankings(dense_ids, T, meta, len(pos))

    t0 = time.perf_counter()
    tf_ids, tf_all_ids, tf_all_scores = _tfidf(q_texts, POOL)
    timings["tfidf_seconds"] = round(time.perf_counter() - t0, 2)
    tfidf_metrics = evaluate_rankings(tf_ids, T, meta, len(pos))

    # hybrid needs the full pool ids/scores and the corpus embeddings aligned to meta rows
    t0 = time.perf_counter()
    hy_ids = _hybrid(q_emb, d_ids, d_sims, tf_all_ids, tf_all_scores, emb, corpus_positions, args.k)
    timings["hybrid_seconds"] = round(time.perf_counter() - t0, 2)
    hybrid_metrics = evaluate_rankings(hy_ids, T, meta, len(pos))

    systems = {"dense": dense_metrics, "tfidf": tfidf_metrics, "hybrid": hybrid_metrics}

    bm25 = _load_bm25(args.sample, out)
    if bm25:
        systems["bm25"] = {**bm25["metrics"], "_per_query": bm25["per_query"]}

    ce_path = ROOT / "results" / "retrieval_baselines" / "cross_encoder" / "metrics.json"
    if ce_path.exists():
        ce = json.loads(ce_path.read_text(encoding="utf-8"))
        if ce.get("protocol", {}).get("sample") == args.sample:
            for name, block in ce.get("systems", {}).items():
                systems[name] = {**block["metrics"], "_per_query": block["per_query"]}

    # paired bootstrap vs dense on topic_hit_at_1, recall_at_5, mrr
    comparisons = {}
    for name, block in systems.items():
        if name == "dense":
            continue
        comparisons[name] = {}
        for metric in ("topic_hit_at_1", "recall_at_5", "mrr"):
            a = np.asarray(block["_per_query"][metric])
            b = np.asarray(dense_metrics["_per_query"][metric])
            comparisons[name][metric] = paired_bootstrap(a - b)

    # Holm across the primary metric (recall_at_5) for all systems
    pvals = {name: comp["recall_at_5"]["p_value"] for name, comp in comparisons.items()}
    adjusted = holm(pvals)

    table = {name: {k: v for k, v in block.items() if not k.startswith("_")}
             for name, block in systems.items()}
    metrics = {
        "protocol": {
            "split": "Batch A grouped split (StratifiedGroupKFold, seed 0)",
            "sample": int(len(pos)), "seed": SEED,
            "relevance": "weak-label topic match (proxy, not correctness)",
            "hybrid_alpha": ALPHA, "candidate_pool": POOL,
            "note": "all systems share the same sampled queries; dense/TF-IDF/hybrid use the committed grouped artifacts",
        },
        "systems": table,
        "latency_seconds": timings,
        "paired_vs_dense": comparisons,
        "holm_adjusted_p_recall_at_5": adjusted,
        "environment": environment(),
        "limitations": [
            "Relevance is a 10-class topical proxy, not agronomic correctness.",
            "Cross-encoder reranks a top-20 pool, so its recall@k is bounded by that pool.",
            "BM25 has no stemming and no Hindi morphology; the corpus is English/Romanised.",
            "Exact-match@1 is 0.0 by construction (grouped split removes exact duplicates).",
        ],
    }
    write_json(out / "metrics.json", metrics)
    pd.DataFrame(
        [{"system": n, **v} for n, v in table.items()]
    ).to_csv(out / "summary.csv", index=False)

    print(json.dumps({n: {"topic@1": v["topic_match_at_1"], "r@5": v["recall_at_5"],
                          "mrr": v["mrr_at_10"], "ndcg@5": v["ndcg_at_5"]}
                      for n, v in table.items()}, indent=2))
    print("latency:", timings)
    print("written:", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
