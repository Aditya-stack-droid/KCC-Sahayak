#!/usr/bin/env python
"""Multilingual cross-encoder reranking (experimental, standalone).

Architecture: candidate retrieval (dense / BM25 / hybrid) -> top-20 pool ->
cross-encoder scoring -> reranked list.

Model: ``cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`` (multilingual mMARCO
reranker, Apache-2.0). It is loaded locally; no paid API. It is **not** a
dependency of the original application.

For fairness the un-reranked pool order is scored too (``*_pool20``), so the
reranked systems are compared against the *same* candidate pool.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (HERE, ROOT, ROOT / "experiments"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from bm25 import BM25  # noqa: E402
from common import (  # noqa: E402
    ART, SEED, environment, evaluate_rankings, load_embeddings, load_index_metadata,
    load_split, sample_positions, test_frame, write_json,
)
from compare import _dense_rankings, _hybrid, _tfidf  # noqa: E402

MODEL_NAME = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
POOL = 20
ALPHA = 0.4


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sample", type=int, default=2000)
    parser.add_argument("--pool", type=int, default=POOL)
    parser.add_argument("--out", default=str(ROOT / "results" / "retrieval_baselines" / "cross_encoder"))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "metrics.json").exists() and not args.overwrite:
        print(f"refusing to overwrite {out / 'metrics.json'} (pass --overwrite)")
        return 1

    t0 = time.perf_counter()
    meta = load_index_metadata()
    test_df, test_idx = test_frame()
    pos = sample_positions(args.sample)
    T = test_df.iloc[pos].reset_index(drop=True)
    q_texts = T["questions"].astype(str).tolist()
    emb = load_embeddings()
    q_emb = np.ascontiguousarray(emb[test_idx[pos]], dtype="float32")
    split = load_split().to_numpy()
    corpus_positions = np.flatnonzero((split == "train") | (split == "val"))
    data_seconds = time.perf_counter() - t0

    # ---- candidate pools
    t0 = time.perf_counter()
    _, dense_pool, dense_sims = _dense_rankings(q_emb, args.pool, pool=args.pool)
    bm25 = BM25(meta["questions"].astype(str).tolist())
    bm25_pool = np.full((len(pos), args.pool), -1, dtype=np.int64)
    for i, q in enumerate(q_texts):
        for j, (doc, _score) in enumerate(bm25.search(q, args.pool)[:args.pool]):
            bm25_pool[i, j] = doc
    tf_ids, tf_all_ids, tf_all_scores = _tfidf(q_texts, 50)
    hybrid_pool = _hybrid(q_emb, dense_pool, dense_sims, tf_all_ids, tf_all_scores,
                          emb, corpus_positions, args.pool)
    pools = {"dense": dense_pool, "bm25": bm25_pool, "hybrid": hybrid_pool}
    pool_seconds = time.perf_counter() - t0

    # ---- cross-encoder
    from sentence_transformers import CrossEncoder

    t0 = time.perf_counter()
    ce = CrossEncoder(MODEL_NAME, max_length=256)
    load_seconds = time.perf_counter() - t0

    systems = {}
    t_score = 0.0
    for name, pool in pools.items():
        # pool baseline: same candidates, original order, first 10
        base = evaluate_rankings(pool[:, :10], T, meta, len(pos))
        base_per = base.pop("_per_query")
        systems[f"{name}_pool20"] = {"metrics": base, "per_query": base_per}

        ranked = np.full((len(pos), min(10, args.pool)), -1, dtype=np.int64)
        t0 = time.perf_counter()
        for i in range(len(pos)):
            ids = pool[i]
            ids = ids[ids >= 0]
            if not len(ids):
                continue
            docs = meta["questions"].iloc[ids].astype(str).tolist()
            scores = ce.predict([(q_texts[i], d) for d in docs], show_progress_bar=False)
            order = np.lexsort((ids, -np.asarray(scores)))  # tie-break by doc id
            ranked[i] = ids[order][:ranked.shape[1]]
        t_score += time.perf_counter() - t0

        m = evaluate_rankings(ranked, T, meta, len(pos))
        per = m.pop("_per_query")
        systems[f"{name}_ce"] = {"metrics": m, "per_query": per}
        print(f"{name}_ce:", {k: m[k] for k in ("topic_match_at_1", "recall_at_5", "mrr_at_10", "ndcg_at_5")})

    res = {
        "system": "cross_encoder_reranking",
        "protocol": {
            "sample": int(len(pos)), "seed": SEED, "pool": args.pool,
            "relevance": "weak-label topic match (proxy, not correctness)",
            "model": MODEL_NAME, "licence": "Apache-2.0 (mMARCO cross-encoder)",
            "rerank_pool_note": "recall@k for *_ce is bounded by the top-20 pool",
        },
        "timing_seconds": {"data_load": round(data_seconds, 1), "pools": round(pool_seconds, 1),
                           "model_load": round(load_seconds, 1), "rerank_total": round(t_score, 1),
                           "rerank_per_query_ms": round(1000 * t_score / max(1, len(pos) * len(pools)), 2)},
        "systems": systems,
        "environment": environment(),
        "limitations": [
            "Cross-encoder reranks a 20-document pool; it cannot recover documents outside the pool.",
            "Relevance is a 10-class topical proxy, not agronomic correctness.",
            "mMARCO is multilingual but the corpus is English/Romanised, so Hindi reranking is untested.",
            "Cross-encoder latency is much higher than bi-encoder retrieval; measured on CPU only.",
        ],
    }
    write_json(out / "metrics.json", res)
    print("timing:", res["timing_seconds"])
    print("written:", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
