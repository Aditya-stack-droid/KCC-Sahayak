"""Rebuild the Batch A dense and hybrid rankings (Improvement 3).

The evaluation must compare systems on identical queries under identical
conditions, and it must be anchored to the frozen Batch A numbers before any
new metric is reported. This module rebuilds rankings from Batch A's own
artefacts, replaying exactly the same protocol:

* FAISS ``IndexFlatIP`` over grouped train+val, ``k=200`` dense candidates;
* word TF-IDF (1-2 grams, sublinear) fitted on the same grouped train+val rows;
* fusion ``fused = alpha * dense_cosine + (1 - alpha) * tfidf_cosine`` with
  ``alpha = 0.4``, the deployed weight;
* the fixed 3,000-query hybrid sample drawn with ``default_rng(42)`` and the
  same ``choice`` call, so the sample is identical by construction.

It keeps 200 ranked ids per query (Batch A kept 10) so a screened
(informative-only) variant can be derived without re-ranking.
"""
from __future__ import annotations

import time

import faiss
import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp

import mref_refdata as refdata

ALPHA = 0.4
SEED = 42
SAMPLE = 3000
POOL = 200


def load_artifacts():
    """Load Batch A's grouped index, metadata, embeddings and TF-IDF."""
    df = refdata.load_grouped_corpus()
    split = pd.read_csv(refdata.GROUPED_ART / "corpus_split.csv")
    if len(split) != len(df) or not np.array_equal(
            split["qid"].to_numpy(), df["qid"].to_numpy()):
        raise SystemExit("corpus_split.csv does not align with the corpus")

    emb = np.load(refdata.GROUPED_ART / "corpus_embeddings.npy")
    s = split["split"].to_numpy()
    corpus_idx = np.flatnonzero((s == "train") | (s == "val"))
    test_idx = np.flatnonzero(s == "test")

    meta = df.iloc[corpus_idx].reset_index(drop=True)
    valid = np.load(refdata.GROUPED_ART / "grouped_retrieval_informative_mask.npy")
    index = faiss.read_index(str(refdata.GROUPED_ART / "grouped_retrieval_index.faiss"))
    vec = joblib.load(refdata.GROUPED_ART / "grouped_tfidf.joblib")
    matrix = sp.load_npz(refdata.GROUPED_ART / "grouped_tfidf_matrix.npz").tocsr()

    if len(meta) != index.ntotal or len(meta) != len(valid) or len(meta) != matrix.shape[0]:
        raise SystemExit("grouped index, metadata, mask and TF-IDF matrix disagree")
    return df, split, emb, corpus_idx, test_idx, meta, valid, index, vec, matrix


def fixed_sample(n_test: int, size: int = SAMPLE, seed: int = SEED) -> np.ndarray:
    """The exact sample used by Batch A's hybrid evaluation."""
    rng = np.random.default_rng(seed)
    sample = rng.choice(n_test, size=min(size, n_test), replace=False)
    sample.sort()
    return sample


def rank_hybrid(sample_tf, sample_ids, tf_ids, tf_scores, sample_emb, matrix, emb,
                corpus_idx, alpha: float = ALPHA, pool: int = POOL) -> np.ndarray:
    """Fused ranking for each query; rows padded with -1 past the candidate pool.

    The loop mirrors the Batch A script line for line so first-10 order is
    identical; only the retained depth differs (200 instead of 10).
    """
    n_queries = int(sample_tf.shape[0])
    out = np.full((n_queries, pool), -1, dtype=np.int64)
    for i in range(n_queries):
        cand = np.union1d(sample_ids[i], tf_ids[i])
        cand = cand[cand >= 0]
        if len(cand) == 0:
            continue
        dsim = emb[corpus_idx[cand]] @ sample_emb[i]
        score_map = dict(zip(tf_ids[i].tolist(), tf_scores[i].tolist()))
        missing = [int(c) for c in cand if int(c) not in score_map]
        if missing:
            extra = np.asarray((matrix[missing] @ sample_tf[i].T).todense()).ravel()
            score_map.update({int(c): float(v) for c, v in zip(missing, extra)})
        ts = np.array([score_map[int(c)] for c in cand], dtype="float32")
        fused = alpha * dsim + (1.0 - alpha) * ts
        order = np.argsort(-fused)[:pool]
        out[i, :len(order)] = cand[order]
    return out


def screened_top_k(ids: np.ndarray, valid: np.ndarray, k: int = 10) -> np.ndarray:
    """First ``k`` informative candidates per query (deployed screening), padded -1."""
    out = np.full((len(ids), k), -1, dtype=np.int64)
    valid = np.asarray(valid, dtype=bool)
    for i, row in enumerate(ids):
        kept = [int(j) for j in row if j >= 0 and valid[int(j)]][:k]
        out[i, :len(kept)] = kept
    return out


def build_rankings(df=None) -> dict:
    """Dense ranking for every test query plus the fixed-sample hybrid ranking."""
    t0 = time.time()
    df, split, emb, corpus_idx, test_idx, meta, valid, index, vec, matrix = load_artifacts()
    if df is not None and not np.array_equal(df["qid"].to_numpy(),
                                            refdata.load_grouped_corpus()["qid"].to_numpy()):
        raise SystemExit("supplied corpus frame does not match the on-disk corpus")

    Q_emb = np.ascontiguousarray(emb[test_idx])
    Q_text = df.iloc[test_idx]["questions"].astype(str).tolist()
    Q_tf = vec.transform(Q_text)
    y_test = df.iloc[test_idx]["query_type"].to_numpy()
    answers_test = df.iloc[test_idx]["answers"].astype(str).to_numpy()

    dense_sims, dense_ids = index.search(Q_emb, POOL)
    print(f"dense ranking for {len(test_idx)} queries in {time.time()-t0:.1f}s")

    sample = fixed_sample(len(test_idx))
    Qs_tf = Q_tf[sample]
    Qs_ids = dense_ids[sample]
    Qs_sims = dense_sims[sample]
    Qs_emb = Q_emb[sample]

    tf_ids = np.zeros((len(sample), POOL), dtype=np.int64)
    tf_scores = np.zeros((len(sample), POOL), dtype=np.float32)
    t0 = time.time()
    block = 128
    for start in range(0, len(sample), block):
        stop = min(start + block, len(sample))
        S = (Qs_tf[start:stop] @ matrix.T).toarray()
        part = np.argpartition(-S, POOL - 1, axis=1)[:, :POOL]
        tf_ids[start:stop] = part
        tf_scores[start:stop] = np.take_along_axis(S, part, axis=1)
    print(f"TF-IDF candidate search for {len(sample)} queries in {time.time()-t0:.1f}s")

    t0 = time.time()
    hybrid_ids = rank_hybrid(Qs_tf, Qs_ids, tf_ids, tf_scores, Qs_emb, matrix, emb,
                            corpus_idx)
    print(f"hybrid ranking for {len(sample)} queries in {time.time()-t0:.1f}s")

    return {
        "df": df,
        "meta": meta,
        "valid": valid,
        "emb": emb,
        "test_idx": test_idx,
        "sample": sample,
        "y_test": y_test,
        "answers_test": answers_test,
        "dense_ids": dense_ids,
        "dense_sims": dense_sims,
        "hybrid_ids": hybrid_ids,
        "alpha": ALPHA,
        "sample_seed": SEED,
        "n_index": int(len(meta)),
        "n_test": int(len(test_idx)),
    }
