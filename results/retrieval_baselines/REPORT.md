# Project B — advanced retrieval baselines

**Evidence class:** evaluation on the Batch A grouped split (leakage-resistant),
relevance = weak-label topic match (a topical proxy, **not** agricultural correctness).
No production index, model or threshold was modified. Nothing is integrated.

Sample: **2000** grouped-test queries (seed 42); candidate pool for fusion/reranking: 50.

## 1. Comparative results

| System | topic@1 | Recall@5 | MRR@10 | nDCG@5 | exact@1 | answer F1@1 |
|---|---:|---:|---:|---:|---:|---:|
| dense | 0.8395 | 0.9440 | 0.8880 | 0.8996 | 0.0000 | 0.2932 |
| tfidf | 0.8010 | 0.9595 | 0.8655 | 0.8863 | 0.0000 | 0.2910 |
| hybrid | 0.8700 | 0.9620 | 0.9094 | 0.9225 | 0.0000 | 0.3402 |
| bm25 | 0.8290 | 0.9600 | 0.8852 | 0.9014 | 0.0000 | 0.3253 |
| dense_pool20 | 0.8395 | 0.9440 | 0.8880 | 0.8996 | 0.0000 | 0.2930 |
| dense_ce | 0.8210 | 0.9520 | 0.8764 | 0.8946 | 0.0000 | 0.2706 |
| bm25_pool20 | 0.8290 | 0.9600 | 0.8852 | 0.9014 | 0.0000 | 0.3253 |
| bm25_ce | 0.8305 | 0.9595 | 0.8855 | 0.9041 | 0.0000 | 0.3037 |
| hybrid_pool20 | 0.8700 | 0.9620 | 0.9094 | 0.9225 | 0.0000 | 0.3398 |
| hybrid_ce | 0.8385 | 0.9565 | 0.8866 | 0.9056 | 0.0000 | 0.3144 |

`*_pool20` is the un-reranked candidate pool (same 20 candidates); `*_ce` is the same
pool reranked by the cross-encoder. Exact-match@1 is 0.0 by construction (grouped split
removes exact duplicates).

## 2. Paired statistical comparison (vs dense)

| System | Δ topic@1 [95% CI] | Δ Recall@5 [95% CI] | Δ MRR [95% CI] | Holm-adj p (R@5) |
|---|---|---|---|---:|
| tfidf | -0.0385 [-0.0605, -0.0175] | +0.0155 [0.003, 0.0275] | -0.0225 [-0.0384, -0.0077] | 0.048 |
| hybrid | +0.0305 [0.013, 0.049] | +0.0180 [0.0075, 0.0295] | +0.0213 [0.0084, 0.0344] | 0.036 |
| bm25 | -0.0105 [-0.0295, 0.009] | +0.0160 [0.0045, 0.0275] | -0.0028 [-0.0166, 0.0105] | 0.048 |
| dense_ce | -0.0185 [-0.0355, -0.001] | +0.0080 [-0.0025, 0.0185] | -0.0117 [-0.024, -0.0003] | 0.322 |
| bm25_ce | -0.0090 [-0.027, 0.01] | +0.0155 [0.0045, 0.0275] | -0.0025 [-0.0149, 0.0103] | 0.036 |
| hybrid_ce | -0.0010 [-0.0195, 0.0185] | +0.0125 [0.001, 0.024] | -0.0014 [-0.0146, 0.0122] | 0.111 |

Paired bootstrap, 2,000 resamples, same queries for both systems; Holm across the
comparisons on Recall@5.

## 3. Cross-encoder reranking (top-20 pool)

- Model: `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` (Apache-2.0), pool 20.
- Model load 24.3s; reranking 45.4 ms per query per pool.

| System | topic@1 | Recall@5 | MRR@10 | nDCG@5 |
|---|---:|---:|---:|---:|
| dense_ce | 0.8210 | 0.9520 | 0.8764 | 0.8946 |
| bm25_ce | 0.8305 | 0.9595 | 0.8855 | 0.9041 |
| hybrid_ce | 0.8385 | 0.9565 | 0.8866 | 0.9056 |

## 4. Fine-tuned encoder

- Status: **failed**.
- Supervision: weak: query paired with its own logged advisor answer; no human relevance labels.
- Grouped-train rows: 111,213; informative: 101,590; call-log-note share: 8.65%.
- Suitability: limited: weak, noisy positives and a 10-class topic proxy mean a fine-tune can overfit label noise and may not improve topical retrieval; run as a bounded check only.
- Blocker: sentence-transformers `fit` requires the `datasets` package, which is not part of the project environment. It was NOT installed, to avoid adding a dependency the original application does not need. The feasibility analysis and full training / evaluation configuration above are complete and reproducible once `datasets` is available in an isolated environment.

## 5. Computational cost

- BM25: build 0.9s over 125,044 docs, vocab 12,477; query p50 222.164 ms, p95 340.977 ms (CPU).
- Cross-encoder: CPU, load 24.3s, 45.4 ms/query/pool (top-20).
- Batch retrieval timings (2,000 queries): dense 1.45s, TF-IDF 7.7s, hybrid fusion 0.22s (CPU).
- No GPU was used; no GPU timings are reported.

## 6. Limitations

- Relevance is a 10-class topical proxy, not agronomic correctness.
- Cross-encoder reranks a top-20 pool, so its recall@k is bounded by that pool.
- BM25 has no stemming and no Hindi morphology; the corpus is English/Romanised.
- Exact-match@1 is 0.0 by construction (grouped split removes exact duplicates).
- Cross-encoder reranks a 20-document pool; it cannot recover documents outside the pool.
- Relevance is a 10-class topical proxy, not agronomic correctness.
- mMARCO is multilingual but the corpus is English/Romanised, so Hindi reranking is untested.
- Cross-encoder latency is much higher than bi-encoder retrieval; measured on CPU only.
