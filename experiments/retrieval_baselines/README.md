# Project B — advanced retrieval baselines (experimental, not integrated)

Compares **BM25**, **multilingual cross-encoder reranking**, and a **fine-tuned
encoder** against the existing dense / TF-IDF / hybrid retrieval on the Batch A
grouped split. Nothing here is wired into production; no index, model, dataset or
threshold is modified.

**Relevance = weak-label topic match** (retrieved `query_type` == test `query_type`) —
a topical proxy, **not** agricultural correctness. Sample: 2,000 grouped-test queries,
seed 42, same queries for every system. Exact-match@1 is 0.0 by construction (the grouped
split removes exact duplicates; no query retrieves itself).

## Results (2,000 queries, CPU)

| System | topic@1 | Recall@5 | MRR@10 | nDCG@5 | answer F1@1 |
|---|---:|---:|---:|---:|---:|
| dense | 0.8395 | 0.9440 | 0.8880 | 0.8996 | 0.2932 |
| tfidf | 0.8010 | 0.9595 | 0.8655 | 0.8863 | 0.2910 |
| **hybrid** | **0.8700** | **0.9620** | **0.9094** | **0.9225** | **0.3402** |
| bm25 | 0.8290 | 0.9600 | 0.8852 | 0.9014 | 0.3253 |
| dense + cross-encoder | 0.8210 | 0.9520 | 0.8764 | 0.8946 | 0.2706 |
| bm25 + cross-encoder | 0.8305 | 0.9595 | 0.8855 | 0.9041 | 0.3037 |
| hybrid + cross-encoder | 0.8385 | 0.9565 | 0.8866 | 0.9056 | 0.3144 |

Paired bootstrap vs dense (2,000 resamples, Holm-corrected on Recall@5):

| System | Δ topic@1 [95% CI] | Δ Recall@5 [95% CI] | Holm p (R@5) |
|---|---|---|---:|
| tfidf | -0.0385 [-0.0605, -0.0175] | +0.0155 [0.0030, 0.0275] | 0.048 |
| hybrid | +0.0305 [0.0130, 0.0490] | +0.0180 [0.0075, 0.0295] | 0.036 |
| bm25 | -0.0105 [-0.0295, 0.0090] | +0.0160 [0.0045, 0.0275] | 0.048 |
| dense_ce | -0.0185 [-0.0355, -0.0010] | +0.0080 [-0.0025, 0.0185] | 0.322 |
| bm25_ce | -0.0090 [-0.0270, 0.0100] | +0.0155 [0.0045, 0.0275] | 0.036 |
| hybrid_ce | -0.0010 [-0.0195, 0.0185] | +0.0125 [0.0010, 0.0240] | 0.111 |

**Honest conclusion:** hybrid retrieval is the best system overall and is significantly
better than dense on every metric. BM25 is a strong, cheap baseline. **Cross-encoder
reranking does not improve topical retrieval here** — compared with the same 20-document
pool it is neutral-to-worse (`dense_ce` significantly lowers topic@1, `hybrid_ce` lowers it
by ~3 points). A more complicated model is not automatically better.

## Fine-tuned encoder

Implemented (`fine_tuning.py`) with `MultipleNegativesRankingLoss` on grouped-train
(query → own logged answer) pairs; test rows never used. **It could not be executed in
this environment**: `SentenceTransformer.fit` requires the `datasets` package, which is not
part of the project environment and was deliberately **not** installed (to avoid adding a
dependency the original app does not need). The feasibility analysis, training config,
seed and reduced-corpus evaluation design are recorded in
`results/retrieval_baselines/fine_tuning/metrics.json`. Re-run after `pip install datasets`
in an isolated environment.

Feasibility: 111,213 grouped-train rows, 101,590 informative, 8.65% call-log notes; weak,
noisy positives with no human relevance labels and a 10-class topic proxy — fine-tuning may
overfit label noise.

## Cost (measured on this machine, CPU)

| Approach | Model | Index | Build | Query latency |
|---|---|---|---|
| BM25 | none | 12,477-term inverted index (in memory) | 0.9 s | p50 222 ms, p95 341 ms |
| Dense (reused) | MiniLM-L12-v2, 117.7 M params (~470 MB fp32) | FAISS 183 MB | — (committed) | ~0.4 ms/query |
| Cross-encoder | mMiniLMv2-L12-H384, 117.6 M params (~470 MB fp32) | — | 24 s load | 45 ms/query/pool (top-20) |

No GPU was used; no GPU timings are reported.

## Files

`common.py` (data, metrics, paired bootstrap, Holm) · `bm25.py` · `cross_encoder.py` ·
`fine_tuning.py` · `compare.py` (orchestrator) · `report.py` · `__init__.py`

## Reproduce

```bash
OMP_NUM_THREADS=1 .venv312/bin/python experiments/retrieval_baselines/bm25.py --sample 2000 --overwrite
OMP_NUM_THREADS=1 .venv312/bin/python experiments/retrieval_baselines/cross_encoder.py --sample 2000 --overwrite
OMP_NUM_THREADS=1 .venv312/bin/python experiments/retrieval_baselines/compare.py --sample 2000 --overwrite
OMP_NUM_THREADS=1 .venv312/bin/python experiments/retrieval_baselines/fine_tuning.py --feasibility-only --overwrite
OMP_NUM_THREADS=1 .venv312/bin/python experiments/retrieval_baselines/report.py
OMP_NUM_THREADS=1 .venv312/bin/python -m pytest tests/unit/test_retrieval_baselines.py -q
```

Order matters: `compare.py` merges the BM25 and cross-encoder results, so run those first.
