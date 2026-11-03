# Batch A — leakage-resistant evaluation: combined report

All numbers below are read from the JSON files produced by this pipeline. No original result file was modified.

## 1. Question grouping and original-split leakage

- Corpus rows: **138,875** in **48,674** groups (threshold 0.95, k=64).
- Largest group: **1,623** rows; p95 size 7, p99 33.
- Exact test question present in train: **44.98%**.
- Test query shares a group with a train query: **70.82%**
- Test query has a >=0.95 dense neighbour in the shipped index: **71.38%**.
- Test query has no >=0.80 neighbour (novel): **4.75%**.

## 2. Grouped intent evaluation

| Model | Accuracy (mean ± sd, 5 seeds) | Macro F1 | Weighted F1 |
|---|---:|---:|---:|
| Refit, embedding + char TF-IDF + LR (augmentation) | 0.9372 ± 0.0045 | 0.9205 ± 0.0048 | 0.9358 ± 0.0047 |
| Refit, same recipe without paraphrases | 0.9372 ± 0.0054 | — | — |

Novelty buckets (refit with augmentation, mean over seeds):

| Bucket | Accuracy |
|---|---:|
| near_duplicate | 0.9630 ± 0.0219 |
| similar | 0.9580 ± 0.0122 |
| novel | 0.9025 ± 0.0107 |

Deployed artefact on the same grouped test sets (contaminated share is disclosed):

| Seed | Deployed accuracy | Share of test rows seen in its training pool |
|---:|---:|---:|
| 0 | 0.9940 | 85.00% |
| 1 | 0.9940 | 83.63% |
| 2 | 0.9960 | 87.80% |
| 3 | 0.9840 | 85.20% |
| 4 | 0.9940 | 85.40% |

Full detail: `/Users/chiranthml/Downloads/KCC-Sahayak-main/results/improved_evaluation/GROUPED_INTENT_REPORT.md`

## 3. Grouped retrieval evaluation

- Index rows: 125,044; grouped test rows: 13,831.
- Test rows with a >=0.95 neighbour in the grouped index: **0.09%**.
- Dense topic match @1: **0.8519** (95% CI [0.8462, 0.8576]); recall@5 0.9508; MRR@10 0.8961; nDCG@5 0.9071.
- Hybrid (alpha=0.4, n=3000) topic match @1: 0.8733; recall@5 0.9630; answer token-F1 0.3800.

Full detail: `/Users/chiranthml/Downloads/KCC-Sahayak-main/results/improved_evaluation/GROUPED_RETRIEVAL_REPORT.md`

## 4. What changed and what did not

- Split leakage is removed by construction: no question group appears on both sides.
- Annotator bias is **not** removed: labels, gold queries and paraphrases remain LLM-generated.
- Retrieval relevance remains a weak-label topical proxy, not agronomic correctness.
- Human verification, real speech and expert safety review remain PENDING.
