# Grouped intent evaluation (Batch A)

Protocol: exact-duplicate + MiniLM cosine >= 0.95 question groups; StratifiedGroupKFold (7 folds, fold 0 = test) over the 3,496 LLM-labelled queries; five split seeds; refit with the documented deployed recipe.

**Labels are still LLM-generated. This removes split leakage, not annotator bias.**

## Aggregate (mean ± sd over 5 seeds)

| Model | Accuracy | Macro F1 | Weighted F1 |
|---|---:|---:|---:|
| Refit, embedding + char TF-IDF + LR (with augmentation) | 0.9372 ± 0.0045 | 0.9205 ± 0.0048 | 0.9358 ± 0.0047 |
| Refit, same recipe without paraphrases | 0.9372 ± 0.0054 | — | — |

## Per seed

| Seed | Train | Test | Acc (with aug) | Acc (no aug) | Acc (deployed artefact) | Test rows seen in deployed training |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 2996 | 500 | 0.9420 | 0.9440 | 0.9940 | 85.00% |
| 1 | 2995 | 501 | 0.9321 | 0.9321 | 0.9940 | 83.63% |
| 2 | 2996 | 500 | 0.9320 | 0.9300 | 0.9960 | 87.80% |
| 3 | 2996 | 500 | 0.9420 | 0.9380 | 0.9840 | 85.20% |
| 4 | 2996 | 500 | 0.9380 | 0.9420 | 0.9940 | 85.40% |

## Novelty buckets (refit, mean over seeds)

| Bucket | N (seed 0) | Accuracy |
|---|---:|---:|
| near_duplicate | 78 | 0.9630 ± 0.0219 |
| similar | 236 | 0.9580 ± 0.0122 |
| novel | 186 | 0.9025 ± 0.0107 |

## Gold set (independent phrasing, refit model)

| Seed | Accuracy | English | Hinglish | Hindi |
|---:|---:|---:|---:|---:|
| 0 | 0.8933 | 0.9200 | 0.9000 | 0.8600 |
| 1 | 0.8933 | 0.9200 | 0.9000 | 0.8600 |
| 2 | 0.8800 | 0.9200 | 0.8600 | 0.8600 |
| 3 | 0.8733 | 0.9200 | 0.8400 | 0.8600 |
| 4 | 0.8800 | 0.9400 | 0.8400 | 0.8600 |

## Original row-level reference (not directly comparable)

- Silver accuracy: 0.9355 (CI [0.9153, 0.9556])
- Gold accuracy: 0.9
- Source: `results/research_eval.json (original row-level split, LLM labels)`

## Limitations

- Meaning labels are LLM-generated, not human-verified; this pipeline removes split leakage, not annotator bias.
- The annotated pool is 3,496 queries; stratified grouping can only redistribute these rows.
- Grouping uses a k=64 candidate cap and a 0.95 threshold; residual high-similarity pairs are reported per seed.
- The gold set shares its author with the paraphrases, so its numbers remain optimistic.
