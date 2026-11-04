# Improvement 4 — out-of-scope detection

Generated from `results/out_of_scope/metrics.json` (2026-11-04T11:30:07+00:00). Every value below is read from that file.

> **Evidence class: synthetic negatives.** Out-of-scope queries were authored for this study by its authors. They are not real farmer traffic and are not a representative sample. In-scope queries are real logged KCC questions from Batch A's grouped test split.

## 1. What this measures, and what it does not

The task is binary: is a query agricultural (in scope for KCC-Sahayak) or not? Two standalone detectors are compared — a kNN distance rule over the existing grouped index, and a logistic-regression baseline over the existing MiniLM embeddings. Neither is wired into production.

- It does **not** estimate real-world out-of-scope prevalence; the synthetic negatives are not a sample of what callers actually ask.
- It does **not** measure multilingual in-scope performance: the corpus is English by the project register detector, so Hindi and Hinglish appear only as synthetic negatives.
- It does **not** replace the production intent classifier, retriever or safety gate.

## 2. Data and split integrity

| Quantity | Value |
|---|---:|
| Corpus rows (Batch A grouped split) | 138,875 |
| Corpus test rows (agricultural) | 13,831 |
| Synthetic out-of-scope queries | 180 |
| Synthetic question families | 60 |
| Categories | 12 |
| Families per split (train / val / test) | 36 / 12 / 12 |
| Queries per split (train / val / test) | 108 / 36 / 36 |
| Synthetic families spanning splits | 0 |
| Corpus question groups spanning splits | 0 |

Each family holds the English, Hindi and Hinglish realisation of one question, and the family is the split unit, so no translation or paraphrase of a test question is visible during training or threshold selection.

Corpus test register mix (5,000-question sample): {'en': 5000}.

## 3. Threshold selection (validation only)

| Approach | Threshold | Validation FRR | Validation recall | n (in / out) |
|---|---:|---:|---:|---:|
| A. kNN distance | 0.7670 | 2.0% | 100.0% | 5,000 / 36 |
| B. Embedding + logistic regression | 0.8093 | 2.0% | 100.0% | 5,000 / 36 |

max out-of-scope recall subject to validation FRR <= 0.02.

## 4. Benchmark results

Benchmark: 144 agricultural grouped-test queries + 36 synthetic out-of-scope queries. Positive class = out-of-scope.

| Approach | Precision | Recall | F1 | F1 95% CI | Balanced acc. | AUROC |
|---|---:|---:|---:|---|---:|---:|
| A. kNN distance | 0.8537 | 0.9722 | 0.9091 | [0.8315, 0.9706] | 0.9653 | 0.9873 |
| B. Embedding + logistic regression | 0.8780 | 1.0000 | 0.9351 | [0.871, 0.9851] | 0.9826 | 0.9927 |

| Approach | False acceptance rate (out-of-scope let through) | False rejection rate (agricultural rejected) |
|---|---:|---:|
| A. kNN distance | 2.8% | 4.2% |
| B. Embedding + logistic regression | 0.0% | 3.5% |

### Confusion matrices

| Approach | TP | FN (out-of-scope accepted) | FP (agricultural rejected) | TN |
|---|---:|---:|---:|---:|
| A. kNN distance | 35 | 1 | 6 | 138 |
| B. Embedding + logistic regression | 36 | 0 | 5 | 139 |

## 5. Full agricultural test set (not a sample)

| Approach | Agricultural test rows | Rejected | False rejection rate |
|---|---:|---:|---:|
| A. kNN distance | 13,831 | 864 | 6.2% |
| B. Embedding + logistic regression | 13,831 | 502 | 3.6% |

## 6. By register (synthetic negatives only)

| Register | n | A. kNN rejection | B. LR rejection |
|---|---:|---:|---:|
| en | 12 | 100.0% (12/12) | 100.0% (12/12) |
| hi | 12 | 100.0% (12/12) | 100.0% (12/12) |
| hinglish | 12 | 91.7% (11/12) | 100.0% (12/12) |

No in-scope Hindi or Hinglish queries exist in the corpus, so this table says nothing about false rejections for those registers.

## 7. Ambiguous and mixed-intent behaviour

These sets are descriptive only: they are excluded from threshold selection and from the headline benchmark.

| Set | Approach | n | in_scope | ambiguous | out_of_scope | Rejected |
|---|---|---:|---:|---:|---:|---:|
| Ambiguous | A. kNN distance | 16 | 1 | 11 | 4 | 75.0% |
| Ambiguous | B. Embedding + logistic regression | 16 | 6 | 7 | 3 | 56.2% |
| Mixed intent | A. kNN distance | 16 | 1 | 15 | 0 | 93.8% |
| Mixed intent | B. Embedding + logistic regression | 16 | 6 | 10 | 0 | 50.0% |

Agricultural-evidence override: a query scored below the rejection band but containing a crop/pest/agronomy term is flagged `ambiguous` (needs review) rather than silently rejected.

| Set | n | With agricultural evidence |
|---|---:|---:|
| Agricultural validation sample | 5000 | 4125 |
| Synthetic test negatives | 36 | 0 |
| Ambiguous set | 16 | 10 |
| Mixed-intent set | 16 | 16 |

## 8. Threshold sensitivity (descriptive, not used for selection)

| Approach | Threshold | FAR | FRR | F1 |
|---|---:|---:|---:|---:|
| A. kNN distance | 0.3 | 97.2% | 0.0% | 0.0541 |
| A. kNN distance | 0.35 | 91.7% | 0.0% | 0.1538 |
| A. kNN distance | 0.4 | 88.9% | 0.0% | 0.2000 |
| A. kNN distance | 0.45 | 66.7% | 0.7% | 0.4898 |
| A. kNN distance | 0.5 | 41.7% | 0.7% | 0.7241 |
| A. kNN distance | 0.55 | 33.3% | 0.7% | 0.7869 |
| A. kNN distance | 0.6 | 25.0% | 2.1% | 0.8182 |
| A. kNN distance | 0.65 | 11.1% | 2.1% | 0.9014 |
| A. kNN distance | 0.7 | 5.6% | 2.1% | 0.9315 |
| A. kNN distance | 0.75 | 2.8% | 3.5% | 0.9211 |
| A. kNN distance | 0.8 | 2.8% | 8.3% | 0.8434 |
| B. Embedding + logistic regression | 0.05 | 50.0% | 0.0% | 0.6667 |
| B. Embedding + logistic regression | 0.15 | 25.0% | 1.4% | 0.8308 |
| B. Embedding + logistic regression | 0.25 | 19.4% | 1.4% | 0.8657 |
| B. Embedding + logistic regression | 0.35 | 11.1% | 1.4% | 0.9143 |
| B. Embedding + logistic regression | 0.45 | 2.8% | 1.4% | 0.9589 |
| B. Embedding + logistic regression | 0.55 | 2.8% | 2.8% | 0.9333 |
| B. Embedding + logistic regression | 0.65 | 0.0% | 2.8% | 0.9474 |
| B. Embedding + logistic regression | 0.75 | 0.0% | 2.8% | 0.9474 |
| B. Embedding + logistic regression | 0.85 | 0.0% | 5.6% | 0.9000 |
| B. Embedding + logistic regression | 0.95 | 0.0% | 11.1% | 0.8182 |

## 9. Interpretation

- **A. kNN distance** rejects 97.2% of synthetic negatives while rejecting 4.2% of agricultural benchmark queries; AUROC 0.9873.
- **B. Embedding + logistic regression** rejects 100.0% of synthetic negatives while rejecting 3.5% of agricultural benchmark queries; AUROC 0.9927.
- Neither detector is integrated into production; both are standalone and evaluated offline.
- A higher synthetic recall is not evidence of real-world safety or usefulness: the negatives share author, domain and vocabulary with each other.

## 10. Limitations

- Out-of-scope queries are authored synthetic text, not real KCC traffic; these are synthetic-negative results, not deployment estimates.
- The synthetic set is small (180 queries / 60 families) and same-author, so confidence intervals are wide and vocabulary overlap with training templates cannot be excluded.
- In-scope corpus queries are English by the project register detector; no Hindi or Hinglish in-scope evaluation data exists, so no multilingual in-scope claim is made. Hindi/Hinglish appear only as synthetic negatives.
- Thresholds were selected on validation data; the test split was scored once, but the ambiguity band widths are design choices, not tuned quantities.
- The agricultural-evidence override uses an English glossary only; it cannot flag Devanagari or Roman-Hindi farm terms.
- No human annotation of scope exists; agreement with the synthetic labels is not a validated measure of real-world out-of-scope detection.
- The kNN reference set and the corpus are the same logged call text, so in-scope queries are lexically and semantically close to the index by construction; that is what the detector exploits.

Environment: Python 3.12.13, macOS-26.6.2-arm64-arm-64bit; numpy 2.5.4, scikit-learn 1.9.1, faiss 1.15.1, sentence-transformers 6.2.0.

## 11. Reproduce

```bash
OMP_NUM_THREADS=1 .venv312/bin/python experiments/out_of_scope/run_oos_evaluation.py
OMP_NUM_THREADS=1 .venv312/bin/python experiments/out_of_scope/oos_report.py
OMP_NUM_THREADS=1 .venv312/bin/python -m pytest tests/unit/test_out_of_scope.py -q
```

Fixed seeds: dataset 42, bootstrap 0. Thresholds are selected on validation data only; the test split is scored once.
