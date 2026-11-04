# Out-of-scope detection (Improvement 4)

Standalone experiment comparing two ways to decide whether a query belongs to
agriculture at all. Additive: reads the original corpus, Batch A's grouped split
and index, and the project's existing MiniLM encoder. Writes only under
`experiments/out_of_scope/` and `results/out_of_scope/`. Production code is
untouched.

## Evidence class

**Synthetic negatives.** The 180 out-of-scope queries (60 question families ×
English / Hindi / Hinglish) were authored for this study. They are not real
farmer traffic and not a representative sample. In-scope queries are real logged
KCC questions from Batch A's grouped test split, which is English-only by the
project's register detector — so no Hindi/Hinglish *in-scope* claim is made.

## Approaches

| Approach | Score | Notes |
|---|---|---|
| A. kNN distance | mean cosine to the 5 nearest grouped-index questions | threshold on the score |
| B. Embedding + LR | logistic regression (C=1, balanced) over MiniLM embeddings | trained on 20k agricultural corpus queries + 108 synthetic negatives |

Both emit `in_scope` / `out_of_scope` / `ambiguous` plus a reason. Queries scored
below the rejection band but containing a crop/pest/agronomy term are flagged
`ambiguous` rather than silently rejected (English glossary only).

## Leakage control and policy

* A **family** is one question with its English, Hindi and Hinglish
  realisations; families are the split unit (36 train / 12 val / 12 test), so no
  translation or paraphrase of a test question is seen during training or
  threshold selection.
* In-scope rows use Batch A's grouped split; no question group spans splits.
* Threshold = highest value whose **validation** false-rejection rate is ≤2%,
  maximising out-of-scope recall. The test split is scored once. The ambiguity
  band widths are design choices, not tuned on test.

## Run

```bash
OMP_NUM_THREADS=1 .venv312/bin/python experiments/out_of_scope/run_oos_evaluation.py
OMP_NUM_THREADS=1 .venv312/bin/python experiments/out_of_scope/oos_report.py
OMP_NUM_THREADS=1 .venv312/bin/python -m pytest tests/unit/test_out_of_scope.py -q
```

Seeds: dataset 42, bootstrap 0.

## Outputs

```
experiments/out_of_scope/data/            authored synthetic + ambiguous + mixed CSVs
experiments/out_of_scope/artifacts/       synthetic-query embeddings, fitted LR model
results/out_of_scope/metrics.json         thresholds, metrics, CIs, sensitivity
results/out_of_scope/per_query_benchmark.csv
results/out_of_scope/per_query_in_scope.csv
results/out_of_scope/REPORT.md
```

## What this does not establish

* Real-world out-of-scope prevalence or detector precision on actual traffic.
* Multilingual in-scope behaviour (no Hindi/Hinglish in-scope data).
* Agronomic correctness, safety or production readiness.
* That either detector should replace or precede the production classifier; both
  remain offline experiments.
