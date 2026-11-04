# Multi-reference retrieval evaluation (Improvement 3)

Scores a retrieved answer against **every** distinct answer KCC advisors logged
for the same question group, not just the one answer attached to the evaluation
row. Additive: reads the original corpus and Batch A artefacts, writes only
under `experiments/multi_reference/artifacts/` and `results/multi_reference/`.
Production retrieval is untouched.

## What it answers — and what it does not

It answers: *does the retrieved answer agree with some historical advisor answer
to this question?* It does **not** answer whether the advice is agronomically
correct. References are logged advisor answers, not verified gold; some
contradict each other (see Improvement 1). Semantic similarity is reported
separately and is not correctness.

## Reference definition

For a test row in question group *g*:

* candidates are the answers logged for rows of *g* on the **test** side of
  Batch A's grouped split — never index-side rows, so no leakage is possible;
* only informative answers qualify (`src/answer_quality`);
* answers are deduplicated by normalised text;
* the row's own answer is a pool member only when it is informative.

## Metrics

| Metric | Meaning |
|---|---|
| `own_f1` | F1 against the row's own logged answer (the original metric) |
| `max_f1` | best F1 over the pool plus own answer (≥ own by construction) |
| `max_other_f1` | best F1 over the pool **excluding** the own answer — the honest multi-reference measure |
| `random_other_f1` | mean over five seeded random non-own references — chance control |
| `max_semantic` | max MiniLM cosine to any pool answer — wording closeness only |

## Protocol

* **P1 (primary, comparable with Batch A):** unscreened grouped index; dense on
  all grouped test rows; hybrid (α=0.4) on the same fixed 3,000-query sample;
  dense recomputed on that sample for the paired comparison.
* **P2 (deployed screening):** same 3,000 queries, informative-only candidates.

A hard **reproduction gate** first recomputes Batch A's own-reference F1 and
topic-match numbers from the rebuilt rankings; if any differs by more than
0.001 the run aborts instead of reporting incomparable new metrics.

## Run

```bash
OMP_NUM_THREADS=1 .venv312/bin/python experiments/multi_reference/run_evaluation.py
OMP_NUM_THREADS=1 .venv312/bin/python experiments/multi_reference/mref_report.py
OMP_NUM_THREADS=1 .venv312/bin/python -m pytest tests/unit/test_multi_reference.py -q
```

Fixed seeds: hybrid sample 42, bootstrap 0, random-reference draws 42.

Module files are prefixed `mref_` (`mref_refdata`, `mref_metrics`,
`mref_embeddings`, `mref_ranking`, `mref_report`) so they cannot shadow another
suite's `metrics`/`report` module in `sys.modules` when the full test suite runs.

## Outputs

```
experiments/multi_reference/artifacts/
  reference_pools.json        group_id -> answer ids
  reference_answers.csv       answer_id, group_id, row_id, n_words, text_sha1
  pool_stats.json             availability + diversity diagnostics
  reference_answer_embeddings.npy / .json
  retrieved_answer_embeddings.npy / .json
results/multi_reference/
  metrics.json
  per_query_dense_all.csv
  per_query_paired_sample.csv
  REPORT.md
```

No answer text is written to any result file: reports and artefacts carry ids,
hashes and metrics only. The pools point at rows already present in the corpus,
so no personal information is newly exposed.

## Limitations

* References are historical answers, not correctness labels.
* Grouping is transitive at cosine ≥ 0.95, so large groups can merge related
  questions.
* Contradictory answers can share a pool; max-reference F1 rewards matching any.
* One grouped split seed only (the index rebuild is expensive).
* Human relevance and expert agronomic review remain PENDING (Batch B Phase 2).
