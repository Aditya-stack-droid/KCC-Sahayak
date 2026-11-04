# Improvement 3 — multi-reference retrieval evaluation

Generated from `results/multi_reference/metrics.json` (2026-11-04T11:12:14+00:00). Every value below is read from that file; nothing is estimated or typed by hand.

## 1. What this measures, and what it does not

A retrieved answer is scored against **every** distinct answer KCC advisors actually logged for the same question group, not only against the one answer attached to the evaluation row. That answers a narrow question: *does the retrieved answer agree with some historical advisor answer to this question?*

- It does **not** measure agricultural correctness. Historical advisor answers are not verified gold, and some of them contradict each other.
- Semantic similarity is reported separately and is **not** correctness.
- No reference answer is generated, synthesised or inferred; references come from the logged corpus.
- Human relevance and expert agronomic review remain **PENDING** (Batch B Phase 2 annotation infrastructure exists but has no human labels).

## 2. Do the reference pools support multi-reference evaluation?

Verdict: **YES** (pre-registered criteria: ≥50% of test rows with ≥2 references and ≥50% of sampled multi-reference groups containing genuinely different answers).

| Quantity | Value |
|---|---:|
| Grouped test rows | 13,831 |
| Test question groups | 4,850 |
| Groups with at least one reference | 4,457 |
| Distinct reference answers | 11,747 |
| Rows with ≥1 reference | 13,362 (96.6%) |
| Rows with ≥2 references | 9,669 (69.9%) |
| Rows with ≥5 references | 8,050 (58.2%) |
| Rows with no usable reference (excluded) | 469 (3.4%) |
| References per test row: mean / median / p90 / max | 175.2293 / 10.0 / 1247.0 / 1,247 |
| References per group with ≥1 reference: mean / median / p90 / max | 2.6356 / 1.0 / 3.0 / 1,247 |
| Test rows in groups with ≥100 references | 3,024 (21.9%) |
| Mean max pairwise token-F1 inside a group | 0.6201 |
| Sampled multi-reference groups with max pairwise F1 < 0.80 | 64.2% |

References are built only from the **test** side of the grouped split, so no pool can contain an answer that is also in the retrieval index.
Split check: groups spanning splits = 0 of 48,674.

## 3. Batch A reproduction gate

Result: **PASS** — the rebuilt rankings must reproduce the frozen Batch A numbers before any new metric is reported.

| Check | Batch A | Recomputed | Abs. diff | OK |
|---|---:|---:|---:|:---:|
| dense all test | own-reference token-F1 | 0.3213 | 0.3213 | 0.00004211 | yes |
| dense all test | own-reference token-F1 (informative gold) | 0.3293 | 0.3293 | 0.00002250 | yes |
| dense all test | topic match @1 | 0.8519 | 0.8519 | 0.00002683 | yes |
| dense 3k sample | own-reference token-F1 | 0.3220 | 0.3220 | 0.00003203 | yes |
| dense 3k sample | topic match @1 | 0.8403 | 0.8403 | 0.00003333 | yes |
| hybrid 3k sample | own-reference token-F1 | 0.3800 | 0.3800 | 0.00001020 | yes |
| hybrid 3k sample | topic match @1 | 0.8733 | 0.8733 | 0.00003333 | yes |

Sample seed for the hybrid comparison: 42; 3,000 queries in both systems.

## 4. Primary results (P1: unscreened index, comparable with Batch A)

`own` = the original single-reference metric; `max` = best over the pool including the row's own answer (≥ own by construction); `max-other` = best over the pool **excluding** the row's own answer; `random-other` = mean over five seeded random non-own references (chance control).

| Metric | Dense, all test rows | Dense, 3,000 sample | Hybrid, 3,000 sample |
|---|---:|---:|---:|
| Rows | 13,831 | 3,000 | 3,000 |
| Own-reference token-F1 (original metric) | 0.3217 | 0.3222 | 0.3802 |
| Max-reference token-F1 | 0.5096 | 0.5115 | 0.5409 |
| Max-other-reference token-F1 | 0.5788 | 0.5821 | 0.6041 |
| Random-other-reference token-F1 (control) | 0.2948 | 0.2931 | 0.3113 |
| Max semantic similarity (not correctness) | 0.7755 | 0.7789 | 0.7920 |

The last three rows use the `pool_ge_2` subset (n = 9,669 for dense-all), because excluding the query's own answer is only defined when another reference exists.

Rows where the single reference scored below 0.5 but another advisor answer to the same question matched at ≥0.5 (dense, all test rows):

- Comparable rows: 9,658
- Single-reference misses (own F1 < 0.5): 7,212
- Recovered by another reference: 3,773 (52.3% of misses)

## 5. Paired statistical comparisons

Paired bootstrap over identical queries (2,000 resamples, seed 0); NaNs are dropped pairwise and `n` reports the usable pairs.

| Comparison | n | Mean delta | 95% CI | Improved | Worse | Cohen's d_z |
|---|---:|---:|---|---:|---:|---:|
| P1_dense_all \| max-other-reference vs own-reference | 9,768 | 0.2415 | [0.236, 0.246] | 80.9% | 12.3% | 0.910 |
| P1_dense_all \| max-reference vs own-reference | 13,346 | 0.1867 | [0.183, 0.191] | 59.2% | 1.5% | 0.784 |
| P1_dense_all \| random-other-reference vs own-reference | 9,768 | -0.0393 | [-0.043, -0.035] | 42.3% | 51.1% | -0.195 |
| P1_dense_all \| max-other vs own \| informative own answer only | 9,029 | 0.2453 | [0.240, 0.250] | 82.0% | 11.1% | 0.997 |
| P1_sample \| hybrid vs dense \| own-reference | 2,998 | 0.0581 | [0.048, 0.068] | 45.1% | 31.3% | 0.210 |
| P1_sample \| hybrid vs dense \| max-other-reference | 2,149 | 0.0207 | [0.009, 0.032] | 41.6% | 41.0% | 0.073 |
| P1_sample \| hybrid vs dense \| max-reference | 2,910 | 0.0294 | [0.019, 0.039] | 41.7% | 37.7% | 0.107 |
| P1_sample \| hybrid vs dense \| max-other \| informative own answer only | 1,947 | 0.0244 | [0.012, 0.037] | 42.1% | 40.7% | 0.086 |
| P2_sample \| hybrid vs dense \| own-reference | 2,998 | 0.0532 | [0.044, 0.062] | 44.6% | 30.8% | 0.213 |
| P2_sample \| hybrid vs dense \| max-other-reference | 2,149 | 0.0182 | [0.007, 0.029] | 40.8% | 40.8% | 0.067 |

A CI that excludes zero indicates a difference on these queries under this protocol; it is not a claim about agricultural quality. Comparisons marked `informative own answer only` drop rows whose logged answer is a call-log note, so both sides compare answer against answer.

## 6. Subsets: reference availability and novelty

Dense retrieval on all test rows, by pool size:

| Subset | n | Own F1 | Max-other F1 | Random-other F1 |
|---|---:|---:|---:|---:|
| Exactly one reference | 3,693 | 0.2954 | 0.2184 | 0.2184 |
| Two or more references | 9,669 | 0.3337 | 0.5788 | 0.2948 |
| Twenty or more references | 5,721 | 0.3263 | 0.6641 | 0.2793 |

The single-reference row can show a higher own-reference F1 than the informative-reference maximum. That is expected: 1,208 test rows (8.7%) have a call-log note as their own logged answer. The original metric still scores the retrieved answer against that note (mean 0.2380), which is not an answer, while the reference maximum uses only informative references (mean 0.4051). Comparisons restricted to informative own answers are reported in §5.

By novelty bucket (dense similarity to the grouped index, Batch A thresholds):

| Bucket | n | Own F1 | Max F1 | Max-other F1 |
|---|---:|---:|---:|---:|
| Near-duplicate (≥0.90) | 9,673 | 0.3323 | 0.5588 | 0.5856 |
| Similar (0.80–0.90) | 3,308 | 0.3136 | 0.4146 | 0.5220 |
| Novel (<0.80) | 850 | 0.2330 | 0.2970 | 0.5603 |

## 7. Screening variant (P2: informative-only candidates)

The deployed retriever screens call-log notes out of its candidates. This variant applies the same screen to both systems on the same 3,000 queries.

| System | n | Own F1 | Max-other F1 |
|---|---:|---:|---:|
| Dense, screened | 3,000 | 0.3155 | 0.6007 |
| Hybrid, screened | 3,000 | 0.3687 | 0.6196 |

## 8. Interpretation

- On rows with an alternative reference, dense own-reference F1 is 0.3337 and max-other-reference F1 is 0.5788: the same retrieved answer agrees substantially more with the range of advisor answers than with the single logged one.
- The random-reference control scores 0.2948, so part of any gap is mechanical (more chances to match). The max-other − own delta must be read against that control, not on its own.
- Maximum semantic similarity to any reference is 0.7755; it is a maximum over the pool, so it rises with pool size, and it says the retrieved wording is close to some historical answer, not that the advice is right.
- Dense vs hybrid differences on identical queries are in the paired table; they are conditional on one grouped split, not a general ranking of the methods.

## 9. Limitations

- References are historical advisor answers, not verified-correct advice; agreement with them is not agricultural correctness.
- Near-duplicate grouping is transitive at cosine >= 0.95, so some large groups may merge related but not identical questions.
- Contradictory answers can coexist in one pool (Improvement 1 shows such conflicts exist); max-reference F1 rewards matching any of them.
- Semantic similarity measures wording closeness only; it is not correctness and is reported separately from F1.
- No human relevance or expert agronomic judgement exists for these pools (Batch B Phase 2 annotation remains PENDING).
- Max-reference F1 and max semantic similarity both rise mechanically with pool size, because both are maxima over more candidates; the random-reference control is reported for that reason.
- On rows whose own logged answer is a call-log note, the original single-reference metric scores against that note; those rows are reported separately and excluded from the informative-own paired comparison.
- Only one grouped split seed was used, because the index rebuild is expensive; results are conditional on that split.

Environment: Python 3.12.13, macOS-26.6.2-arm64-arm-64bit; numpy 2.5.4, scikit-learn 1.9.1, faiss 1.15.1, sentence-transformers 6.2.0.

## 10. Reproduce

```bash
OMP_NUM_THREADS=1 .venv312/bin/python experiments/multi_reference/run_evaluation.py
OMP_NUM_THREADS=1 .venv312/bin/python experiments/multi_reference/report.py
OMP_NUM_THREADS=1 .venv312/bin/python -m pytest tests/unit/test_multi_reference.py -q
```

The evaluation refuses to report new metrics if the Batch A reproduction gate fails.
