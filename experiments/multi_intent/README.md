# Multi-intent detection (rule-based, standalone, escalate-only)

**Status: standalone experiment, NOT integrated.** The shipped pipeline predicts one
intent per query. This module detects when a query carries two or more distinct requests,
splits it into sub-queries, labels each, and flags when any sub-query is high-stakes or
time-sensitive. It only ever **adds** an escalation signal and is not wired into the gate.

## Approach

Comparison of the options the brief lists (multi-label classifier / query decomposition /
rule-assisted extraction): a trained multi-label model needs multi-intent supervision that
does not exist in the corpus, and an LLM is explicitly out of scope. A **rule-assisted
decomposer** is the simplest approach justified by the available evidence. It reuses
`src/weak_labels.py` and `src/glossary.py` plus a small bilingual cue lexicon, and adds no
dependency and no model.

`decompose(text)` splits on sentence ends, coordinating/contrastive conjunctions and
commas, keeps request-like segments, labels each, and returns the distinct intent set, the
sub-queries (with character offsets) and `safety_escalation_required`.

## Results (authored synthetic cases — not a field estimate)

110 cases (74 multi-intent, 36 single-request controls; 81 en, 11 roman-Hindi, 10
Devanagari, 8 Hinglish). Baseline = the shipped single-label keyword rule.

| Metric | single-label baseline | rule-based decomposition |
|---|---:|---:|
| Exact intent-set match | 0.2182 | **0.7818** |
| Micro precision | 0.6410 | **0.9097** |
| Micro recall | 0.5100 | **0.9005** |
| Micro F1 | 0.5686 | **0.9051** |
| Macro F1 | 0.6289 | **0.9130** |
| Missed-intent rate | 0.5503 | **0.1164** |
| Extra-intent rate | 0.2273 | **0.0722** |
| Decomposition accuracy | 0.0000 | **0.9054** |
| Safety-escalation recall | 0.7375 | **0.9500** |

Per language (decomposition micro-F1): Devanagari 0.97, Hinglish 1.00, roman-Hindi 0.92,
English 0.89.

**Residual failures (recorded, not hidden):** 21 cases with a missed intent, 13 with an
extra intent, and 4 high-risk queries not flagged (`mi-036, mi-067, mi-087, mi-109`).
Main causes: context-inferred intent ("the price of urea" ties between Market Price and
Nutrient Management), and the reused English rules being plural-unaware.

## Reproduce

```bash
OMP_NUM_THREADS=1 .venv312/bin/python experiments/multi_intent/cases.py
OMP_NUM_THREADS=1 .venv312/bin/python experiments/multi_intent/evaluate.py --overwrite
OMP_NUM_THREADS=1 .venv312/bin/python -m pytest tests/unit/test_multi_intent.py -q
```

---

# v2 — risk-first multi-intent detection (safety study)

**Status: standalone, escalate-only, NOT integrated.** `v2` is a new module
(`detect.py` + `risk.py`); `decompose.py` above is kept unchanged as the v1 baseline for
comparison. The safety problem it fixes: v1 derived safety only from the classified
intent, so a classification or decomposition miss silently removed the risk signal.

## Design (the required flow)

`normalise → identify multi-intent → decompose → preserve entities → assess risk per
sub-query → overall decision`

* Risk is assessed **per sub-query and again on the whole query**, so a bad decomposition
  cannot hide a dangerous request.
* Any high/medium-risk sub-query escalates the whole query; a safe sub-query can never
  override a dangerous one.
* Unknown-intent agricultural queries escalate (medium) — unknown intents are never
  auto-approved.
* Entities not present are reported as `"unknown"`, never invented.

Risk signals (any of the first four ⇒ high): high-stakes/time-sensitive intent; a
chemical/dosage/measurement instruction (reuses the answer screener); a pest/disease term
with a treatment request; a toxicity/exposure/harvest-restriction question.

## Results — old vs new on the **same** cases

220 authored cases = the original 110 (`cases.py`) + 110 new adversarial
(`adversarial_cases.py`). `gold_safety` = any gold intent in the high-stakes/time-sensitive
set.

| Metric (combined 220) | baseline (single) | v1 (decompose) | v2 (risk-first) |
|---|---:|---:|---:|
| Exact intent-set match | 0.186 | **0.746** | 0.673 |
| Micro F1 | 0.516 | **0.882** | 0.856 |
| Macro F1 | 0.587 | **0.888** | 0.856 |
| Missed-intent rate | 0.593 | **0.137** | 0.163 |
| Safety-escalation precision | 1.000 | 1.000 | 0.962 |
| Safety-escalation recall | 0.650 | 0.910 | **1.000** |
| **High-risk false negatives** | 62 | 16 | **0** |

On the new adversarial set alone: v2 high-risk FN **0** (v1 12, baseline 41), safety recall
1.000. v2 trades ~2.6 points of micro-F1 for removing all high-risk misses — the intended
safety-first trade. v1's four original failures (`mi-036, mi-067, mi-087, mi-109`) now
escalate; the residual v1 safety gaps are **not** rewritten in the v1 numbers.

## Reproduce (v2)

```bash
OMP_NUM_THREADS=1 .venv312/bin/python experiments/multi_intent/adversarial_cases.py
OMP_NUM_THREADS=1 .venv312/bin/python experiments/multi_intent/evaluate_v2.py --overwrite
OMP_NUM_THREADS=1 .venv312/bin/python -m pytest tests/unit/test_multi_intent_v2.py -q
```
