# Answer screening — negation and measurement awareness (candidate rule)

**Status: evaluated candidate, NOT integrated into the production gate.** The shipped
rule (`src/confidence_gate.py::answer_risk`) is unchanged; this package provides a
second-generation content screener and the evidence for or against adopting it.

## What it fixes

Two measured weaknesses of the shipped regex, both reported by earlier audits:

1. **Negation false positives.** `"Do not spray chlorpyrifos."` is flagged because the
   regex sees the product and the verb but ignores the prohibition. The Improvement 5
   behavioural audit recorded 10 of these (`C01, C02, C03, C04, C05, C06, C08, C10, C11, C13`).
2. **Measurement false negatives.** Dosages written with containers or household measures
   (`"one capful per litre"`, `"two spoonfuls per tank"`, `"1 sachet per 20 litre"`,
   `"2 bori per acre"`, `"2 chammach"`) are not matched by the shipped numeric+metric regex.

## Design (conservative, because this is safety code)

- **Default is RISKY.** A span is suppressed only when a negation cue clearly governs it
  inside the same clause.
- **Negation is clause-local.** Clauses split on sentence ends, contrastive *and result*
  conjunctions (`but`, `instead`, `so`, `therefore`, `isliye`), commas, and on `and`/`or`
  **only when a new application verb follows** — so `"do not spray A and apply B"` still
  screens B.
- **No conversion is ever invented.** `Measurement.converted` is always `None`; household,
  container, packaging, tank and area measures are escalated, never converted to ml/g.
- **Ambiguous text stays risky** (questions, hedged statements) and escalates.

## Files

| Path | Purpose |
|---|---|
| `screening.py` | the candidate screener (`screen_answer`, `risk_reason`, `extract_measurements`) |
| `cases.py` | 93 authored cases across the 14 negation situations and the measurement taxonomy |
| `data/screening_cases.csv` | generated case set (`build`/`--overwrite`) |
| `evaluate.py` | scores `answer_risk` (v1) and the candidate (v2) on the **same** cases |
| `report.py` | renders `REPORT.md` from measured metrics |

## Results (authored synthetic cases — not a field estimate)

Same 93 cases (82 scored: 34 unsafe, 48 safe, 11 ambiguous excluded), identical rows:

| Metric | v1 shipped | v2 candidate |
|---|---:|---:|
| True positives | 25 | 34 |
| False positives | 26 | 2 |
| False negatives | 9 | 0 |
| Precision | 0.4902 | **0.9444** |
| Recall | 0.7353 | **1.0000** |
| F1 | 0.5882 | **0.9714** |
| Negation false-positive rate | 0.6364 | **0.0000** |
| Unsafe-answer acceptance rate | 0.2647 | **0.0000** |
| New false negatives vs v1 | — | **0** |
| New false positives vs v1 | — | **0** |

Corroborating check on the **existing** adversarial set (`experiments/safety_evaluation`,
149 cases): v1 `TP=63 FP=23 FN=18`; v2 `TP=80 FP=11 FN=1`. The single remaining v2 miss is
`obf-01` (letter-spaced product name `m o n o c r o t o p h o s`).

**Residual errors (not hidden):** v2 still flags two factual statements that v1 also flags —
`hist-02` (`"He had applied urea before the rain."`) and `safe-01` (`"Contact the district
agriculture officer for the current spray schedule."`). These are recommendation-vs-mention
problems, not negation/measurement problems, and are left for a separate change.

## Limitations

- Cases are synthetic and written by the screener's author; they are not real traffic.
- v2 also widens vocabulary (brands, Devanagari), so part of the recall change is
  vocabulary, not only the negation/measurement logic.
- Detecting a pattern is not agronomic correctness.

## Reproduce

```bash
OMP_NUM_THREADS=1 .venv312/bin/python experiments/answer_screening/cases.py
OMP_NUM_THREADS=1 .venv312/bin/python experiments/answer_screening/evaluate.py --overwrite
OMP_NUM_THREADS=1 .venv312/bin/python -m pytest tests/unit/test_answer_screening.py -q
```
