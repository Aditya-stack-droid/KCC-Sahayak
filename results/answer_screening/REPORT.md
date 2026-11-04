# Answer-screening evaluation: shipped rule vs negation/measurement-aware screener

**Evidence class:** authored synthetic cases, deterministic, no model training and no human
or expert review. The shipped production gate is **unchanged**; this measures a candidate
rule (`answer_screening.screening`) against the frozen rule (`confidence_gate.answer_risk`).

**Cases:** 93 total — 34 unsafe, 48 safe, 11 ambiguous (excluded from P/R/F1).

## 1. Headline comparison

| Metric | v1 (shipped) | v2 (candidate) | Change |
|---|---:|---:|---:|
| True positives | 25 | 34 | +9 |
| False positives | 26 | 2 | -24 |
| False negatives | 9 | 0 | -9 |
| True negatives | 22 | 46 | +24 |
| Precision | 0.4902 | 0.9444 | |
| Recall | 0.7353 | 1.0000 | |
| F1 | 0.5882 | 0.9714 | |
| F1 95% CI | [0.45, 0.7033] | [0.9231, 1.0] | |
| **Unsafe-answer acceptance rate** | 26.47% | 0.00% | -0.2647 |
| **Negation false-positive rate** | 63.64% | 0.00% | -0.6364 |

Confusion matrices — v1: TP=25 FP=26 FN=9 TN=22; v2: TP=34 FP=2 FN=0 TN=46.

## 2. Before / after by language

| Language | n | v1 recall | v2 recall | v1 FP | v2 FP |
|---|---:|---:|---:|---:|---:|
| devanagari | 7 | 0.0000 | 1.0000 | 0 | 0 |
| en | 59 | 0.8000 | 1.0000 | 21 | 2 |
| hinglish | 8 | 0.7500 | 1.0000 | 1 | 0 |
| roman_hindi | 8 | 1.0000 | 1.0000 | 4 | 0 |

## 3. Negation situations (Priority 1)

Of 33 safe negation/warning cases, v1 flagged 21 and v2 flagged 0.

| Situation | n | v1 flagged | v2 flagged |
|---|---:|---:|---:|
| ambiguous | 0 | FP=0 | FP=0 |
| area_measure | 3 | FP=0 | FP=0 |
| container_measure | 3 | FP=0 | FP=0 |
| dangerous_combination | 3 | FP=2 | FP=0 |
| direct_negation | 5 | FP=4 | FP=0 |
| hindi_negation | 3 | FP=0 | FP=0 |
| hinglish_negation | 3 | FP=1 | FP=0 |
| historical_description | 2 | FP=1 | FP=1 |
| household_measure | 6 | FP=0 | FP=0 |
| informal_measure | 3 | FP=0 | FP=0 |
| metric_standard | 5 | FP=0 | FP=0 |
| mixed_clauses | 5 | FP=1 | FP=0 |
| multi_clause | 2 | FP=2 | FP=0 |
| negated_dosage | 4 | FP=4 | FP=0 |
| negation_after_chemical | 3 | FP=3 | FP=0 |
| negation_before_chemical | 3 | FP=3 | FP=0 |
| packaging_measure | 5 | FP=0 | FP=0 |
| punctuation | 3 | FP=2 | FP=0 |
| question | 4 | FP=0 | FP=0 |
| roman_hindi_negation | 3 | FP=2 | FP=0 |
| safe_control | 10 | FP=1 | FP=1 |
| spelling_variation | 3 | FP=0 | FP=0 |
| tank_measure | 1 | FP=0 | FP=0 |

## 4. Measurement classes (Priority 3)

v2 escalates household/container/packaging measures it cannot interpret; v1 does not
recognise most of them. Recall by situation is in `metrics.json`.

## 5. Residual errors (not hidden)

- v2 false negatives: none
- v2 false positives: hist-02, safe-01
- New false negatives vs v1: none
- New false positives vs v1: none

## 6. Limitations

- All cases are synthetic and authored by the screener's author; not a sample of real queries.
- Gold labels are authored expectations about escalation, not expert-adjudicated agronomic harm.
- The author had read both rule sets, so the 'challenge' cases are not an unbiased field estimate.
- v2 adds a wider vocabulary (brands, Devanagari), so part of the recall change is vocabulary, not only negation/measurement logic.
- Detecting a pattern is not agronomic correctness; a miss does not prove an answer is safe.
- The shipped production gate is unchanged; v2 is an evaluated candidate, not yet integrated.
