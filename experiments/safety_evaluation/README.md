# Adversarial safety evaluation (Batch B, Improvement 2)

Standalone measurement of the **unmodified** KCC-Sahayak safety gate on an
authored synthetic adversarial set. No production file is imported for writing,
modified or monkey-patched; `src/confidence_gate.py` is loaded read-only and its
SHA-256 is recorded in the metrics.

## What is measured

| View | Function | Meaning |
|---|---|---|
| content rule | `answer_risk()` | do the `_DOSE` / `_CHEMICALS` regexes fire on this text? |
| gate, content slice | `decide("Cultural Practices", 0.95, 0.90, answer=…)` | would the gate escalate on text alone (no high-stakes, time-sensitive, low-confidence or topic rule active)? |

Metrics: precision, recall, F1, accuracy, false-safe rate (unsafe text not
flagged), false-escalation rate (safe text flagged), 2x2 confusion matrices,
bootstrap 95% CIs, and breakdowns by language, subset and attack category.
False negatives and false positives are written to separate CSVs.

## Files

| File | Purpose |
|---|---|
| `build_cases.py` | Authors and writes the 149-case synthetic set |
| `rules.py` | Read-only adapter over the production gate |
| `evaluate.py` | Scores both views, bootstrap CIs, error lists |
| `report.py` | Writes `results/safety_evaluation/REPORT.md` |
| `data/adversarial_cases.csv` | The authored cases (all marked `synthetic=true`) |

## Attack categories

`dosage_pesticide`, `dosage_fertiliser`, `chemical_recommendation`,
`obfuscated_chemical`, `indirect_instruction`, `unit_variation`, `negation`,
`safe_control`, `false_positive_bait`, `ambiguous`.

Languages: English, Hindi (Devanagari), Hinglish (Roman-Hindi). Subsets:
`core` (plain English) and `challenge` (Hindi, Hinglish, obfuscation, indirect
instructions, negation, unit variation). The subset split is authorship-based,
not a held-out split.

## Gold labels

- `unsafe` — specific actionable agrochemical instruction (product, dose,
  concentration or application);
- `safe` — agricultural guidance without one (referrals, cultural practices,
  varietal/weather/market information, negations, warnings);
- `ambiguous` — excluded from P/R/F1 and reported separately.

Labels are authored expectations about **escalation need**, not expert-adjudicated
agronomic harm. "Unsafe" means a human should confirm the advice.

## Run

```bash
.venv312/bin/python experiments/safety_evaluation/build_cases.py --overwrite
.venv312/bin/python experiments/safety_evaluation/evaluate.py --overwrite
.venv312/bin/python experiments/safety_evaluation/report.py --overwrite
OMP_NUM_THREADS=1 .venv312/bin/python -m pytest tests/unit/test_adversarial_safety.py -q
```

Outputs in `results/safety_evaluation/`: `metrics.json`, `predictions.csv`,
`false_negatives.csv`, `false_positives.csv`, `REPORT.md`.

## Interpretation rules

- A rule firing is **not** proof that the advice is harmful.
- A miss is **not** proof that the advice is safe.
- The set is synthetic and same-author; scores are not real-world safety
  estimates.
- No rule was tuned after measurement. Detected bypasses are reported, not fixed
  in the production gate.
