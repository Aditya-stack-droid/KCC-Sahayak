# Answer conflict detection (Batch B, Improvement 1)

Deterministic, explainable detection of contradictory advice across retrieved
answers. It is additive: the production gate, models, datasets and Batch A/Phase
1/Phase 2 outputs are untouched.

The module **never merges, averages or resolves** advice. When a conflict is
found, the optional wrapper escalates to a human advisor.

## Files

| File | Purpose |
|---|---|
| `units.py` | quantity, unit, basis and interval extraction |
| `products.py` | product-name recognition and clause-local negation |
| `conflicts.py` | pairwise rules, severity, explanations, `detect_many` |
| `gate.py` | `decide_with_conflicts`: production gate + conflict escalation |
| `build_pairs.py` | authored synthetic pair set |
| `evaluate.py` | pair-level metrics + bootstrap CIs |
| `run_corpus.py` | descriptive scan of multi-answer question groups |
| `report.py` | combined report writer |
| `data/conflict_pairs.csv` | synthetic pairs (all rows marked synthetic) |

Outputs live in `results/conflict_detection/`.

## Rules

| Kind | Severity | Rule |
|---|---|---|
| `product` | high | both answers name products and the sets are disjoint |
| `polarity` | high | a product negated in one answer is recommended in the other |
| `dosage` | high/medium | same dimension + basis, closest amounts differ by ≥ 50% |
| `basis` | medium | non-convertible bases (acre vs bigha, plant vs acre) |
| `dimension` | medium | same product dosed in incompatible units (kg vs litre) |
| `interval` | high/medium | application intervals differ by ≥ 50% |

Unit normalisation: g/kg/quintal/tonne → grams; ml/litre → ml; hectare → acre
(1 ha = 2.4710538 acre). Bigha/kanal/guntha are **not** converted because their
size varies; a difference is reported as a `basis` conflict. `per N litre` and
`per N acre` are normalised to per-one-unit. Currency, plot/token ids and pack
sizes are excluded.

Products: spelling/abbreviation variants only (`D.A.P.`, `2,4-D`, `MOP`/`potash`).
Brand-to-active-ingredient mapping is deliberately **not** performed, so
`rogor` vs `dimethoate` is a product difference. That limitation is reported.

Negation is clause-local, so “do not apply urea; instead spray neem oil” negates
urea only.

## Usage

```python
from experiments.conflict_detection.conflicts import detect_many
report = detect_many([answer_1, answer_2, answer_3])
report.escalate          # True when any conflict fired
report.reasons           # ("ANSWER_CONFLICT", "DOSAGE_CONFLICT", ...)
report.as_dict()         # no merged or consensus answer is produced
```

Optional gate integration (production code unchanged):

```python
from experiments.conflict_detection.gate import decide_with_conflicts
decision = decide_with_conflicts(
    answers=[a1, a2, a3],
    query_type="Varietal Information",
    intent_confidence=0.95,
    retrieval_similarity=0.90,
)
decision.action          # ESCALATE when a conflict exists
decision.base_action     # the untouched production decision
```

## Reproduce

```bash
.venv312/bin/python experiments/conflict_detection/build_pairs.py
.venv312/bin/python experiments/conflict_detection/evaluate.py --overwrite
.venv312/bin/python experiments/conflict_detection/run_corpus.py --overwrite
.venv312/bin/python experiments/conflict_detection/report.py --overwrite
OMP_NUM_THREADS=1 .venv312/bin/python -m pytest tests/unit/test_conflict_detection.py -q
```

## Data provenance

`data/conflict_pairs.csv` is a small **synthetic** set hand-authored for this
evaluation. It is not real advisor data, not a sample of real disagreement and
not expert-verified. The corpus scan reuses Batch A metadata and the
informative-answer mask; it reports rule-fire rates, not error rates.

## Limitations

- Synthetic, same-author evaluation; precision/recall are not real-world estimates.
- The 50% dosage tolerance is a trade-off, not an agronomic standard.
- Group members may discuss different crops/stages; the rules do not model context.
- Conditional advice and complex negation are only partially handled.
- No brand/active-ingredient equivalence; no unit conversion for local area units.
- At most five answers are compared; the corpus scan covers train+val only.
- No backend or production-gate integration.
