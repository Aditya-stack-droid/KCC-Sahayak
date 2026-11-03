# Answer conflict detection (Batch B, Improvement 1)

**Status:** implemented and measured on an authored synthetic pair set plus a
descriptive corpus scan. **No expert adjudication exists**, and a rule firing is
not proof that an answer is wrong.

The module never merges, averages or resolves advice. When any conflict is
found, the optional wrapper turns an auto-answer into **ESCALATE** for a human
advisor; the production gate is imported read-only and remains unchanged.

## 1. What is detected

| Kind | Severity | Rule |
|---|---|---|
| `product` | high | both answers name products and the product sets are disjoint |
| `polarity` | high | a product is negated in one answer and recommended in the other (clause-local negation) |
| `dosage` | high / medium | same product dimension and basis, but even the closest pair of amounts differs by ≥ 50% |
| `basis` | medium | doses are expressed on different non-convertible bases (acre vs bigha; plant vs acre) |
| `dimension` | medium | the same product is dosed in incompatible units (kg vs litre) |
| `interval` | high / medium | application intervals differ by ≥ 50% |

Unit handling: g/kg/quintal/tonne and ml/litre are normalised; hectare is
converted to acre (1 ha = 2.4710538 acre); bigha/kanal/guntha are **not**
converted because their size varies by district. `per N litre` and `per N acre`
are normalised to per-one-unit. Prices, plot/token ids and pack sizes are
excluded from quantity extraction.

Product handling: spelling and abbreviation variants only (`D.A.P.`, `2,4-D`,
`MOP`/`potash`). Brands are **not** mapped to active ingredients, so `rogor` and
`dimethoate` count as different products. This is a documented limitation, not a
claim of chemical equivalence.

## 2. Synthetic pair evaluation

- Pairs: **77** (52 conflict,
  25 no-conflict), hand-authored and **synthetic**.
- Matching: strict pair-level label; per-kind metrics treat each kind as a binary
  label over all pairs. Dose tolerance: 0.5.
- Bootstrap: 2000 resamples, seed
  0.

| TP | FP | FN | TN | Precision | Recall | F1 | F1 95% CI | Accuracy |
|---:|---:|---:|---:|---:|---:|---:|---|---:|
| 47 | 0 | 5 | 25 | 1.0000 |
0.9038 | 0.9495 | [0.8989, 0.9901] |
0.9351 |

Per-kind:

| Kind | Support | TP | FP | FN | Precision | Recall | F1 |
|---|---|---|---|---|---|---|---|
| dosage | 21 | 19 | 0 | 2 | 1.0000 | 0.9048 | 0.9500 |
| product | 13 | 12 | 0 | 1 | 1.0000 | 0.9231 | 0.9600 |
| basis | 6 | 5 | 0 | 1 | 1.0000 | 0.8333 | 0.9091 |
| dimension | 4 | 4 | 0 | 0 | 1.0000 | 1.0000 | 1.0000 |
| interval | 3 | 3 | 0 | 0 | 1.0000 | 1.0000 | 1.0000 |
| polarity | 5 | 4 | 0 | 1 | 1.0000 | 0.8000 | 0.8889 |

By authored subset (the challenge cases were written after the core rules
existed, to expose likely failure modes; they are still authored, not a held-out
benchmark):

| Subset | Pairs | Precision | Recall | F1 | Accuracy |
|---|---|---|---|---|---|
| challenge | 10 | 1.0000 | 0.4444 | 0.6154 | 0.5000 |
| core | 67 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

Small support (especially dimension/basis) means wide uncertainty; these
per-kind values are not stable estimates.

## 3. Errors found

- `h01` gold=['product'] predicted=[]
- `h02` gold=['dosage'] predicted=[]
- `h03` gold=['dosage'] predicted=[]
- `h04` gold=['polarity'] predicted=[]
- `h05` gold=['basis'] predicted=[]

## 4. Descriptive corpus scan

Reuses Batch A's grouped metadata and informative mask. It reports how often the
rules fire, not how often advice is actually wrong.

- Eligible multi-answer groups: **8,619**

- Sampled groups: **1,000** (seed 0)

- Answer pairs compared: **4,918**

- Pairs where a rule fired: **1,258** (25.58%)

- Groups with at least one flagged pair: **382** (38.20%)

Kinds (pairs with at least one such conflict): `basis` 105, `dimension` 488, `dosage` 406, `interval` 12, `polarity` 2, `product` 246



| Informative answers in group | Groups | Flagged groups | Fire rate |
|---|---|---|---|
| 2 | 404 | 78 | 0.1931 |
| 3-5 | 596 | 304 | 0.5101 |



The redacted review queue (`group_review.csv`) contains 200 rows. It is a candidate list for a human reviewer, not a set of confirmed errors.

## 5. Escalation behaviour

`decide_with_conflicts` runs the production gate on the top-ranked answer and
adds the conflict rule:

- no conflict: the production decision is returned unchanged;
- conflict: the decision becomes `ESCALATE`, with machine-readable reasons
  (`ANSWER_CONFLICT`, `DOSAGE_CONFLICT`, …), even for a category the production
  gate would auto-answer;
- an existing production ESCALATE keeps its original code as `base_code`.

The wrapper returns no suggested or merged answer. Focused tests cover these
three cases.

## 6. Limitations

- The synthetic pair set is small, authored by the same person who wrote the
  rules, and not expert-verified. Precision/recall are not real-world estimates.
- Corpus flags are rule-fire rates. Group members may discuss different crops or
  stages, which the current rules do not model as context.
- Brand/active-ingredient equivalence is not inferred; `rogor` vs `dimethoate`
  will be reported as a product difference.
- Negation is clause-local and pattern-based; conditional advice (“if aphids
  persist, then spray…”) is only partially handled.
- Dosages within 50% are treated as compatible. This tolerance is a deliberate
  trade-off to limit false positives and is not agronomically validated.
- Only the first five answers are compared, and the corpus scan covers the
  grouped train+val side only.
- No integration into the backend or production gate.

## 7. Reproduction

```bash
.venv312/bin/python experiments/conflict_detection/build_pairs.py
.venv312/bin/python experiments/conflict_detection/evaluate.py --overwrite
.venv312/bin/python experiments/conflict_detection/run_corpus.py --overwrite
.venv312/bin/python experiments/conflict_detection/report.py --overwrite
OMP_NUM_THREADS=1 .venv312/bin/python -m pytest tests/unit/test_conflict_detection.py -q
```
