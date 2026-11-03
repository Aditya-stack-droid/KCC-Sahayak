# PII detection and redaction (Batch B, Phase 1)

Standalone, additive privacy module for KCC-Sahayak. It does not import
`src/`, the backend/frontend, the models or the original retrieval index, and
it does not modify any original file or Batch A output.

**Status:** experimental. The evaluation uses a small synthetic dataset written
for this phase; it is not evidence of production readiness or real-world PII
recall.

## What is supported

| Entity | Recogniser | Notes |
|---|---|---|
| `PHONE` | Indian mobile regex + context rules; international regex | Indian numbers may have `+91`/`0091`, spaces or hyphens in 5+5 / 3+3+4 groups. International numbers require an explicit `+`/`00`. |
| `EMAIL` | RFC-ish ASCII regex | Supports dots, `+` and uppercase. |
| `PERSON` | Anchor/honorific heuristics | Only names after `my name is`, `name:`, `naam`, `Mr./Mrs./Dr.`, `श्री/श्रीमती`, or `Rahul ji`. Conservative by design. |
| `AADHAAR` | 12-digit shape + Verhoeff checksum | A valid checksum is strong evidence; an explicit `aadhaar/aadhar/UID/आधार` cue is accepted even if the checksum or first digit is invalid. |
| `IFSC` | `^[A-Z]{4}0[A-Z0-9]{6}$` | No context needed; pattern is distinctive. |
| `BANK_ACCOUNT` | 9–18 digits with an account/bank cue | The cue is required to avoid redacting arbitrary long numbers. |
| `URL` | `http(s)://` and `www.` | Trailing punctuation is trimmed. |

Devanagari digits (`०१२३४५६७८९`) are transliterated 1:1 before matching, so
spans still refer to the original string and redaction preserves it.

## What is deliberately not detected

- **Unanchored names** such as `Rahul said ...` or `रमेश ने कहा ...`.
- **Initials** (`R. K. Sharma`) and **lowercase honorifics** (`mr. verma`).
- **Lowercase surnames** in Roman-Hindi (`Rahul sharma` captures only `Rahul`).
- **Dot-grouped** (`98765.43210`) and **trunk-zero** (`09876543210`) phones.
- Names/addresses in scripts or formats not covered by the patterns.
- Any claim that redaction is complete.

These are measured as false negatives in the synthetic evaluation, not hidden.

## Two phone policies

- `balanced` (default): accepts a context-free 10-digit number that starts 6–9
  if there is no local negative cue. It is more useful for free text but can
  redact a random 10-digit identifier.
- `strict`: requires a phone cue, separator grouping, `+91`/`00` prefix or an
  international prefix. It avoids standalone ambiguous numbers but misses bare
  mobile numbers.

Negative context (plot, token, order, invoice, sample ID, account, currency,
units, year) suppresses a phone unless a **strong** phone cue such as `call`,
`mobile`, `WhatsApp` or `संपर्क` is present. Negative checks are local: an
Aadhaar or email later in the same sentence does not suppress an earlier phone.

## API

```python
from experiments.privacy.recognizers import detect
from experiments.privacy.redact import RedactionConfig, redact

detections = detect("Call 9876543210 or email rahul@example.com")
# (Detection(type='PHONE', ...), Detection(type='EMAIL', ...))

result = redact(
    "My name is Rahul, call 9876543210.",
    RedactionConfig(placeholders={"PHONE": "<phone>"}),
)
print(result.redacted)   # My name is [PERSON], call <phone>.
print(result.counts())   # {'PERSON': 1, 'PHONE': 1}
```

`detect(..., enabled_types=("EMAIL",))` limits detection. `redact` is
idempotent: applying it twice produces the same string because placeholders
contain no detectable PII.

Matching is strict exact span + type in evaluation; partial name matches are
counted as both a false positive and a false negative.

## Synthetic evaluation dataset

`data/synthetic_pii_eval.csv` contains **87 hand-authored synthetic examples**:

- 47 English, 21 Hindi (Devanagari), 19 Hinglish/Roman-Hindi.
- 70 gold entities: 23 phone, 22 person, 10 Aadhaar, 8 email, 4 bank account,
  2 IFSC, 1 URL.
- 31 negative/ambiguous examples: plots, tokens, orders, prices, quantities,
  PIN codes, invalid Aadhaar-like numbers, organisation/crop names, unanchored
  names.

All rows have `synthetic=true`. Emails use `example.com`; identifiers and phone
numbers are fake. The builder computes exact spans from declared values and
refuses to write on missing/overlapping annotations.

Rebuild:

```bash
.venv312/bin/python experiments/privacy/build_eval_set.py
```

## Results (balanced policy)

Measured by `run_evaluation.py` on the synthetic set:

| Measure | Value |
|---|---:|
| Precision | 0.9848 |
| Recall | 0.9286 |
| F1 | 0.9559 |
| F1 95% bootstrap CI | [0.9077, 0.9882] |
| Exact document matches | 0.9425 |
| Document-level false-positive rate | 0.0000 |

Per entity F1: phone 0.9545, person 0.9048, email/Aadhaar/IFSC/bank/URL 1.0000.
Per language F1: English 0.9394, Hindi 1.0000, Hinglish 0.9474.

The perfect-looking categories are a consequence of the synthetic set and the
same-author design; they are **not** independent validation. Five errors remain:
initials, lowercase honorific, dotted phone, trunk-zero phone, and a lowercase
surname partial match. Full tables and redacted examples:
`results/privacy_audit/REPORT.md`.

## Commands

```bash
# rebuild the synthetic dataset
.venv312/bin/python experiments/privacy/build_eval_set.py

# evaluate and write results/privacy_audit/{metrics.json,REPORT.md,errors_balanced.csv}
.venv312/bin/python experiments/privacy/run_evaluation.py --overwrite

# focused tests
OMP_NUM_THREADS=1 .venv312/bin/python -m pytest tests/unit/test_privacy.py -q

# full existing + Batch A + Batch B suite
OMP_NUM_THREADS=1 .venv312/bin/python -m pytest tests -q

# protect originals and Batch A
.venv312/bin/python experiments/batch_b/verify_baseline.py --check
```

## Reproduction details

- Python 3.12.13, CPU only, no new packages.
- Bootstrap: 2,000 resamples over documents, seed 0.
- No model training, no network access, no Presidio.
- `run_evaluation.py` writes only under `results/privacy_audit/`.
- `metrics.json` contains counts, spans and redacted text; it does not contain
  detected raw values.
- The evaluation dataset is synthetic; it is not a substitute for an
  adjudicated real or de-identified corpus audit.

## Limitations

- The dataset and recognisers share an author; coverage is overfitted to the
  authored patterns. The labels are author-provided and **not independently
  adjudicated**; human adjudication of PII errors is PENDING.
- Names are anchor-based, not NER. Recall on real text will be lower.
- Phone format coverage is incomplete; see the hard cases.
- Hindi/Hinglish handling is regex-based and does not understand morphology.
- Aadhaar validation is checksum-based, not identity verification. Aadhaar-like
  test numbers with an explicit cue are redacted even when invalid.
- No real corpus scan was performed in this phase, so no prevalence claim is
  made.
- Redaction is a mitigation, not a guarantee that no personal information
  remains. A second review or a trained NER model would be needed before
  production use.
- The original safety gate, backend and API are unchanged. A `pii_in_answer`
  check is a later integration step, not part of this phase.

## Files

| File | Purpose |
|---|---|
| `recognizers.py` | Entity recognisers, Verhoeff checksum, span resolution |
| `redact.py` | Configurable placeholder redaction |
| `build_eval_set.py` | Deterministic synthetic dataset builder |
| `audit.py` | Strict entity-level metrics and bootstrap CI |
| `run_evaluation.py` | Writes the Phase 1 metrics/report |
| `data/synthetic_pii_eval.csv` | Synthetic evaluation set |
| `tests/unit/test_privacy.py` | 36 unit/behavioural tests |
| `results/privacy_audit/` | Metrics, error rows and report |
