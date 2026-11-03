# PII detection and redaction — synthetic evaluation (Batch B, Phase 1)

All examples in this report are **synthetic**. They were authored for this evaluation and contain fake names, numbers and `example.com` addresses. No real farmer data, annotation or expert review was used. These numbers do **not** establish production readiness or real-world PII recall.

## Protocol

- Dataset: `experiments/privacy/data/synthetic_pii_eval.csv` (87 examples; {'en': 47, 'hi': 21, 'hinglish': 19}; 31 negative/ambiguous).
- Matching: strict exact character span **and** entity type. A partial name match is counted as one false positive and one false negative.
- Two policies: `balanced` accepts a context-free 10-digit mobile-like number; `strict` requires a phone cue, separator, +91/00 prefix or international prefix.
- Bootstrap: 2,000 document resamples, seed 0, percentile 95% CI for micro F1.
- Random seed: 0. No model training and no new dependency.

## Overall

| Policy | TP | FP | FN | Precision | Recall | F1 | F1 95% CI | Exact docs | Doc-level FPR |
|---|---|---|---|---|---|---|---|---|---|
| balanced | 65 | 1 | 5 | 0.9848 | 0.9286 | 0.9559 | [0.9077, 0.9882] | 0.9425 | 0.0000 |
| strict | 63 | 1 | 7 | 0.9844 | 0.9000 | 0.9403 | [0.8868, 0.9793] | 0.9195 | 0.0000 |

Balanced predicted **66** entities against **70** gold entities. The synthetic set has 31 documents with no PII; 0 of them produced a false positive.

## Entity-wise performance (balanced policy)

| Entity | Gold | Predicted | TP | FP | FN | Precision | Recall | F1 |
|---|---|---|---|---|---|---|---|---|
| PHONE | 23 | 21 | 21 | 0 | 2 | 1.0000 | 0.9130 | 0.9545 |
| EMAIL | 8 | 8 | 8 | 0 | 0 | 1.0000 | 1.0000 | 1.0000 |
| PERSON | 22 | 20 | 19 | 1 | 3 | 0.9500 | 0.8636 | 0.9048 |
| AADHAAR | 10 | 10 | 10 | 0 | 0 | 1.0000 | 1.0000 | 1.0000 |
| IFSC | 2 | 2 | 2 | 0 | 0 | 1.0000 | 1.0000 | 1.0000 |
| BANK_ACCOUNT | 4 | 4 | 4 | 0 | 0 | 1.0000 | 1.0000 | 1.0000 |
| URL | 1 | 1 | 1 | 0 | 0 | 1.0000 | 1.0000 | 1.0000 |

## Language-wise performance (balanced policy)

| Language | Documents | Negative docs | Precision | Recall | F1 | Doc-level FPR |
|---|---|---|---|---|---|---|
| en | 47 | 19 | 1.0000 | 0.8857 | 0.9394 | 0.0000 |
| hi | 21 | 7 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| hinglish | 19 | 5 | 0.9474 | 0.9474 | 0.9474 | 0.0000 |

Language labels are author-assigned (`en`, `hinglish`, `hi`) and describe the synthetic sentence, not a validated language identification result.

## False positives

| ID | Language | Gold | Predicted | FP spans | Redacted text |
|---|---|---|---|---|---|
| hin-name-lowercase-surname | hinglish | PERSON | PERSON | PERSON@10-15 | mera naam [PERSON] hai |

No negative/ambiguous document produced any detection under the balanced policy. This is a synthetic-set result; it does not mean the detector will have zero false positives on real text.

## False negatives and known failures

| ID | Language | Gold | Predicted | FN spans | Redacted text | Note |
|---|---|---|---|---|---|---|
| en-name-initials | en | PERSON | none | PERSON@11-23 | My name is [PERSON]. | hard: initials are not supported by the name heuristics |
| en-honorific-lowercase | en | PERSON | none | PERSON@4-9 | mr. [PERSON] advised a neem spray. | hard: honorific matching is case-sensitive |
| en-phone-dots | en | PHONE | none | PHONE@5-16 | Call [PHONE] now. | hard: dot-grouped mobile is not supported |
| en-phone-trunk-zero | en | PHONE | none | PHONE@5-16 | Call [PHONE] now. | hard: Indian trunk prefix 0 is not supported |
| hin-name-lowercase-surname | hinglish | PERSON | PERSON | PERSON@10-22 | mera naam [PERSON] hai | hard: lowercase surname is only partially captured |

The hard cases are intentional. They show that the name heuristics do not handle initials, lowercase honorifics or lowercase surnames, and that dotted and trunk-zero phone formats are not recognised.

## What this establishes

- On **this synthetic set**, the balanced recogniser finds most declared entities (F1 0.9559, 95% CI [0.9077, 0.9882]).
- The negative/ambiguous examples did not trigger phone redaction, including plot/token/order/price/quantity contexts.
- Aadhaar-like values are only accepted with a valid Verhoeff checksum or an explicit Aadhaar cue, and a bank cue takes precedence for account numbers.

## What this does not establish

- It does **not** measure real-world PII prevalence, recall or precision. The dataset is small, synthetic and authored by the same person who wrote the recognisers. Perfectly covered categories reflect that shared authorship, not independent validation.
- **Names are best-effort and anchor-based.** Unanchored names (e.g. `Rahul said ...`) are deliberately not detected; this is a recall gap.
- **Phone format coverage is incomplete.** Dot-grouped numbers, trunk prefix `0`, and some other real formats are missed.
- Hindi/Hinglish handling is regex- and digit-normalisation-based. It does not use a trained NER model, gazetteer or Presidio. Devanagari digit normalisation is 1:1 and preserves offsets.
- No real corpus was scanned in this phase, so no claim is made about how much PII exists in the historical KCC-Sahayak corpus.
- Aadhaar detection is identifier-shaped, not an identity verification system. An invalid test number with an explicit cue is still redacted.
- Recall cannot be guaranteed. Redaction is a mitigation, not a guarantee that no personal information remains.

## Reproduction

```bash
.venv312/bin/python experiments/privacy/build_eval_set.py
.venv312/bin/python experiments/privacy/run_evaluation.py --overwrite
.venv312/bin/python -m pytest tests/unit/test_privacy.py -q
.venv312/bin/python experiments/batch_b/verify_baseline.py --check
```

Metrics: `results/privacy_audit/metrics.json`. Error rows: `results/privacy_audit/errors_balanced.csv` (redacted text only).

Generated: 2026-11-03T11:21:38+00:00
