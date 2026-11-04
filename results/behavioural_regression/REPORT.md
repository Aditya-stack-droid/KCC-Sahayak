# Behavioural regression report (Improvement 5)

**Evidence class:** authored synthetic probes, deterministic execution, no model
training and no human labels or expert review. This is a behavioural audit,
not a real-world accuracy or safety estimate.

**Cases:** 168
**Passed:** 109
**Failed:** 59
**Pass rate:** 64.88%
**Unhandled crashes:** 0
**Case-set SHA-256:** `ed71cef9b848724adc70a1e644903eba9ea2d680decbc27365284b5573cb0ea0`

Expected values encode the behaviour the study treats as correct for robustness,
safety or interface handling. A failed case records a capability gap; it is not
an exception in the harness. Production and previous experiment outputs are not
modified by this run.

## Coverage

| Category | Scope | Cases | Passed | Failed | Failure types |
|---|---|---:|---:|---:|---|
| A | Typo and spelling robustness | 24 | 12 | 12 | degraded 1, false_negative 11 |
| B | Hindi / Roman-Hindi / Hinglish | 24 | 13 | 11 | degraded 5, false_negative 5, missing_capability 1 |
| C | Negation and prohibitions | 18 | 8 | 10 | false_positive 10 |
| D | Numbers, units, dosage and ambiguity | 23 | 16 | 7 | false_negative 7 |
| E | Ambiguous or underspecified queries | 9 | 9 | 0 | — |
| F | Multi-intent queries | 18 | 1 | 17 | missing_capability 17 |
| G | Empty, malformed and oversized input | 27 | 26 | 1 | validation_gap 1 |
| H | Cross-component interactions | 25 | 24 | 1 | false_negative 1 |

## Failure types

| Type | Cases |
|---|---:|
| degraded | 6 |
| false_negative | 24 |
| false_positive | 10 |
| missing_capability | 18 |
| validation_gap | 1 |

## Failures by category

All failing cases are listed. Full machine-readable records are in `failures.json`;
every case, including passing controls, is in `cases.csv`.

### A — Typo and spelling robustness

| ID | Probe | Input | Expected | Observed | Type |
|---|---|---|---|---|---|
| A03 | entities | whea rust control | {"crop": ["wheat"], "symptom": ["rust"]} | {"crop": [], "symptom": ["rust"]} | false_negative |
| A04 | entities | wheatt rustt | {"crop": ["wheat"], "symptom": ["rust"]} | {"crop": [], "symptom": []} | false_negative |
| A05 | entities | wheatrust control | {"crop": ["wheat"], "symptom": ["rust"]} | {"crop": [], "symptom": []} | false_negative |
| A06 | entities | mustrad aphid | {"crop": ["mustard"], "symptom": ["aphid"]} | {"crop": [], "symptom": ["aphid"]} | false_negative |
| A07 | entities | cottton bollworm | {"crop": ["cotton"], "symptom": ["bollworm"]} | {"crop": [], "symptom": ["bollworm"]} | false_negative |
| A08 | entities | tomatoe blight | {"crop": ["tomato"], "symptom": ["blight"]} | {"crop": [], "symptom": ["blight"]} | false_negative |
| A09 | entities | paddyy stem borer | {"crop": ["paddy"], "symptom": ["stem borer"]} | {"crop": [], "symptom": ["stem borer"]} | false_negative |
| A10 | entities | whiteflyy on cotton | {"crop": ["cotton"], "symptom": ["whitefly"]} | {"crop": ["cotton"], "symptom": []} | false_negative |
| A12 | intent | wheatt rustt | Plant Protection | Other | degraded |
| A18 | risk | use monocrotophus | true | false | false_negative |
| A20 | risk | dose 2 gram per ltr | true | false | false_negative |
| A21 | entities | aphidd infestation in mustard | {"crop": ["mustard"], "symptom": ["aphid"]} | {"crop": ["mustard"], "symptom": ["infestation"]} | false_negative |

### B — Hindi / Roman-Hindi / Hinglish

| ID | Probe | Input | Expected | Observed | Type |
|---|---|---|---|---|---|
| B03 | register | gehu me ratua | hinglish | en | degraded |
| B04 | register | gehun me peela rust | hinglish | en | degraded |
| B05 | register | gayhoon me peela ratua | hinglish | en | degraded |
| B09 | entities | गेहूं में पीला रतुआ | {"crop": ["wheat"], "symptom": ["rust"]} | {"crop": [], "symptom": []} | false_negative |
| B10 | entities | gehu me rust | {"crop": ["wheat"], "symptom": ["rust"]} | {"crop": [], "symptom": ["rust"]} | false_negative |
| B11 | intent | गेहूं में यूरिया डालें | Nutrient Management | Other | degraded |
| B13 | intent | gehu ka bhav kya hai | Market Price | Other | degraded |
| B17 | risk | प्रति लीटर पानी में 2 मिली रोगोर मिलाकर छिड़काव करें | true | false | false_negative |
| B19 | risk | dawa 2 mili prati litre dalen | true | false | false_negative |
| B22 | conflict | 50 किलो यूरिया प्रति एकड़ डालें। \|\| 50 किलो यूरिया प्रति बीघा डालें। | ["basis"] | [] | false_negative |
| B23 | evidence | गेहूं में पीला रतुआ | true | [] | missing_capability |

### C — Negation and prohibitions

| ID | Probe | Input | Expected | Observed | Type |
|---|---|---|---|---|---|
| C01 | risk | Do not spray chlorpyrifos. | false | true | false_positive |
| C02 | risk | Never apply urea at this stage. | false | true | false_positive |
| C03 | risk | Spraying should be avoided when honey bees are active. | false | true | false_positive |
| C04 | risk | Do not apply 50 kg urea per acre. | false | true | false_positive |
| C05 | risk | No need to spray anything. | false | true | false_positive |
| C06 | risk | Avoid using monocrotophos; it is banned. | false | true | false_positive |
| C08 | risk | spray mat karein | false | true | false_positive |
| C10 | risk | do not apply 2 ml monocrotophos per litre | false | true | false_positive |
| C11 | risk | 2 ml per litre spray mat karein | false | true | false_positive |
| C13 | risk | the label warns not to spray when windy | false | true | false_positive |

### D — Numbers, units, dosage and ambiguity

| ID | Probe | Input | Expected | Observed | Type |
|---|---|---|---|---|---|
| D07 | risk | Apply two bottles | true | false | false_negative |
| D08 | risk | Apply 1 bottle dawa 1 tank | true | false | false_negative |
| D12 | risk | Apply 50 dawa | true | false | false_negative |
| D17 | conflict | Apply 2 ml per gallon water. \|\| Apply 5 ml per 10 litre water. | ["basis"] | [] | false_negative |
| D18 | conflict | Apply 1 sachet per 20 litre water. \|\| Apply 3 sachet per 20 litre water. | ["dosage"] | [] | false_negative |
| D19 | conflict | Apply 1 bori per acre. \|\| Apply 2 bori per acre. | ["dosage"] | [] | false_negative |
| D20 | conflict | Apply 2 chammach per 10 litre water. \|\| Apply 4 chammach per 10 litre water. | ["dosage"] | [] | false_negative |

### F — Multi-intent queries

| ID | Probe | Input | Expected | Observed | Type |
|---|---|---|---|---|---|
| F01 | multi_intent | My tomato leaves are turning yellow and also write a Python program for me. | multi_intent_signal | single_label_only | missing_capability |
| F02 | multi_intent | Which fertiliser is best for paddy and what is the capital of Japan? | multi_intent_signal | single_label_only | missing_capability |
| F03 | multi_intent | My buffalo is not eating and please book a train ticket to Delhi. | multi_intent_signal | single_label_only | missing_capability |
| F04 | multi_intent | How do I control aphids in mustard and who won yesterday's cricket match? | multi_intent_signal | single_label_only | missing_capability |
| F05 | multi_intent | What is the urea dose for wheat and explain blockchain technology. | multi_intent_signal | single_label_only | missing_capability |
| F06 | multi_intent | My mango tree has white spots and what is a mutual fund? | multi_intent_signal | single_label_only | missing_capability |
| F07 | multi_intent | Tell me the market price of cotton and write a poem about rain. | multi_intent_signal | single_label_only | missing_capability |
| F08 | multi_intent | How do I sow bajra and how do I open a savings account? | multi_intent_signal | single_label_only | missing_capability |
| F09 | multi_intent | My cow has fever and which movie won the award? | multi_intent_signal | single_label_only | missing_capability |
| F10 | multi_intent | What pesticide for bollworm and how do I apply for a passport? | multi_intent_signal | single_label_only | missing_capability |
| F11 | multi_intent | Best seed variety for gram and what is the interest rate on fixed deposit? | multi_intent_signal | single_label_only | missing_capability |
| F12 | multi_intent | My brinjal crop has wilt and give me a recipe for brinjal curry. | multi_intent_signal | single_label_only | missing_capability |
| F13 | multi_intent | How do I irrigate sugarcane and who is the prime minister? | multi_intent_signal | single_label_only | missing_capability |
| F14 | multi_intent | Which fungicide for blight and how do I fix my laptop battery? | multi_intent_signal | single_label_only | missing_capability |
| F15 | multi_intent | My goat is limping and what is the schedule of the next train? | multi_intent_signal | single_label_only | missing_capability |
| F16 | multi_intent | Fertilizer for tomato plants and what is the boiling point of water? | multi_intent_signal | single_label_only | missing_capability |
| F17 | multi_intent | wheat rust control and urea dose for wheat | multi_intent_signal | single_label_only | missing_capability |

### G — Empty, malformed and oversized input

| ID | Probe | Input | Expected | Observed | Type |
|---|---|---|---|---|---|
| G26 | query_request | {"text": " "} | invalid | valid | validation_gap |

### H — Cross-component interactions

| ID | Probe | Input | Expected | Observed | Type |
|---|---|---|---|---|---|
| H25 | privacy_types | Call ०९८७६५४३२१० for details | ["PHONE"] | [] | false_negative |

## Passing controls

109 cases passed. They cover clean English baselines, empty-input
safety, API length/type validation, clause-local negation, unit conversion,
conflict escalation, privacy redaction, gate routing and out-of-scope decisions.
See `cases.csv` for the complete per-case record.

## Protocol

1. Cases are authored in `br_cases.py`; each is marked synthetic and has one
   expected value fixed before the evaluator was run.
2. `br_probes.py` calls existing production or Batch B components read-only.
3. The evaluator writes `metrics.json`, `cases.csv`, `failures.json` and this
   report under `results/behavioural_regression/` only.
4. No threshold, rule, lexicon or model was tuned on these cases.

## What this establishes

- A reproducible behavioural baseline for the components named in `cases.csv`.
- Separately counted false positives and false negatives, plus missing
  capabilities and validation gaps.
- Explicit evidence that multi-intent detection is absent: mixed queries still
  receive one weak-label category and no multi-intent signal.
- Explicit evidence of language, typo, unit, negation and PII edge cases.

## What this does not establish

- Real-world frequency of any failure. The cases are authored probes.
- End-to-end retrieval or answer correctness: no MiniLM/FAISS model is loaded.
- ASR accuracy; only transcript normalisation is exercised.
- Human or agronomist agreement, severity, or production readiness.
- That a passing case is agronomically correct; it only matches the expected
  component behaviour.

## Reproduce

```bash
OMP_NUM_THREADS=1 .venv312/bin/python experiments/behavioural_regression/run_evaluation.py
OMP_NUM_THREADS=1 .venv312/bin/python -m pytest tests/unit/test_behavioural_regression.py -q
```
