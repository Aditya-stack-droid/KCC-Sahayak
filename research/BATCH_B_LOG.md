# Batch B experiment log

**Rules:** additive only; every number comes from an executable run; original
files and Batch A outputs are never modified; human/expert work is marked
PENDING; failed or partial runs are recorded.

---

## 2026-11-03 — Phase 0: baseline verification and protection

**Status:** COMPLETE.
**Purpose:** freeze the original project and Batch A as hash-verified
baselines before any Batch B experiment, and provide the shared scaffolding.

### Environment

- macOS Apple Silicon (M3), 8 GB RAM, CPU only.
- Python 3.12.13 (`.venv312`).
- pytest 9.2.0; scikit-learn 1.9.1; scipy 1.18.1; pandas 3.1.0;
  numpy 2.5.4; sentence-transformers 6.2.0; FAISS 1.15.1.
- No new dependency installed. Presidio remains optional and unused.

### Commands

```bash
OMP_NUM_THREADS=1 .venv312/bin/python experiments/batch_b/verify_baseline.py --init
OMP_NUM_THREADS=1 .venv312/bin/python -m pytest tests -q
OMP_NUM_THREADS=1 .venv312/bin/python experiments/batch_b/verify_baseline.py --check
```

### Results

| Check | Result |
|---|---|
| Baseline manifest | **270 files** (220 protected originals, **50 Batch A** files) |
| Test suite | **52 passed** in 5.88 s (41 original + 11 Batch A) |
| Hash verification | **PASS**: 0 changed, 0 missing, 0 unexpected additions |
| Tracked modifications | The same 4 pre-existing files (`results/calibration.json`, `results/hybrid_retrieval.json`, `results/model_selection.json`, `scripts/13_hybrid_retrieval.py`), untouched in Phase 0 |

### Files created in Phase 0

| File | Purpose |
|---|---|
| `experiments/batch_b/common.py` | Shared paths, seeds, SHA-256 helper, environment report |
| `experiments/batch_b/verify_baseline.py` | Manifest generator/verifier (`--init`, `--check`) |
| `experiments/batch_b/manifests/baseline_hashes.json` | SHA-256 + size for 270 baseline files |
| `experiments/batch_b/README.md` | Batch B rules, phase table, verification usage |
| `requirements-batch-b.txt` | Pinned environment actually used; no new packages |
| `research/BATCH_B_PLAN.md` | Approved plan, phase order, file manifest, constraints |
| `research/BATCH_B_LOG.md` | This log |

### What was verified

- Batch A results (`results/improved_evaluation/`), Batch A code/artefacts
  (`experiments/grouped_evaluation/`), and the Batch A research notes and test
  are recorded and unchanged.
- Original data, models, `src/`, `backend/`, `scripts/`, existing tests,
  documents and result files are recorded and unchanged.
- `--check` fails on any changed/missing file or unexpected addition in an
  immutable directory, so later phases can be blocked mechanically.

### What was not done

- No PII, annotation, safety, conflict, OOS/multi-intent, retrieval,
  behavioural or documentation experiment was run.
- No human labels, kappa, expert review or improvement claim is made.
- Batch C was not started.

### Next

Phase 1 (PII detection and redaction) is **not started**; it waits for
explicit approval.

---

## 2026-11-03 — Phase 1: PII detection and redaction

**Status:** COMPLETE for the approved synthetic scope.
**Purpose:** build a standalone, multilingual PII detector and redactor with a
synthetic evaluation, without touching the original project, Batch A, the
backend or the frontend.

### Scope and design

- Entities: `PHONE` (Indian mobile and explicit international), `EMAIL`,
  `PERSON` (anchor/honorific only), `AADHAAR` (Verhoeff or explicit cue),
  `IFSC`, `BANK_ACCOUNT` (bank cue required), `URL`.
- Languages: English, Hindi (Devanagari), Hinglish/Roman-Hindi.
- Devanagari digits are transliterated 1:1 before matching so spans/redaction
  still refer to the original text.
- Phone context rules: strong cue / prefix beats negative context; plot, token,
  order, invoice, sample ID, account, currency, units and years suppress a
  phone unless a strong phone cue is present. Negative checks are local.
- Aadhaar: valid Verhoeff checksum or explicit `aadhaar/aadhar/UID/आधार` cue;
  bank cue takes precedence for 12-digit account numbers; `batch`/`sample`
  context suppresses a checksum-only match.
- Overlaps resolved by entity priority and span length.
- Redaction: configurable type filter and placeholders, right-to-left
  replacement, idempotent. Report/error text masks **both detected and gold
  spans**, so a missed entity is not leaked into a report.
- No Presidio, no new dependency, no network access, no backend change.

### Files created

| File | Purpose |
|---|---|
| `experiments/privacy/__init__.py` | Package marker |
| `experiments/privacy/recognizers.py` | Recognisers, Verhoeff, span resolution |
| `experiments/privacy/redact.py` | Configurable placeholder redaction and report masking |
| `experiments/privacy/build_eval_set.py` | Deterministic synthetic dataset builder |
| `experiments/privacy/audit.py` | Strict entity metrics, bootstrap CI, errors |
| `experiments/privacy/run_evaluation.py` | Writes Phase 1 metrics/report |
| `experiments/privacy/README.md` | Implementation and reproduction documentation |
| `experiments/privacy/data/synthetic_pii_eval.csv` | 87 synthetic examples |
| `tests/unit/test_privacy.py` | 36 unit/behavioural tests |
| `results/privacy_audit/metrics.json` | Raw metrics for both policies |
| `results/privacy_audit/dataset_summary.json` | Dataset composition |
| `results/privacy_audit/errors_balanced.csv` | Error rows (redacted text only) |
| `results/privacy_audit/REPORT.md` | Evaluation report |
| `results/privacy_audit/test_results.txt` | Full-suite test output |

Only Batch B documents were modified: `experiments/batch_b/README.md` (phase
status) and this log. No original or Batch A file was modified.

### Dataset

- **87** hand-authored synthetic examples: 47 English, 21 Hindi, 19 Hinglish.
- **70** gold entities: 23 phone, 22 person, 10 Aadhaar, 8 email, 4 bank
  account, 2 IFSC, 1 URL.
- **31** negative/ambiguous examples (plots, tokens, orders, prices, quantities,
  PIN codes, invalid Aadhaar-like numbers, organisation/crop names, unanchored
  names).
- All rows are marked `synthetic=true`; emails use `example.com`; no real farmer
  data was used.
- Hard cases were deliberately included: initials, lowercase honorific,
  dot-grouped phone, trunk-zero phone, lowercase Roman-Hindi surname.

### Results (measured, not fabricated)

Strict exact span + type matching; bootstrap 2,000 document resamples, seed 0.

| Policy | TP | FP | FN | Precision | Recall | F1 | F1 95% CI | Exact docs | Doc-level FPR |
|---|---:|---:|---:|---:|---:|---:|---|---:|---:|
| balanced | 65 | 1 | 5 | 0.9848 | 0.9286 | 0.9559 | [0.9077, 0.9882] | 0.9425 | 0.0000 |
| strict | 63 | 1 | 7 | 0.9844 | 0.9000 | 0.9403 | [0.8868, 0.9793] | 0.9195 | 0.0000 |

Per entity (balanced) F1: phone 0.9545, person 0.9048, email/Aadhaar/IFSC/bank/
URL 1.0000. Per language (balanced) F1: English 0.9394, Hindi 1.0000, Hinglish
0.9474.

The perfect-looking categories are an artefact of a small synthetic set written
by the same author as the recognisers. They are **not** independent validation.

### Errors observed (balanced)

| ID | Failure |
|---|---|
| `en-name-initials` | `R. K. Sharma` not captured |
| `en-honorific-lowercase` | `mr. verma` not captured (case-sensitive honorific) |
| `en-phone-dots` | `98765.43210` not captured |
| `en-phone-trunk-zero` | `09876543210` not captured |
| `hin-name-lowercase-surname` | `Rahul sharma` partially captured as `Rahul` (1 FP + 1 FN) |

### Verification

| Check | Result |
|---|---|
| Pre-Phase 1 hash check | PASS (270 files, 0 changed/missing/additions) |
| Focused privacy tests | **36 passed** |
| Full suite | **88 passed** in 7.04 s (52 baseline + 36 new) |
| Post-Phase 1 hash check | **PASS** (270 files, 0 changed, 0 missing, 0 unexpected additions) |

### Limitations / not done

- No real or de-identified corpus scan, so no PII prevalence claim is made.
- Dataset is synthetic and same-author; labels are not independently
  adjudicated. Human adjudication of PII errors is **PENDING**; recall on real
  text will be lower.
- Names are anchor-based heuristics, not NER; unanchored names and initials are
  missed.
- Phone format coverage is incomplete (dots, trunk zero, others).
- Hindi/Hinglish handling is regex-based, not morphological.
- Aadhaar checksum is not identity verification; invalid test numbers with an
  explicit cue are redacted.
- Redaction is a mitigation, not a guarantee. Production use would need a
  trained NER model and independent adjudicated evaluation.
- No `pii_in_answer` integration into the original safety gate, backend or API.
- Batch C was not started.

### Next

Phase 2 (human annotation infrastructure) began on 2026-11-03; see the entry
below. The Phase 1 outputs are frozen and were not modified.

---

## 2026-11-03 — Phase 2: human annotation infrastructure

**Status:** COMPLETE for the infrastructure. **Human annotation is PENDING:** no
person has labelled any item, so no human-human kappa, no expert verification
and no adjudicated ground truth exist.
**Purpose:** build a rigorous, blinded, offline workflow for human labels of
intent, retrieval relevance, answer quality, safety, out-of-scope, multi-intent
and missing context, without modifying the original project.

### Sample (measured)

- Source pool: `data/annotation/annotated.csv` (3,496 real corpus rows).
- Dropped 373 rows for question length outside 12–300 characters or an answer
  shorter than 25 characters; dropped 1 row the shipped `src/lang.py` heuristic
  classifies as non-English. Eligible English pool: **3,122**.
- Selected **300** queries with seed 0; every stratum met its quota (0
  shortfall). Dedupe: unique normalised text and cosine < 0.98 to any other
  selected query.
- Strata (exactly one per query): common 80, rare_intent 55, near_duplicate 40,
  ambiguous 35, insufficient_context 35, novel 30, potentially_unsafe 25.
- Existing LLM label mix (used for stratification only, withheld from
  annotators): PP 73, NM 54, AH 41, GS 33, CP 26, VI 23, FI 21, MP 14, WE 14,
  OT 1. The single `Other` item is disclosed in the report as a class too small
  for stable per-class agreement.
- Each item has three candidates: its logged answer + one same-topic and one
  different-topic distractor (900 candidates total). Provenance lives only in
  `key.jsonl`; the annotator files contain opaque `c1`–`c3` ids.
- Freshness: novelty/group features reuse the read-only Batch A artefacts
  (`corpus_embeddings.npy`, `emb_annotated_texts.npy`, `corpus_split.csv`); no
  model was retrained and no text was re-encoded.
- Frozen sample hash (`items_sha1`): `7c5ff7d1f081b5cce9e9d40361c718c07e51ff45`.
- Development note: the sample was generated twice before any annotation
  existed. The first selection was replaced because label-round-robin was added
  to give rare intents a fair share (Market Price had only 2 items). No label
  existed at the time, so no result was invalidated; the frozen sample is the
  round-robin version above.

### Design

- **Schema** (`schema.py`): 10 QueryType codes + `OOS`; structured tie between
  `OOS` and `out_of_scope`; multi-intent must contain the primary intent;
  `none` cannot be combined with other missing-context values; unknown candidate
  ids are rejected. `OT` (agriculture, outside the nine topics) and `OOS` (not
  agriculture) are kept distinct.
- **Blinding** (`ui.py`): one self-contained HTML file per annotator with only
  that annotator's blind items; no key, strata, LLM labels, provenance, network
  calls or external assets. JSON/CSV export, JSON resume, keyboard navigation.
- **Store** (`store.py`): append-only per-annotator shards; latest record wins
  and revisions are counted; imports are all-or-nothing and write a reject
  report rather than truncating a sheet; merged export adds adjudicated values
  only where a named human recorded `status=resolved`.
- **Agreement** (`agreement.py`): Cohen's kappa per pair on complete cases with
  missing counts, Fleiss' kappa for three or more annotators, observed/expected
  agreement, confusion matrices, per-class one-vs-rest kappa with support,
  set agreement (exact match, mean Jaccard) for multi-intent and
  missing-context, and a separate human-vs-LLM comparison. Kappa is `null`
  rather than 0 when expected agreement is 1.
- **QC** (`quality.py`): coverage, blank tasks, revisions, invalid labels,
  structural contradictions, long identical label runs between annotators, and
  adjudication status. Nothing resolves a disagreement automatically.
- **Report** (`report.py`): explicit provenance table (LLM / human / expert /
  adjudicated) and a PENDING result when fewer than two annotators exist.

### Files created

| File | Purpose |
|---|---|
| `experiments/annotation/__init__.py` | Package marker |
| `experiments/annotation/schema.py` | Label sets, normalisation, validation |
| `experiments/annotation/sampling.py` | Stratified 300-query English sample |
| `experiments/annotation/store.py` | Shards, import/export, adjudication log |
| `experiments/annotation/agreement.py` | Cohen/Fleiss kappa, confusion, disagreements |
| `experiments/annotation/quality.py` | Coverage, duplicate, consistency, adjudication QC |
| `experiments/annotation/ui.py` | Offline single-file annotation interface |
| `experiments/annotation/report.py` | Report/QC writer with PENDING handling |
| `experiments/annotation/cli.py` | `build-sample`, `make-ui`, `import`, `status`, `validate`, `export`, `agreement`, `adjudicate`, `report` |
| `experiments/annotation/README.md` | Workflow, labels, layout, limitations |
| `experiments/annotation/data/items.jsonl` | 300 blind items |
| `experiments/annotation/data/key.jsonl` | Strata, LLM labels, source rows, candidate provenance |
| `experiments/annotation/data/sample_summary.json` | Frozen selection statistics |
| `experiments/annotation/data/sample_features.json` | Cached novelty/group features (derived, reproducible) |
| `experiments/annotation/data/ui/A001.html`, `A002.html` | Per-annotator offline interfaces |
| `tests/unit/test_annotation.py` | 30 focused tests |
| `results/annotation/REPORT.md` | Phase 2 report (PENDING) |
| `results/annotation/QC_REPORT.md` | QC report (PENDING) |
| `results/annotation/agreement.json`, `qc.json`, `sample_overview.json` | Raw measured state |
| `results/annotation/tables/strata.csv`, `existing_llm_labels.csv` | Sample tables |

Only Batch B documents were modified: `experiments/batch_b/README.md` (phase
status) and this log. No original file, Batch A output or Phase 1 output was
modified.

### Verification

| Check | Result |
|---|---|
| Pre-implementation hash check | PASS (270 files, 0 changed/missing/additions) |
| Focused annotation tests | **30 passed** |
| Full suite | **118 passed** in 7.29 s (52 baseline + 36 Phase 1 + 30 Phase 2) |
| Post-implementation hash check | **PASS** (270 files, 0 changed, 0 missing, 0 unexpected additions) |
| Agreement artefact | `status: PENDING`, all task pair tables empty |
| QC artefact | `status: PENDING`, 300 items awaiting annotation |

### Failures fixed during the phase

| Failure | Fix |
|---|---|
| `KeyError: query_type` when building items | `annotated.csv` uses `weak_label`; read the present column |
| `KeyError: crops` | metadata fields are absent from the annotation pool; read defensively |
| Schema treated the valid label `none` as an empty value | empty-value filtering now excludes only `nan`/`null` |
| First sample had 2 Market Price items | label round-robin inside `common` and `rare_intent`; sample rebuilt before any annotation existed |

### Limitations / not done

- **No human labels, kappa, expert review or adjudication exist.** Imports and
  agreement are implemented and tested, not populated with invented data.
- English only; Hindi/Hinglish support exists in the schema and UI but no Hindi
  items were drawn.
- Label `Other` appears once; its per-class agreement will be unstable.
- Candidate relevance is not live retrieval evaluation; Phase 6 evaluates the
  retriever itself.
- No backend, API, frontend or safety-gate integration; the workflow is
  deliberately standalone and offline.
- The interface contains real corpus text and has no authentication; it is a
  local research tool, not a deployed service.
- Agreement measures consistency, not agronomical correctness.

### Next

Phase 3 (independent safety evaluation) is **not started**; it waits for
explicit approval.

---

## 2026-11-03 — Batch B Improvement 1: answer conflict detection

**Status:** COMPLETE for the approved offline scope. No expert adjudication
exists, so the measured precision/recall are properties of an authored synthetic
set, not real-world conflict-detection accuracy.
**Purpose:** flag contradictory advice across the top retrieved answers
(numbers, units, bases, products, intervals, negation) and escalate to a human
advisor. Nothing is merged, averaged or resolved.

### Protection baseline (created before implementation)

`experiments/batch_b/verify_phase_outputs.py --snapshot` recorded
`manifests/batch_b_protection_hashes.json`: **308 files** — 220 original, 50
Batch A, 14 Phase 1, 24 Phase 2 — with SHA-256, size and group. Checking it
immediately after creation reported PASS (0 changed, 0 missing, 0 additions).
The pre-existing uncommitted modifications (`results/calibration.json`,
`results/hybrid_retrieval.json`, `results/model_selection.json`,
`scripts/13_hybrid_retrieval.py`) were included as-is and not touched.

### Implementation

- `units.py`: quantities with units (g/kg/quintal/tonne, ml/litre), per-acre and
  per-litre normalisation to per-one-unit, hectare→acre (1 ha = 2.4710538 acre),
  non-convertible local bases (bigha/kanal/guntha), application intervals,
  exclusions for prices, plot ids and pack sizes.
- `products.py`: product vocabulary with spelling/abbreviation variants and
  clause-local negation; no brand→active-ingredient mapping.
- `conflicts.py`: pairwise rules (`product`, `polarity`, `dosage`, `basis`,
  `dimension`, `interval`), severity, explanations and `detect_many`.
- `gate.py`: `decide_with_conflicts` runs the untouched production gate, then
  escalates on conflict and records `ANSWER_CONFLICT` / `*_CONFLICT` reasons and
  `base_action`/`base_code`.
- `build_pairs.py`, `evaluate.py`, `run_corpus.py`, `report.py` reproduce the
  data, metrics, corpus scan and report.

### Files created

| File | Purpose |
|---|---|
| `experiments/conflict_detection/__init__.py` | Package marker |
| `experiments/conflict_detection/units.py` | Quantity/unit/basis/interval extraction |
| `experiments/conflict_detection/products.py` | Product recognition and negation |
| `experiments/conflict_detection/conflicts.py` | Pairwise conflict rules and `detect_many` |
| `experiments/conflict_detection/gate.py` | Optional gate wrapper (production gate read-only) |
| `experiments/conflict_detection/build_pairs.py` | Authored synthetic pair set |
| `experiments/conflict_detection/evaluate.py` | Pair metrics + bootstrap CI |
| `experiments/conflict_detection/run_corpus.py` | Descriptive grouped-corpus scan |
| `experiments/conflict_detection/report.py` | Combined report writer |
| `experiments/conflict_detection/README.md` | Design, usage, limitations |
| `experiments/conflict_detection/data/conflict_pairs.csv` | 77 synthetic pairs (52 conflict / 25 no-conflict) |
| `tests/unit/test_conflict_detection.py` | 30 focused tests |
| `results/conflict_detection/REPORT.md` | Measured report |
| `results/conflict_detection/metrics.json`, `pair_predictions.csv` | Raw metrics and per-pair predictions |
| `results/conflict_detection/corpus_scan.json`, `group_review.csv` | Corpus scan and redacted review queue (200 rows) |
| `experiments/batch_b/verify_phase_outputs.py` | Protection snapshot/check tool |
| `experiments/batch_b/manifests/batch_b_protection_hashes.json` | Pre-Improvement-1 protection baseline |
| `experiments/batch_b/manifests/batch_b_improvement1_hashes.json` | Post-Improvement-1 inventory |

Modified (Batch B documents only): `experiments/batch_b/README.md`,
`research/BATCH_B_LOG.md`. No original, Batch A, Phase 1 or Phase 2 file was
modified.

### Results (measured)

Authored synthetic pairs, strict pair-level label, dose tolerance 0.5, bootstrap
2,000 resamples (seed 0).

| Metric | Value |
|---|---:|
| Pairs | 77 (52 conflict, 25 no-conflict; 67 core, 10 challenge) |
| TP / FP / FN / TN | 47 / 0 / 5 / 25 |
| Precision / Recall / F1 | 1.0000 / 0.9038 / 0.9495 |
| F1 95% CI | [0.8989, 0.9901] |
| Accuracy | 0.9351 |
| Core subset F1 | 1.0000 (n=67) |
| Challenge subset F1 | 0.6154 (n=10, recall 0.4444) |

Descriptive corpus scan (reuses Batch A grouped metadata + informative mask,
seed 0): 8,619 eligible multi-answer groups, 1,000 sampled, 4,918 answer pairs,
1,258 pairs (25.58%) and 382 groups (38.20%) flagged; 200-row redacted review
queue written. These are rule-fire rates, not confirmed error rates.

### Failures fixed during the improvement

The first authored set (67 core pairs) produced perfect scores, which is not
credible evidence. Ten challenge pairs were added specifically to probe known
weak spots; five were missed (brand vs active ingredient, "per acre" vs "per
bigha" written without an explicit dose, a negated product phrased indirectly,
an interval stated only once). They remain unfixed and are reported, rather than
having the rules tuned to the challenge set and then re-scored on it.

### Verification

| Check | Result |
|---|---|
| Protection snapshot created before implementation | 308 files |
| Protection check before implementation | PASS (0 changed/missing/additions) |
| Focused conflict tests | **30 passed** in 0.22 s |
| Full suite | **148 passed** in 8.76 s (52 pre-Batch-B: 41 original + 11 Batch A; + 36 Phase 1 + 30 Phase 2 + 30 Improvement 1) |
| `verify_baseline.py --check` (original + Batch A) | **PASS** (270 files, 0 changed/missing/additions) |
| `verify_phase_outputs.py --check --allow-additions` | **PASS** (308 protected files, 0 changed, 0 missing; 16 approved new files) |
| Batch B documents modified | only `experiments/batch_b/README.md`, `research/BATCH_B_LOG.md` |

### Limitations

- Synthetic, same-author pair set; precision/recall are not real-world estimates.
  Expert adjudication of conflicts is **PENDING**.
- The corpus scan reports how often rules fire, not whether advice is wrong;
  group members may discuss different crops, stages or contexts.
- Brand/active-ingredient equivalence is not inferred (`rogor` vs `dimethoate`
  counts as a product conflict).
- 50% dosage tolerance is a trade-off, not an agronomic standard.
- Negation is clause-local and pattern-based; conditional advice is only partly
  handled.
- Local area units (bigha/kanal/guntha) are not converted; differences are
  reported as `basis` conflicts.
- At most the first five answers are compared; the scan covers grouped train+val.
- No backend, API or production-gate integration; escalation is available only
  through the optional wrapper.
- Batch C was not started; Improvement 2 was not started.

### Next

Improvement 2 (adversarial safety testing) began on 2026-11-03; see the entry
below. Improvement 1 outputs were frozen and not modified.

---

## 2026-11-03 — Batch B Improvement 2: adversarial safety testing

**Status:** COMPLETE for the approved synthetic scope. No expert review exists
and every case is synthetic, so the scores describe this authored set, not
real-world safety.
**Purpose:** measure the **unmodified** production safety gate against
adversarial and edge-case text, separate false positives from false negatives,
and document bypasses instead of tuning rules to the test set.

### What is measured

- `answer_risk()` — the shipped `_DOSE` / `_CHEMICALS` regexes.
- `decide("Cultural Practices", 0.95, 0.90, answer=text)` — a content-only gate
  slice where high-stakes, time-sensitive, low-confidence and topic mismatch
  rules cannot fire, so only answer screening can escalate.
- Gold labels (`unsafe` / `safe` / `ambiguous`) encode **escalation need**, not
  expert-adjudicated agronomic harm. Ambiguous cases are excluded from P/R/F1
  and reported separately.
- The production gate is imported read-only; its SHA-256 is stored in
  `metrics.json` (`modified_for_this_evaluation: false`).

### Dataset (authored, synthetic)

- **149 cases**: 81 unsafe, 58 safe, 10 ambiguous (excluded).
- Languages: English 104, Hinglish 26, Hindi 19. Subsets: core 64 (plain
  English), challenge 85 (Hindi, Hinglish, obfuscated names, indirect
  instructions, negation, unit variation). The split is authorship-based, not
  held out.
- Attack categories: dosage_pesticide 32, dosage_fertiliser 14,
  chemical_recommendation 5, obfuscated_chemical 10, indirect_instruction 10,
  unit_variation 10, negation 10, safe_control 39, false_positive_bait 9,
  ambiguous 10.
- No real farmers, advisors or experts were involved; no contact details are in
  the data. Labels were not adjudicated.

### Results (rule level, measured)

| TP | FP | FN | TN | Precision | Recall | F1 | F1 95% CI | Accuracy | False escalation | False safe |
|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|
| 63 | 23 | 18 | 35 | 0.7326 | 0.7778 | 0.7545 | [0.6790, 0.8208] | 0.7050 | 0.3966 | 0.2222 |

Gate slice: TP 63, FP 24, FN 18, TN 34, precision 0.7241, recall 0.7778,
F1 0.7500; **18 of 81 unsafe texts would still be auto-answered**.

| Subset | N | Recall | F1 | False safe |
|---|---:|---:|---:|---:|
| core | 58 | 1.0000 | 0.8136 | 0.0000 |
| challenge | 81 | 0.6842 | 0.7222 | 0.3158 |

| Language | Unsafe | Recall | Precision (rule) |
|---|---:|---:|---:|
| en | 54 | 0.8889 | 0.7164 |
| hi | 11 | 0.0000 | — |
| hinglish | 16 | 0.9375 | 0.7895 |

False negatives (18): dosage_pesticide 8, dosage_fertiliser 4,
obfuscated_chemical 4, indirect_instruction 2. Severity: 13 high, 5 medium.
All 11 Hindi dosage/instruction cases were missed. Other reproducible bypasses:
letter-spaced/hyphenated/digit-substituted product names (`m o n o c r o t o p h o s`,
`mono-crotophos`, `R0gor`), brands outside the vocabulary (`Dithane`), number
words with no Latin unit token (`two millilitres … one litre`), and non-standard
measures (`bottle`, `tank`, `bori`).

False positives (23): safe_control 10, negation 8, false_positive_bait 5,
Hinglish safe controls included. They come from `spray`/`spraying` being a
chemical signal regardless of negation, and from any number+dosage-unit pair
(pack sizes, prices, pump capacity, seed rates, sticker volumes). This direction
is conservative (escalation costs coverage, not safety) and is reported, not
suppressed.

Ambiguous, excluded: 10 cases; the rule fired on 4 and the gate escalated 4.

### Files created

| File | Purpose |
|---|---|
| `experiments/safety_evaluation/__init__.py` | Package marker |
| `experiments/safety_evaluation/build_cases.py` | Authors the 149-case synthetic set |
| `experiments/safety_evaluation/rules.py` | Read-only adapter over the production gate |
| `experiments/safety_evaluation/evaluate.py` | Scoring, bootstrap CIs, error lists |
| `experiments/safety_evaluation/report.py` | Report writer |
| `experiments/safety_evaluation/README.md` | Scope, labels, limitations |
| `experiments/safety_evaluation/data/adversarial_cases.csv` | The authored cases |
| `tests/unit/test_adversarial_safety.py` | 13 focused tests |
| `results/safety_evaluation/REPORT.md` | Measured report |
| `results/safety_evaluation/metrics.json` | Raw metrics, gate hash, protocol |
| `results/safety_evaluation/predictions.csv` | Per-case rule/gate predictions |
| `results/safety_evaluation/false_negatives.csv`, `false_positives.csv` | Separate error lists |

Modified (Batch B documents only): `experiments/batch_b/README.md`,
`research/BATCH_B_LOG.md`. The production gate and all original, Batch A,
Phase 1, Phase 2 and Improvement 1 files are unchanged.

### Verification

| Check | Result |
|---|---|
| Focused safety tests | **13 passed** in 1.40 s |
| Full suite | **161 passed** in 9.31 s (41 original + 11 Batch A + 36 Phase 1 + 30 Phase 2 + 30 Improvement 1 + 13 Improvement 2) |
| `verify_baseline.py --check` (original + Batch A) | **PASS** (270 files, 0 changed/missing/additions) |
| `verify_phase_outputs.py --check --allow-additions` | **PASS** (308 protected files, 0 changed, 0 missing; 16 previously approved files) |
| Improvement 1 inventory check | **PASS** (325 files, 0 changed/missing/additions) |
| Production gate hash in `metrics.json` | matches `src/confidence_gate.py`; not modified |

### Limitations

- Synthetic, same-author set; the author had read the regexes, so core recall is
  optimistic and challenge recall is pessimistic. Neither is a field estimate.
- Gold labels are authored escalation expectations, not expert-adjudicated
  agronomic harm; no expert review exists (PENDING).
- Written Hindi only; no ASR/OCR noise or spoken code-mixing.
- A single gate configuration was tested; threshold sensitivity is not measured.
- The verification snapshot's scan patterns do not yet include
  `experiments/safety_evaluation/` or `results/safety_evaluation/`; those new
  files are listed above instead of being silently counted as verified.
- No production rule was changed to raise the score. Bypasses remain open and
  are documented in the report.
- Batch C and Improvement 3 were not started.

### Next

Improvement 3 (out-of-scope detection) is **not started**; it waits for explicit
approval.

---

## 2026-11-03 — Protection gap closed (Improvement 2 inventory)

**Status:** COMPLETE. Protection coverage now includes Improvement 2.
**Why a new tool:** `verify_phase_outputs.py` hard-codes its phases and is itself
inside the frozen 308-file snapshot, so adding the safety-evaluation paths to its
scan list would have reported the tool itself as a changed protected file. The
scan definition was moved to data instead of editing frozen tooling.

### Files created

| File | Purpose |
|---|---|
| `experiments/batch_b/inventories/protection.json` | Roots, groups, explicit files, immutable prefixes |
| `experiments/batch_b/verify_inventory.py` | Data-driven snapshot/check tool |
| `experiments/batch_b/manifests/batch_b_protection_full.json` | 339-file protection inventory |

Modified (Batch B documents only): `experiments/batch_b/README.md`,
`research/BATCH_B_LOG.md`.

### Inventory composition

339 files: 220 protected original, 50 Batch A, 14 Phase 1, 24 Phase 2,
17 Improvement 1, 13 Improvement 2 (7 code/data + 5 results + 1 test), 1
inventory config. Improvement 2 files: `experiments/safety_evaluation/` (7),
`results/safety_evaluation/` (5), `tests/unit/test_adversarial_safety.py` (1).

### Verification

| Check | Result |
|---|---|
| `verify_inventory.py --snapshot` | 339 files, groups as above |
| `verify_inventory.py --check` | **PASS** (0 changed, 0 missing, 0 additions) |
| Full regression suite | **161 passed** in 8.34 s (41 original + 11 Batch A + 36 Phase 1 + 30 Phase 2 + 30 Improvement 1 + 13 Improvement 2) |
| `verify_baseline.py --check` (original + Batch A) | **PASS** (270 files, 0 changed/missing/additions) |
| `verify_phase_outputs.py --check batch_b_protection_hashes --allow-additions` | **PASS** (308 files, 0 changed, 0 missing) |
| `verify_phase_outputs.py --check batch_b_improvement1_hashes` | **PASS** (325 files, 0 changed/missing/additions) |
| `scripts/13_hybrid_retrieval.py` | still the pre-existing uncommitted edit; unchanged |
| Safety metrics/report | reproduced byte-identically in a scratch directory; originals not rewritten |

### Notes

- No production safety rule, metric, prediction or report was modified.
- The older 308-file and 325-file manifests remain unchanged historical
  baselines; the new inventory is additive.
- Future approved improvements add roots to `protection.json` and rebuild the
  full manifest; the tool code itself is not inside the inventory.
- Improvement 3 was not started.
