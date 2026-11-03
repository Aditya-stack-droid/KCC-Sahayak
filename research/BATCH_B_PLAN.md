# Batch B — approved implementation plan

**Status:** approved 3 November 2026; Phase 0 in progress.
**Rules:** additive only; no original file modified; Batch A outputs immutable;
no Batch C; human/expert work marked PENDING.

This document persists the plan approved by the project owner. Phase order is
fixed: **0 → 1 → 2 → 3 → 4 → 5 → 6 → 7 → 8**, one phase per approval.

---

## 1. Inspection summary

| Objective | Existing implementation | Gap addressed by Batch B |
|---|---|---|
| B1 annotation | `scripts/08`, `scripts/09`, `scripts/18`, `docs/annotation_guidelines.md`; review sheet 0/250 filled | uncertainty flag, adjudication, per-class kappa + CIs, relevance rubric, multi-annotator isolation, 300-English pilot |
| B2 safety | `src/confidence_gate.py`, `src/answer_quality.py`, `scripts/14`, `tests/test_gate.py` | adversarial multilingual test set, independent labels, false-safe/false-escalation, severity taxonomy |
| B3 conflict | top-1 retrieval; single-reference scoring | answer consolidation, numeric/unit/product conflicts, uncertainty score, escalation wrapper |
| B4 PII | none | recognizers, redaction, multilingual tests, adjudicated precision/recall |
| B5 OOS/multi-intent | none | synthetic evaluation sets, kNN-distance OOS, multi-label experiment, clarification rules |
| B6 retrieval | Batch A ranking metrics; one seed; no significance tests | paired bootstrap, McNemar, Holm, multi-reference, topic-matched slice |
| B7 tests | 41 original + 11 Batch A | behavioural and integration suite for new modules and failure modes |
| B8 docs | Batch A log | Batch B log, three-way comparison, version manifest |

**Corpus constraint:** all 138,875 question strings are Roman-script advisor-log
text. Real Hindi/Hinglish farmer queries do not exist in the repository, so a
real 300-per-register set cannot be produced without human collection. The
300-English pilot proceeds; Hindi/Hinglish remain PENDING and the workflow is
built to accept them later.

---

## 2. Preservation rules

- No modifications to `src/`, `backend/`, `frontend/`, `models/`, `data/`,
  `scripts/`, existing `tests/`, `paper/`, `docs/`, `README.md`,
  `research/EXPERIMENT_LOG.md`.
- No writes to `results/improved_evaluation/` or any original `results/*.json`.
- All new outputs go to `experiments/**` and new `results/` subdirectories.
- `experiments/batch_b/verify_baseline.py --check` runs before and after every
  phase; a non-zero exit blocks the phase.
- Original gate, retrieval, model and pipeline are imported read-only or
  reimplemented behind new wrappers; the original remains the fallback.

---

## 3. Phases

### Phase 0 — baseline verification and protection

Files: `experiments/batch_b/{README.md, common.py, verify_baseline.py}`,
`manifests/baseline_hashes.json`, `requirements-batch-b.txt`,
`research/BATCH_B_PLAN.md`, `research/BATCH_B_LOG.md`.

Checks: all 52 tests pass; manifest verifies; no original or Batch A file
changed. No experiment run.

### Phase 1 — PII detection and redaction

Files: `experiments/privacy/{recognizers.py, redact.py, audit.py, README.md}`,
`tests/unit/test_privacy_recognizers.py`.

Scope: Indian phone numbers, email, URL, Aadhaar-like (Verhoeff), IFSC,
bank-account-like, Devanagari digits, best-effort names (honorific patterns).
Span-based redaction with placeholders; idempotent; counts only in reports.

Evaluation: 100-row author-adjudicated sample (not independent expert review);
precision/recall with CIs; adversarial EN/Hindi/Hinglish tests. No raw PII in
any report or log.

### Phase 2 — human annotation infrastructure

Files: `experiments/human_annotation/{ANNOTATOR_HANDBOOK.md, make_sheets.py,
make_relevance_sheets.py, score_agreement.py, adjudicate.py, README.md}`,
`tests/unit/test_annotation_workflow.py`.

Scope: 300 real English corpus queries; two independent blind sheets with
uncertain flag; answer key path supplied externally and never committed;
adjudication sheet; intent kappa (human–human, human–LLM), per-class agreement,
confusion matrices, bootstrap CIs; retrieval relevance rubric (0/1/2).
Existing 150 LLM gold queries are a clearly labelled synthetic pilot only.

Status: `PENDING_HUMAN_ANNOTATION` until real sheets are returned. No kappa is
reported before then.

### Phase 3 — independent safety evaluation

Files: `experiments/safety_evaluation/{build_test_set.py, taxonomy.py,
evaluate_gate.py, report.py, README.md}`,
`tests/unit/test_safety_evaluation.py`.

Scope: independently authored synthetic cases across chemical advice, dosage,
fertiliser, treatment, weather, market price, schemes, ambiguity, high risk,
misleading confidence and irrelevant retrieval; EN/Hindi/Hinglish/Roman-Hindi;
unit variants, obfuscation, negation.

Metrics: false-safe, false-escalation, coverage, unsafe-auto-answer rate,
severity distribution, bootstrap CIs, threshold sensitivity. Labels are authored
independently of the gate regex. Synthetic results are not real-world safety or
agronomic correctness; expert review remains PENDING.

### Phase 4 — answer conflict detection

Files: `experiments/conflict_detection/{extract.py, consolidate.py,
conflict.py, review_report.py, README.md}`,
`tests/unit/test_conflict_detection.py`.

Scope: group answers by Batch A question group; embed answers; cluster;
extract numbers, units, intervals and product/crop names; detect dosage, unit,
product, interval and context conflicts; severity and descriptive uncertainty
score; recommend escalation. Every original answer is preserved; no automatic
merging and no assumption that the majority is correct. Reports are redacted.

### Phase 5 — out-of-scope, multi-intent and missing context

Files: `experiments/out_of_scope/{build_eval_set.py, detectors.py, evaluate.py,
README.md}`, `experiments/multi_intent/{build_eval_set.py, classify.py,
context.py, evaluate.py, README.md}`,
`tests/unit/{test_out_of_scope.py, test_multi_intent.py}`.

Scope: synthetic OOS categories (non-agricultural, general knowledge, health,
irrelevant, near-miss, ambiguous); kNN-distance and class-distance detectors;
synthetic multi-intent combinations; riskiest-intent aggregation; clarification
rules for missing crop/stage/season/location/pest. The original 10-class model
is untouched; results are synthetic and PENDING real validation.

### Phase 6 — extended retrieval evaluation

Files: `experiments/retrieval_evaluation/{compare.py, significance.py,
multi_reference.py, README.md}`,
`tests/unit/test_retrieval_significance.py`.

Scope: dense vs hybrid on identical query IDs; paired bootstrap and McNemar;
Holm correction; effect sizes; CIs; multi-reference max token-F1 and semantic
similarity; slices: novel (<0.80), near-duplicate (≥0.90), topic-matched
(top-1 weak label equals query weak label; for this slice report answer-level
metrics, not topic match). Batch A is read-only; outputs go to
`results/improved_retrieval/`.

### Phase 7 — behavioural and regression testing

Files: `tests/behavioural/{test_gate_edge_cases.py,
test_multilingual_inputs.py, test_malformed_queries.py}`,
`tests/integration/{test_retrieval_failures.py, test_new_modules_e2e.py}`.

Scope: negation, units, multilingual input, empty/malformed/ambiguous queries,
high confidence with unsafe answers, retrieval failures (empty index, short
top-k, missing metadata, NaN similarity), end-to-end new-module flow. All
existing tests must continue passing.

### Phase 8 — documentation and comparison

Files: `research/BASELINE_COMPARISON.md`; finalise `research/BATCH_B_LOG.md`
and `requirements-batch-b.txt`.

Scope: original baseline vs Batch A vs Batch B, with explicit comparability
caveats; commands, seeds, dependency versions, raw outputs, failed experiments,
limitations. No edits to the original paper, README or Batch A log.

---

## 4. Dependencies and resources

- No new packages required. Presidio is not installed and is not used without
  approval.
- CPU only; 8 GB RAM; ~17 GB free disk at approval.
- No external datasets, downloads or paid APIs.
- Human annotators and agronomists are unavailable to the agent; those results
  stay PENDING.

## 5. Integrity constraints

1. Every number traces to an executable run and a saved raw output.
2. LLM labels are never called human labels or gold.
3. Synthetic cases are never called real farmer data.
4. Safety performance on synthetic cases is never called real-world safety.
5. No improvement claim without measured, paired evidence.
6. Failed or partial runs are recorded.
7. Batch A and original artefacts are hash-verified before and after each phase.
