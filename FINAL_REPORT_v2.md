# KCC-Sahayak: Final Report (v2)

> This file is a **new** document created during the final comprehensive project audit
> (4 November 2026). It supersedes `FINAL_REPORT.md`, which is left unchanged. It
> incorporates the Batch A/B and Improvement 1–5 work and reports only figures that are
> traceable to a saved artefact under `results/`.

**Status:** working research prototype with an honest, auditable evaluation; human
verification of labels, real farmer speech, and agronomist review remain outstanding
(see Limitations).

An earlier version of this report claimed 99.37% intent accuracy. **That claim is
withdrawn.** It measured agreement with the keyword rules that generated the labels, not
understanding: against labels assigned by meaning, the same model scores 65.7%. Every
figure below is measured against labels assigned by meaning, with baselines, confidence
intervals and significance tests. Full detail: [`README.md`](README.md),
[`KCC_Sahayak_Verification_Audit.md`](KCC_Sahayak_Verification_Audit.md),
[`KCC_Sahayak_Verification_Audit_v2.md`](KCC_Sahayak_Verification_Audit_v2.md),
`paper/main.tex`, and `results/*.json`.

---

## 1. Objective and problem statement

Kisan Call Centres (KCC) route farmer calls to Farm Tele Advisors (FTAs) who answer
questions about crops, pests, fertiliser doses, weather, schemes and market prices. Calls
are logged as free-text question/answer pairs. The project asks a narrow, measurable
question rather than a generative one:

> Given a farmer's typed or spoken question, decide whether the system can **safely**
> return a previously logged human-advisor answer, or whether the question must be
> **escalated** to a human. Do this in a way that is measurable, auditable, and honest
> about its uncertainty.

This is a **selective-prediction / triage** problem, not an "answer everything" problem.
The system answers a low-risk subset automatically and refuses the rest, reporting
coverage and a rule-defined violation rate alongside accuracy.

**What the system is not.** It is not a generative agronomist (there is no LLM at
inference time — auto-answers are copied from logged human text); not a replacement for
KCC advisors (roughly three quarters of queries are escalated); not Haryana-specific
(the public corpus is overwhelmingly Assamese); not agronomist-validated; and not a
completed human study.

---

## 2. Architecture

End-to-end pipeline (`src/pipeline.py`, `backend/app.py`):

```
voice or text (English / Hinglish / Devanagari Hindi)
   -> Whisper "small" speech-to-text (voice only)          src/asr.py
   -> ONE multilingual sentence encoder                    paraphrase-multilingual-MiniLM-L12-v2
        |-> intent: sentence embedding + char n-gram TF-IDF -> logistic regression (temperature-calibrated)
        `-> retrieval: dense (FAISS) + TF-IDF fusion over ~125k logged answers
              (+ answers human advisors have written since deployment)
   -> crop / pest entity tagging (gazetteer)               src/ner.py
   -> SAFETY GATE  ->  AUTO_ANSWER  or  ESCALATE           src/confidence_gate.py
   -> SQLite log + advisor queue                           backend/db.py
```

The safety gate auto-answers only when **all** of the following hold; otherwise it
escalates and states why:

1. the predicted category is not high-stakes (Plant Protection, Nutrient Management,
   Government Scheme);
2. it is not time-sensitive (Weather, Market Price) — a logged answer is stale by
   construction;
3. intent confidence ≥ 0.70 **and** retrieval similarity ≥ 0.55;
4. the retrieved text is real advice, not a call-log note such as "explained in details";
5. the retrieved text contains no dosage or chemical instruction.

Escalated queries go to the **Advisor Console**; an advisor's answer is returned to the
farmer and added to the in-memory retrieval memory, closing a human-in-the-loop cycle. A
FastAPI backend serves the API and a static single-page front end (live demo, advisor
console, operations dashboard, evaluation dashboard); persistence is a single SQLite file
(`data/app.db`).

Production integration was verified during the audit by running the real pipeline: e.g.
*"which tomato variety is good for me?"* → AUTO_ANSWER (similarity 0.845); a Hinglish pest
query → ESCALATE (`high_stakes`); a Hindi price query → ESCALATE (`time_sensitive`).

---

## 3. Technical stack

| Layer | Technology |
|---|---|
| Language | Python 3.10+ (audited on 3.12.13) |
| Encoder | `sentence-transformers` (paraphrase-multilingual-MiniLM-L12-v2) |
| Intent | scikit-learn `LogisticRegression` + `TfidfVectorizer(char_wb 2–5)` |
| Retrieval | FAISS `IndexFlatIP` (dense) + scikit-learn TF-IDF (sparse), fused |
| ASR | `openai-whisper` small + ffmpeg |
| API | FastAPI + Uvicorn, Pydantic request models |
| Storage | SQLite (`sqlite3`, single `queries` table) |
| Front end | Plain HTML/CSS/JS, Chart.js (CDN) |
| Tests | pytest |

Dependencies are **not pinned** (except `accelerate>=1.1.0`), so exact numerical
reproduction across environments is not guaranteed; reported numbers come from the
committed artefacts.

---

## 4. Data and processing

**Source.** `data/raw/kcc_mapping_full.csv`, a two-column (`questions`, `answers`) public
mirror of the data.gov.in KCC transcript catalogue. It is real advisor-log text, but the
"questions" are advisor-written summaries of calls, not verbatim transcriptions, and there
is no state, date, language, crop or audio metadata.

| Quantity | Value |
|---|---:|
| Raw rows | 139,542 |
| Cleaned rows | 138,875 |
| Train / validation / test | 111,100 / 13,887 / 13,888 |
| Retrieval index (train + validation) | 124,987 |
| Call-log-note answers (index) | 8.75% (10,936 / 124,987) |
| Haryana vs Assam place mentions (of 138,875) | 17 vs 5,840 |

**Cleaning and labelling** (`scripts/01_prepare_data.py`): lowercase and whitespace
normalisation, drop rows shorter than 8 characters (667 rows), 80/10/10 stratified split
(`random_state=42`), weak keyword labels (`src/weak_labels.py`) and gazetteer entities.

**Labels are LLM-assisted, not human-verified.** A single LLM assistant labelled 3,496
corpus queries by meaning (3,000 train pool, 496 test pool) using
`docs/annotation_guidelines.md`. The 150-query gold set (50 each English/Hinglish/Hindi,
15 per class) and 1,200 spoken-style paraphrases were written by the same assistant. The
250-row human review sheet is empty; there is no kappa. Keyword labels agree with meaning
labels only **67.51%** of the time — the source of the retired 99% illusion.

**Leakage.** The split is by row, not by question identity: 44.98% of test question
strings appear exactly in training (46.34% in train or validation), and 70.73% of the
legacy test subset has a dense near-duplicate at cosine ≥ 0.95 (4.93% are novel, <0.80).
Random-row metrics therefore overstate generalisation; Batch A adds a grouped split to
address this.

---

## 5. Model methodology

**Production intent head** (`src/intent_model.py`): a logistic regression over the
concatenation of a normalised MiniLM sentence embedding and character 2–5-gram TF-IDF
features (C=10, embedding weight 1.0), with temperature scaling T=0.75.

- The embedding provides cross-lingual meaning (Hindi ↔ English).
- Character n-grams give in-language spelling evidence and tolerate Roman-Hindi variation
  (`gehu`/`gehun`/`gheu`).
- Temperature 0.75 is fitted on out-of-fold logits and reduces expected calibration error
  from 3.78% to 0.51% out-of-fold; on silver it slightly worsens ECE (1.77% → 2.32%).
  Calibration changes confidence, not argmax labels.

Training data: 3,000 meaning-labelled queries + 1,200 spoken-style paraphrases = 4,200
rows, 10 classes. A legacy DistilBERT fallback (`scripts/02`) is trained on weak labels and
is retained for the historical comparison, not production.

**Model selection** (`scripts/16_model_selection.py`) compares candidate heads on
`mean(silver, gold_en, gold_hi, hinglish_perturbed)` and selects `combo_lr_C10_w1`.
Because the gold set participates in selection, the gold metrics are
**selection-contaminated** and should not be treated as a clean untouched test set.

---

## 6. Retrieval methodology

`src/retrieval.py` builds candidates from both dense and sparse search over the 124,987
train+validation index:

```
fused score = alpha * dense_cosine + (1 - alpha) * tfidf_cosine,  alpha = 0.4
```

Candidates whose predicted topic equals the query topic are reranked first (topic-aware
reranking); non-informative call-log answers are filtered at query time (they remain in the
index). The `similarity` reported to the gate is the **dense cosine**, so the 0.55
threshold keeps one meaning across languages. Advisor answers live in a separate in-memory
dense-only store, not in FAISS/TF-IDF.

---

## 7. Safety mechanism

The gate (`src/confidence_gate.py`) is the project's core contribution: it refuses whole
high-stakes and time-sensitive categories, and it inspects the **retrieved answer text**
(not only the query) for call-log notes (`src/answer_quality.py`) and for dosage/chemical
language. It reports coverage, selective accuracy and rule-defined violations together, and
the ablation shows that confidence thresholds alone are unsafe under the project's own
policy (92.5% coverage but 71.7% violations), whereas the full gate cuts violations to
2.29% by escalating most queries.

A "violation" is compliance with a project-defined rule, not observed harm; no agronomist
reviewed the auto-answers.

---

## 8. Batch A — grouped, leakage-resistant evaluation

`experiments/grouped_evaluation/` re-splits the corpus by question identity
(exact-normalised-question + MiniLM cosine ≥ 0.95, transitive union-find;
`StratifiedGroupKFold`). Result: 125,044 index rows, 13,831 test rows, and leakage after
grouping of **0.09%** at ≥0.95 (vs 70.73% on the row split).

Five-seed intent refit: accuracy **0.9372 ± 0.0045**, macro-F1 0.9205 ± 0.0048; novelty
buckets 0.9630 (near-dup) / 0.9580 (similar) / **0.9025 (novel)**. Dense retrieval (all
13,831 test rows): topic match@1 0.8519, Recall@5 0.9508, MRR@10 0.8961, nDCG@5 0.9071,
answer token-F1 0.3213. On the identical 3,000-query sample, dense 0.8403 vs hybrid (α=0.4)
**0.8733** topic match@1. Relevance is a weak-label topical proxy, not agronomic
correctness.

---

## 9. Batch B — privacy (Phase 1) and annotation infrastructure (Phase 2)

**Phase 1 — PII detection/redaction** (`experiments/privacy/`): recognisers for phone,
email, Aadhaar-like numbers, IFSC/bank, URL and heuristic person names, with configurable
redaction. Evaluated on **87 hand-authored synthetic examples** (47 English, 21 Hindi, 19
Hinglish; 70 gold entities; 31 negative/ambiguous): balanced policy F1 **0.9559** (95% CI
0.9077–0.9882), strict 0.9403; document-level false-positive rate 0.0 on the 31 negatives.
These are synthetic, same-author results — **no corpus prevalence or recall claim is
made**, and redaction is not wired into the production gate or backend.

**Phase 2 — annotation infrastructure** (`experiments/annotation/`): a stratified 300-item
sample, two blinded annotators, Cohen/Fleiss kappa, confusion matrices, disagreement and
adjudication tracking, QC, and an offline UI. The workflow is tested, but **no human has
labelled any item**: `results/annotation/agreement.json` reports `status: PENDING`, all pair
tables are empty, and there is no adjudication. There is therefore **no human–human or
human–LLM kappa**, and the items must not be called human gold labels.

---

## 10. Improvements 1–5

The five improvements wrap **around** the audited system; no `src/`, `backend/`, `data/` or
model file was changed.

| # | Improvement | Method | Evidence class | Headline result |
|---|---|---|---|---|
| 1 | Answer conflict detection | Deterministic rules over quantities/units/bases/products/negation; escalate-only | Synthetic authored pairs + descriptive corpus scan | 77 pairs: P 1.00, R 0.9038, F1 **0.9495** (challenge subset F1 0.6154); corpus scan flags 25.6% of sampled answer pairs |
| 2 | Adversarial safety evaluation | Authored paraphrase/spelling/obfuscation/Hindi stress cases against the read-only gate | Synthetic, same-author | 149 cases: content-rule F1 **0.7545** (P 0.7326, R 0.7778); 18/81 unsafe missed; Hindi recall 0/11 |
| 3 | Multi-reference retrieval evaluation | Score against every distinct historical answer for the same question group | Corpus (historical answers as references) | Max-other-reference F1 **0.5788** vs own 0.3217 (paired +0.2415, CI [0.236, 0.247]); chance control negative (−0.0393) |
| 4 | Out-of-scope detection | kNN embedding distance vs MiniLM+LR classifier; family-grouped split | Synthetic negatives + real agricultural test rows | kNN F1 **0.9091**, LR F1 **0.9351**; full-test false rejection 6.25% / 3.63% |
| 5 | Behavioural regression testing | 168 deterministic probes over deployed rule-based components, no model loading | Authored synthetic probes | **109/168 passed (64.88%)**, 59 failed, 0 crashes |

**Improvement 1** authored a synthetic pair set (52 conflict, 25 no-conflict) and also
scanned the corpus for rule-fire rates (reported as rates, not confirmed errors).

**Improvement 2** imported `src/confidence_gate.py` read-only (SHA-256 recorded;
`modified_for_this_evaluation: false`). Its documented bypasses (Devanagari units/product
names, letter-spaced/hyphenated brand names, number words without a Latin unit,
non-standard measures) were left unfixed deliberately.

**Improvement 3** measures agreement with **historical advisor answers, not gold**:
contradictory advice can share a pool, so matching any of it is agreement, not correctness.
A known artefact is documented rather than hidden: 8.7% of rows have a call-log note as
their own "answer".

**Improvement 4** thresholds were selected on validation only under a fixed policy
(maximise out-of-scope recall subject to ≤2% false rejection). Neither detector is wired
into production. Mixed-intent queries can be accepted by the LR detector (6 of 16).

**Improvement 5** runs the deployed rule-based components read-only. Failures: 6 degraded,
24 false negatives, 10 false positives, 18 missing capability, 1 validation gap. Multi-
intent detection is absent (17 of 18 multi-intent probes fail with `single_label_only`);
Hindi evidence extraction and Devanagari localisation are gaps. No production component was
modified and no threshold was tuned on these cases.

---

## 11. Evaluation methodology

- **Silver set** — 496 held-out corpus queries labelled by meaning (LLM-assisted).
- **Gold set** — 150 independently authored queries in English/Hinglish/Hindi; independent
  of the corpus and training data, but not of the annotator, and used in model selection.
- **Coverage** — fraction of queries auto-answered.
- **Selective accuracy** — intent accuracy among auto-answered queries (not answer
  correctness).
- **Policy violation** — an auto-answer whose true label is high-stakes/time-sensitive or
  whose returned text is non-informative or contains dosage/chemical language; a
  rule-defined measure, not observed harm.
- **Answer F1** — lexical token overlap with the advisor's real answer, not semantic or
  agronomic correctness.
- **Statistics** — bootstrap 95% CIs; McNemar tests for paired classifier comparisons (no
  multiple-comparison correction); ablations and novelty stratification.
- **Grouped evaluation** (Batch A) and **synthetic adversarial/safety** suites (Batch B and
  Improvements) are reported separately and are **not comparable** with each other.

---

## 12. Verified results

Intent accuracy (labels by meaning):

| Model | Silver n=496 | Gold n=150 |
|---|---:|---:|
| Production `combo_lr_C10_w1` | **93.55%** [91.53, 95.56] | **90.0%** [85.33, 94.67] |
| char TF-IDF + SVM | 92.54% | 87.33% |
| word TF-IDF + SVM | 91.33% | 81.33% |
| embedding + LR | 90.52% | 74.0% |
| keyword rules | 66.94% | 37.33% |
| legacy DistilBERT (keyword labels) | 65.73% | 33.33% |

Gold by style: English 94%, Hinglish 90%, Hindi 86%; Hinglish with random re-spelling
85.6% ±2.33. Without spoken-style training text, gold falls to **60.0%**. Novelty
stratification: novel 87.71%, similar 95.35%, near-duplicate 98.62%. McNemar: production vs
char SVM p=0.40 (silver) / 0.48 (gold) — **statistically indistinguishable**; vs word SVM
p=0.015 (gold); vs embedding-only p=0.006 / <0.0001. The embedding's measurable value is
zero-shot cross-lingual transfer (Hindi 68% with no Hindi training data, vs 4% for char
n-grams).

Gate (default thresholds 0.70 / 0.55):

| Policy (silver) | Coverage | Selective acc. | Violations |
|---|---:|---:|---:|
| Thresholds only | 92.54% | 96.95% | 71.68% |
| + high-stakes | 50.20% | 96.39% | 48.19% |
| + time-sensitive | 44.76% | 95.95% | 41.89% |
| + answer screening | 26.81% | 94.74% | 3.01% |
| + topic match (full) | 26.41% | 95.42% | **2.29%** |

Gold full gate: coverage 23.33%, selective accuracy 94.29%, violations 2.86%. Answer
screening in isolation (n=3,000): coverage 29.23% → 16.47%, dosage/chemical auto-answers
42.76% → **0.0%**.

Retrieval answer F1 (n=2,737; n=143 novel): dense-only 0.4514 / 0.1887; TF-IDF-only 0.4587
/ 0.2343; hybrid α=0.4 **0.4673 / 0.2446**.

Voice (50 synthetic English TTS queries): mean WER **1.46%**, median 0; intent accuracy 94%
typed → 94% after ASR; prediction unchanged 100%. This is an upper bound; Hindi speech and
real telephone audio were not tested.

Corpus audit: keyword-label validity 67.51%; near-duplicate test share 70.73% (≥0.95);
call-log notes 8.75%; 17/138,875 Haryana mentions. Batch A grouped split removes the leakage
(0.09% at ≥0.95) and reports intent 0.9372, retrieval topic match@1 0.8519.

> **Note on working-tree drift.** Four files are modified but uncommitted
> (`results/calibration.json`, `results/hybrid_retrieval.json`,
> `results/model_selection.json`, `scripts/13_hybrid_retrieval.py`). The regenerated
> working-tree values differ slightly from the committed ones (e.g. hybrid α=1.0 all
> 0.4514→0.4545, α=0.4 all 0.4673→0.4679; selected-model silver 0.9355→0.9375; calibrated
> silver ECE 0.0232→0.0237). The numbers above are the **committed** values; both sets are
> internally consistent with their own run. They were preserved untouched by this audit.

---

## 13. Technical achievements

1. A working, voice-capable triage prototype (FastAPI + SQLite + advisor console +
   dashboards) with a documented runtime path.
2. A corpus audit quantifying label validity, near-duplicate leakage, call-log-note
   prevalence and geography.
3. A meaning-based annotation set (3,496) and an independent 150-query multilingual gold
   set.
4. A hybrid dense+sparse retrieval system with a real, reproducible novel-query gain.
5. A content-aware safety gate with a quantitative ablation showing what each rule buys in
   coverage, selective accuracy and rule-defined violations.
6. A human-in-the-loop feedback path from escalation to advisor answer to retrieval memory.
7. Batch A/B improvements: a leakage-resistant grouped evaluation, PII detection with
   synthetic evaluation, an annotation-agreement infrastructure, answer conflict detection,
   adversarial safety stress tests, multi-reference retrieval evaluation, out-of-scope
   detection and a behavioural regression suite.
8. An honest reproducibility package: numbered scripts, result JSONs, a 224-test suite and
   documented limitations.

---

## 14. Known limitations

- **Model artefacts are not committed** (`models/` is gitignored); a fresh clone cannot
  start the app until scripts 03, 16 and 13 are run.
- **Labels are LLM-assisted and not human-verified**; no human–human or human–LLM kappa
  exists.
- **Gold set shares the annotator with augmentation** and is used in model selection, so
  gold metrics are optimistic and selection-contaminated.
- **Severe leakage on the original row split** (44.98% exact test/train question overlap;
  70.73% near-duplicate share). The grouped split (Batch A) mitigates this for evaluation.
- **Corpus is Assam-heavy and English/Romanized**; no state-specific or broad multilingual
  claim is supported.
- **No answer-correctness or agronomist evaluation**; "selective accuracy" and "policy
  violation" are intent/rule measures, not measured harm.
- **Raw corpus contains PII-like content** (phone/email/address/bank matches) and no
  redaction is applied to logs.
- **Speech was tested only on synthetic English TTS**; Hindi/Hinglish and telephone audio
  are untested.
- **Dependencies are unpinned**; exact numerical reproduction is not guaranteed.
- **No authentication, rate-limiting, encryption, retention or deployment hardening.**
- **Stale generated paper assets** (`paper/generated/numbers.tex`) and a documented
  124,987-vs-138,875 denominator conflation in `paper/main.tex`.
- **Grouped retrieval uses one split seed**; multi-reference evaluation has no human
  relevance labels.

---

## 15. Future scope

1. Two human annotators complete the 250-row / 300-item review; report human–human and
   human–LLM Cohen's/Fleiss kappa and adjudicate disagreements.
2. Collect real Hindi/Hinglish/English telephone speech and report WER under noise.
3. Independent agronomist review of a stratified sample of auto-answers and flagged
   conflicts, reporting harm severity.
4. A held-out, question-level final test set with model selection on validation only.
5. Fine-tune MuRIL/IndicBERTv2 and complete the encoder comparison.
6. Run the KisanQRS-style retrieval baseline (`scripts/22`) and an LLM+RAG comparison with
   citation/safety constraints.
7. PII redaction wired into indexing/display behind access control, plus a corpus PII
   prevalence study.
8. Learned risk classification (coverage–risk curves) compared with the regex backstop.
9. Multi-intent detection and missing-context clarification.
10. Persist and de-duplicate advisor memory; add concurrency controls.
11. Authentication, rate limiting, TLS, retention and monitoring.
12. Regenerate paper assets from final results and correct the corpus-size and PII claims.

---

## Appendix — source-of-truth and reproduction

Primary result files: `results/research_eval.json` (intent, CIs, McNemar, ablations,
novelty), `results/selective_eval.json` (gate + ablation), `results/hybrid_retrieval.json`,
`results/calibration.json`, `results/answer_screening.json`, `results/voice_eval.json`,
`results/annotation_summary.json`, `results/corpus_geography.json`, and the per-improvement
directories under `results/`.

```bash
# Environment (macOS/Linux); use OMP_NUM_THREADS=1 to avoid a PyTorch/FAISS OpenMP segfault
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Rebuild the app artefacts (models/ is gitignored)
python scripts/03_build_retrieval_index.py
python scripts/16_model_selection.py --save
python scripts/13_hybrid_retrieval.py     # optional: hybrid retrieval

# Run
python -m uvicorn app:app --app-dir backend --port 8420
# Tests
OMP_NUM_THREADS=1 pytest -q     # 224 passed
```

Reported numbers are from the committed artefacts; a fresh re-run may differ slightly
because dependencies and pretrained-model revisions are not pinned.
