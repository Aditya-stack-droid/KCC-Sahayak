# KCC-Sahayak — Dataset Authenticity and Experimental Verification Audit

**Repository:** `KCC-Sahayak` (branch `improvements`, single commit `cf6aab3`, “Initial commit from ZIP source”)
**Audit date:** 1 November 2026
**Last updated:** 4 November 2026 — §10 now covers Batch A through Improvement 5; §11 remains the artefact index.
**Auditor role:** independent technical audit; no application, training script, dataset or model file was modified.
**Companion documents:** `KCC_Sahayak_Research_Report.pdf`, `KCC_Sahayak_Research_Report.docx`, `KCC_Sahayak_Presentation.pptx`, `KCC_Sahayak_Diagrams/`.

---

## 0. Executive summary

| Question | Answer |
|---|---|
| Is the corpus real? | **Yes, with qualifications.** It is real KCC question/answer log text, traceable to a Kaggle mirror of the data.gov.in KCC transcript catalogue. It is **not** verbatim farmer speech, **not** audio, and has no state/date/language metadata. |
| Are the labels real? | **No.** All 3,496 training/silver labels, the 150-query gold set and the 1,200 paraphrases were produced by one LLM assistant. The human review sheet has **0 of 250** labels completed. |
| Are the queries/answers real? | Queries are advisor-written summaries of calls (“asking about…”), not raw utterances. Answers are real advisor log text, but 8.75% are call-log notes rather than advice, and the corpus contains names, phone numbers, emails and bank details. |
| Is the speech real? | **No.** 50 English queries were synthesised with Windows SAPI TTS. There are no farmer recordings, and Hindi/Hinglish speech was not tested. |
| Are the headline results reproducible? | **Largely yes.** 93.75% silver accuracy, 90.0% gold accuracy, 27.0%/95.5%/2.24% gate performance, 0.4679/0.2454 hybrid retrieval, 67.51% label agreement, 70.73% near-duplicate leakage and 8.75% call-log-note share were independently reproduced. Some secondary results are only partially reproducible or not reproducible. |
| Is the system deployment-ready? | **No.** No human-verified labels, no agronomist review, no real speech, no privacy/security engineering, and severe train–test leakage. |
| Are there unsupported claims? | **Yes.** The paper’s “only partly human-verified” and “no personal data” statements, the use of 124,987 as the “cleaned” corpus size, and several stale generated numbers are unsupported or incorrect. |

**Most important finding:** the quantitative results are more honest than the surrounding prose. The model is real and the main metrics reproduce; the evidence base (labels, gold set, speech) is synthetic, and the paper overstates the extent of human verification and privacy protection.

**Update, 4 November 2026:** Batch A (grouped, leakage-resistant evaluation) and Batch B (privacy, annotation infrastructure, answer conflict detection, adversarial safety evaluation, multi-reference retrieval evaluation, out-of-scope detection) were added **around** the audited system. Nothing in `src/`, `backend/`, `data/`, `models/` or any original result file was changed. The findings above remain the audit of record for the shipped project; current status, newly verified metrics, known limitations and remaining gaps are in §10.

---

## 1. Scope and method

The audit did the following:

1. Read every source module (`src/`), the backend (`backend/app.py`, `db.py`), tests, all 23 numbered scripts, the frontend, the README, the final report, the paper source, the annotation guidelines, the synopsis update and the existing reverse-engineering guide (`this_is_it.md`).
2. Inspected every dataset and artefact: raw corpus, processed splits, annotation chunks/labels, gold set, review sheet, model artefacts, FAISS index, TF-IDF matrix, result JSON/CSV files and synthetic audio.
3. Re-executed the test suite and the principal experiments from the shipped artefacts and by refitting the documented pipeline.
4. Recomputed corpus statistics, label agreement, leakage, geography, PII-like content and calibration directly from the files.
5. Traced the dataset externally by matching distinctive rows against public sources.
6. Compared every material claim in the README, `FINAL_REPORT.md`, `paper/main.tex` and `paper/generated/numbers.tex` with the evidence.

**Environment:** macOS (Apple Silicon), Python 3.12.13, PyTorch 2.15.0, scikit-learn 1.9.1, sentence-transformers 6.2.0, Transformers 5.22.1, FAISS CPU. No GPU was used. The working tree contained four uncommitted modifications (`results/calibration.json`, `results/hybrid_retrieval.json`, `results/model_selection.json`, `scripts/13_hybrid_retrieval.py`) and an untracked `.venv312/`; the audit used the working-tree artefacts.

---

## 2. Dataset authenticity and provenance

### 2.1 What the repository says

The repository contains **no source URL, DOI, download date, licence file, checksum or collection note** for `data/raw/kcc_mapping_full.csv`. The README calls it “the public multi-state KCC corpus”; `docs/SYNOPSIS_UPDATE.md` says the official Haryana export is CAPTCHA-protected and that the public corpus has almost no Haryana rows.

**Therefore, from repository artefacts alone, dataset provenance cannot be independently verified.** This statement is made explicitly, as required.

### 2.2 External provenance evidence found during the audit

A distinctive first row was searched verbatim:

> “asking about the control measure for aphid infestation in mustard crops” → “suggested him to spray rogor@2ml/lit.at evening time.”

This matches the public Kaggle dataset **“Farmers Call Query (KCC) Data”** by **Das Koushik**:

- URL: <https://www.kaggle.com/datasets/daskoushik/farmers-call-query-data-qa>
- Kaggle description: “A question-answer dataset generated from data.gov.in.”
- Page snippet: “Data is of Kisan Call Centre where farmers called for some query over phone call, and experts replied with some answers. Dataset has two columns: questions …”
- Updated approximately four years before the audit (i.e., c. 2021–2022).

The corresponding government catalogue entry is:

- **Kisan Call Centre (KCC) – Transcripts of farmers queries and answers**, data.gov.in, Government of India, attributed to the Ministry of Agriculture and Farmers Welfare, **Open Government Licence – India**. The IndiaAI/AIKosh catalogue record redirects to it:
  <https://aikosh.indiaai.gov.in/home/datasets/details/kisan_call_centre_kcc_transcripts_of_farmers_queries_and_answers.html>
  <https://www.data.gov.in/resource/kisan-call-centre-kcc-transcripts-farmers-queries-answers>

**Assessment:** the chain “data.gov.in KCC catalogue → Kaggle mirror → local CSV” is well supported by the exact row match and the two-column schema. It is **not** byte-for-byte verified: the local file has 139,542 rows, but the Kaggle page does not expose a static checksum or row count to this audit, and the local download date is unknown. The local file may be a subset, a later version, or a concatenation of multiple KCC exports.

### 2.3 What is real and what is synthetic

| Component | Source | Real / synthetic | How it was created | Evidence |
|---|---|---|---|---|
| `data/raw/kcc_mapping_full.csv` (139,542 Q/A) | data.gov.in KCC catalogue via Kaggle mirror (Koushik, 2021) | **Real records**, but advisor-written summaries | Advisors logged a summary of each call and the advice given; mirrored as two text columns | Exact first-row match; Kaggle description; 139,542 rows; 17.5 MB |
| Cleaned splits (138,875) | Derived from raw file | Real text | Lowercase, whitespace collapse, length filter, dedup, 80/10/10 stratified split | `scripts/01_prepare_data.py`; counts recomputed |
| Weak query-type labels | Keyword glossary | **Rule-generated** | Keyword matching over ten categories | `src/weak_labels.py`, `src/glossary.py`; 67.51% agreement with meaning labels |
| Meaning labels (3,496) | One LLM assistant (Claude) | **Synthetic annotation** of real text | LLM read each query and applied written guidelines | `data/annotation/chunks/`, `labels/`, `annotated.csv`; no human review |
| Gold set (150) | Same LLM assistant | **Synthetic queries** | LLM wrote 15 queries per class in EN/Hinglish/Hindi and assigned labels | `scripts/06_build_gold_set.py`; no exact overlap with corpus |
| Spoken-style augmentation (1,200) | Same LLM assistant | **Synthetic paraphrases** | 400 train-pool queries × 3 registers; label carried over | `data/annotation/aug/*.txt`; scripts 11–12; source-pool assertions |
| Synthetic speech (50 wav) | Windows SAPI TTS | **Synthetic audio** | English gold queries read by installed TTS voices | `demo/voice_eval/*.wav`, `spec.json`; no Hindi voices |
| Human review labels | Intended human annotators | **Intended real**, not performed | 250-row sheet generated; no rows filled | `data/review/review_sheet.csv`; `human_label` empty; answer key absent |

### 2.4 Direct evidence from the files

- Raw rows: **139,542**; unique question strings: **87,292**; duplicate-question groups: **14,969**; extra rows in duplicate groups: **52,249**.
- Cleaned rows: **138,875**; unique questions: **87,046**.
- Splits: train **111,100**, validation **13,887**, test **13,888**.
- Every question is Roman script. Answers contain **144** rows with Devanagari characters and **111** with Bengali characters.
- **44.98%** of test question strings appear exactly in the training split.
- **70.73%** of the 3,000 legacy test queries have a dense cosine similarity ≥ 0.95 to an indexed train/validation question; **4.93%** are below 0.80.
- **10,936 / 124,987 (8.75%)** index answers are classified as non-informative call-log notes.
- Geography regex: **17 / 138,875** queries mention a Haryana place; **5,840 / 138,875** mention Assam places or terms.
- PII-like regex counts in raw answers: **1,379** rows match a 10-digit Indian mobile pattern; **663** contain an email; **72** contain a URL; **1,016** mention Aadhaar. These are regex counts, not manually confirmed identifiers.

### 2.5 Signs of modification, filtering or fabrication

- **No evidence that rows were fabricated.** The content is internally consistent with real call-log text: regional references (Assam districts, AAU/Jorhat, local KVKs), inconsistent spelling, duplicated questions with different answers, and advisor-specific phrasing.
- **Clear evidence of derivation and filtering.** The local file is a two-column derivative that has dropped the district/month fields advertised by the government catalogue. Cleaning removed 667 short/empty rows and lowercased all text.
- **Clear evidence of synthetic supervision.** Labels, gold queries, paraphrases and speech are LLM- or TTS-generated. They are not independent human ground truth.
- **Unverifiable details.** Original download date, exact Kaggle version, checksum, KCC reporting period, licence acceptance, and whether PII was present in the source or introduced by the mirror.

### 2.6 Authentication conclusion

The dataset is **real KCC log text with synthetic labels, synthetic evaluation queries and synthetic speech**. It is **not** a real farmer speech corpus and **not** a human-annotated benchmark. The repository should describe it in exactly those terms. “Dataset provenance could not be independently verified using the available project artifacts” is true of the repository in isolation; the external row match supplies strong, but not checksum-level, provenance.

---

## 3. Experimental verification

### 3.1 Commands executed

```bash
.venv312/bin/python -m pytest tests -q
# 41 passed in 6.65s

# Reproduce deployed-model accuracy and per-class metrics
# (load model.joblib, encode silver+gold with MiniLM, argmax)
# See Appendix E of the report for the exact snippet.

# Reproduce gate behaviour on 496 silver + 150 gold queries
OMP_NUM_THREADS=1 .venv312/bin/python ...   # pipeline probe

# Reproduce answer screening on 3,000 test queries
OMP_NUM_THREADS=1 .venv312/bin/python ...   # probe + gate masks

# Reproduce hybrid retrieval at alpha = 0.40 and 1.0
OMP_NUM_THREADS=1 .venv312/bin/python ...   # HybridRetriever.search

# Recompute leakage, geography, calibration, label agreement
.venv312/bin/python ...
```

A macOS-specific issue was found: importing PyTorch and then running a hybrid FAISS + SciPy search can **segfault** because both libraries link OpenMP. `OMP_NUM_THREADS=1` avoids it. This is an environment constraint, not a code defect; all reported reproductions above used that setting.

### 3.2 Main verification table

Status labels: **VERIFIED** (independently reproduced), **PARTIALLY VERIFIED** (some evidence, not fully re-run), **NOT REPRODUCIBLE** (artefacts/environment missing), **UNSUPPORTED CLAIM** (no evidence in repository), **INCORRECT** (contradicted by evidence).

| Reported result | Source file / script | Independently reproduced? | Verified result | Status | Comments |
|---|---|---|---|---|---|
| Intent accuracy 93.5% on held-out corpus calls (n=496) | `README.md`; `results/research_eval.json`; `scripts/17_research_eval.py` | Yes | **93.75%** (95% CI 91.7–95.8) | VERIFIED | Saved artefact and refit agree; README rounds down |
| Intent accuracy 90.0% on independent gold set (n=150) | `README.md`; `results/research_eval.json` | Yes | **90.00%** (85.3–94.7) | VERIFIED | Exact |
| Gold accuracy EN 94 / Hinglish 90 / Hindi 86 | `README.md`; `research_eval.json` | Yes | **94 / 90 / 86** | VERIFIED | Exact |
| Hinglish with random re-spelling 85.6% | `README.md`; `model_selection.json` | No | — | PARTIALLY VERIFIED | Five-seed perturbation not re-run |
| Without spoken-style training text, gold 60.0% | `README.md`; `research_eval.json` | Yes | **60.00%** (EN 84 / Hing 30 / HI 66) | VERIFIED | Refit reproduced exactly |
| Gate: auto-answered 26.4% / accuracy 95.4% / violations 2.3% | `README.md`; `selective_eval.json` | Yes | **27.02% / 95.52% / 2.24%** (silver) | VERIFIED | Small drift from rebuilt TF-IDF; gold 23.33/94.29/2.86 matches exactly |
| Thresholds only: 92.3% auto-answered, 72.3% violations | `README.md`; `selective_eval.json` | Yes | **92.34% / 70.96%** (silver) | VERIFIED | README’s 72.3 differs slightly from stored 71.68 |
| Answer F1 dense→hybrid 0.451→0.467; novel 0.189→0.245 | `README.md`; `hybrid_retrieval.json` | Yes | **0.4548→0.4679; 0.1866→0.2454** | VERIFIED | α=0.4 and α=1.0 recomputed exactly |
| Character n-gram SVM tied (McNemar p=0.40 / 0.48) | `README.md`; `research_eval.json` | Partly | Gold **p=0.4807**; silver **p=0.2863** | PARTIALLY VERIFIED | Gold exact; silver differs, conclusion unchanged |
| Keyword labels agree with meaning 67.5% | `annotation_summary.json` | Yes | **67.51%** | VERIFIED | Exact |
| 70.7% of test queries have a near-duplicate | `research_eval.json` | Yes | **70.73%** (≥0.95); 83.53% (≥0.90) | VERIFIED | Recomputed from embeddings + FAISS |
| 8.7% of logged answers are call-log notes | `research_eval.json` | Yes | **8.75%** (10,936 / 124,987) | VERIFIED | Exact |
| 17 Haryana vs 5,840 Assam mentions (of 138,875) | `corpus_geography.json`; `scripts/09_merge_annotations.py` | Yes | **17 / 5,840** | VERIFIED | Regex rerun |
| Old DistilBERT 99.5% on keyword test, 65.7% on meaning | `README.md`; `baselines.json`; `research_eval.json` | No | 33.3% gold / 65.7% silver (stored) | PARTIALLY VERIFIED | `models/intent_classifier/` is absent; predictions could not be regenerated |
| Synthetic speech WER 1.46%, ASR intent accuracy 94% | `results/voice_eval.json`; `scripts/15_voice_eval.py` | Arithmetic only | **1.46% / 94%** | PARTIALLY VERIFIED | All aggregates recomputed from `voice_eval_rows.csv`; ASR not re-run (Whisper checkpoint not cached); audio synthetic English only |
| Calibration ECE silver 0.0177→0.0237 | `calibration.json` | Yes | **0.0177 / 0.0237** | VERIFIED | Recomputed from saved head |
| Calibration ECE out-of-fold 0.0381→0.0060 | `calibration.json` | No | — | NOT REPRODUCED | Requires 5-fold refit; not attempted |
| Encoder comparison: IndicBERTv2 91.3% > deployed 89.3% | `encoder_comparison.json` | No | — | PARTIALLY VERIFIED | File declares itself partial; fine-tuning not run |
| 41 unit/API tests pass | `tests/` | Yes | **41 passed** | VERIFIED | `pytest -q` at audit time; the suite is now 188 tests across legacy + Batch A/B files (§10.2) |
| Demo query “which tomato variety is good for me?” auto-answered | `README.md` | Yes | AUTO_ANSWER, similarity 0.845 | VERIFIED | Run with `OMP_NUM_THREADS=1`; Hinglish pest and Hindi price queries escalated |
| Original 99% accuracy claim | Earlier project version | No | Keyword agreement only | INCORRECT | Withdrawn in README; same model scores 65.7% against meaning labels |
| Paper: 8.2% call-log answers | `paper/generated/numbers.tex` | Yes | **8.75%** | INCORRECT | Stale generated asset |
| Paper: “only partly human-verified” (`κ = n/a`) | `paper/main.tex` | Yes | **0 human labels** | UNSUPPORTED CLAIM | Review sheet empty; no `human_agreement.json` |
| Paper: “No personal data about farmers is used; the corpus is public” | `paper/main.tex` | No | 1,379 phone-like, 663 email, 1,016 Aadhaar mentions | UNSUPPORTED CLAIM | Regex counts; not manual PII verification |
| Paper: corpus of 124,987 “unique pairs after cleaning” | `paper/main.tex`; `numbers.tex` | Yes | Cleaned **138,875**; indexed **124,987** | INCORRECT | Denominator conflation |
| Paper: voice metrics `n/a` | `paper/generated/numbers.tex` | Yes | `voice_eval.json` exists | INCORRECT | Stale generated asset |

### 3.3 Dataset and annotation verification details

| Quantity | Reported / expected | Recomputed | Status |
|---|---:|---:|---|
| Raw rows | 139,542 | 139,542 | VERIFIED |
| Unique raw questions | 87,293 (existing guide) | **87,292** | Minor discrepancy (one duplicate/null difference) |
| Cleaned rows | 138,875 | 138,875 | VERIFIED |
| Train / val / test | 111,100 / 13,887 / 13,888 | same | VERIFIED |
| Retrieval index | 124,987 | 124,987 | VERIFIED |
| Annotated train / test | 3,000 / 496 | 3,000 / 496 | VERIFIED |
| Augmented rows | 1,200 | 1,200 | VERIFIED |
| Gold rows | 150 (50 per style, 15 per class) | same | VERIFIED |
| Keyword vs meaning agreement | 67.51% | 67.51% | VERIFIED |
| Non-informative answers | 8.75% | 8.75% | VERIFIED |
| Haryana / Assam regex counts | 17 / 5,840 | 17 / 5,840 | VERIFIED |
| PII-like rows | not reported | 1,379 phone / 663 email / 72 URL / 1,016 Aadhaar | NEW FINDING |

### 3.4 Gold-set independence

| Check | Result |
|---|---|
| Gold query exactly present in raw corpus | 0 |
| Gold query exactly present in annotated train | 0 |
| Gold query exactly present in augmented data | 0 |
| Max cosine similarity to annotated train ≥ 0.90 | 0.0% |
| Max cosine similarity to annotated train ≥ 0.80 | 11.3% |
| Mean max similarity to annotated train | 0.691 |
| Gold vs augmentation max similarity ≥ 0.90 | 22.0% |

**Interpretation:** the gold set is lexically independent of training, which is a genuine strength. It is **not** independent of the annotator: the same LLM wrote the gold queries, assigned their labels, and wrote the paraphrase training data. The 22% high-similarity overlap with augmentation indicates shared style and vocabulary, so multilingual accuracy is likely optimistic.

---

## 4. Data leakage investigation

### 4.1 Mechanism

`scripts/01_prepare_data.py` splits **rows**, not question identities:

```python
train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["query_type"])
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df["query_type"])
```

The raw corpus contains 14,969 duplicate question strings. The same question can therefore appear in train and test with a different answer.

### 4.2 Magnitude

| Measure | Value |
|---|---:|
| Test question strings exactly present in train | **44.98%** (6,247 / 13,888) |
| Test queries with dense similarity ≥ 0.95 to index | **70.73%** |
| Test queries with dense similarity ≥ 0.90 | **83.53%** |
| Test queries with dense similarity < 0.80 | **4.93%** |
| Retrieval category match, near-duplicate (≥0.95) | **98.07%** |
| Retrieval category match, novel (<0.80) | **73.65%** |
| Intent accuracy, near-duplicate silver (≥0.90) | **98.62%** |
| Intent accuracy, novel silver (<0.80) | **87.71%** |

### 4.3 Effect on claims

- The **93.75% silver** and **93.3% retrieval category match** figures are inflated by row-level leakage. They are not wrong as descriptions of performance on this split, but they are not estimates of performance on new farmers' questions.
- The **90.0% gold** figure is the more defensible generalisation estimate because the gold set has no exact overlap and no pair above 0.90.
- Novel-query performance is **87.7% intent** and **73.7% retrieval category match** — lower, but still useful for triage.
- The project partially mitigates this by reporting novelty-stratified results and an independent gold set. It does **not** re-split by question identity, so the leakage remains in every random-row metric.

**Severity: high.** Any future report must either use a grouped split or foreground the novel/gold results.

---

## 5. Hallucination and unsupported-claim investigation

| Claim | Location | Finding | Severity |
|---|---|---|---|
| “Only partly human-verified” / human–LLM κ = n/a | `paper/main.tex` | **No human labels exist.** `data/review/review_sheet.csv` has 250 rows and an empty `human_label` column. No `results/human_agreement.json`. “Partly” overstates zero. | High |
| “No personal data about farmers is used; the corpus is public” | `paper/main.tex` ethics section | Raw answers contain 1,379 phone-like numbers, 663 emails, 72 URLs and 1,016 Aadhaar mentions (regex, unverified). Public availability does not make PII non-personal. | High |
| Corpus of 124,987 pairs “after cleaning” | `paper/main.tex` | 124,987 is train+validation index size; cleaned corpus is 138,875. | Medium |
| Call-log share 8.2% | `paper/generated/numbers.tex` | Current value is 8.75%; generated asset is stale. | Medium |
| Voice metrics n/a | `paper/generated/numbers.tex` | `voice_eval.json` exists (WER 1.46%, accuracy 94%). | Medium |
| 99.37% / 99% accuracy | earlier report / FILENAME | Was agreement with keyword rules; withdrawn. Scored against meaning, the same model is 65.7% on silver. | Resolved by withdrawal; do not resurrect |
| “Independent” gold set | README/paper | Independent of corpus and training data, **not** independent of the LLM annotator. Paper discloses the shared author, README does too; the word “independent” should always carry that qualifier. | Medium |
| “Policy violation 2.3%” | README | Reproduced 2.24%, but the metric and the filter share the same regex; it is compliance with a project-defined rule, not independent harm. README does disclose this. | Medium |
| “Voice-first” | README/title | Interface is voice-capable, but no real farmer audio exists and ASR was tested only on synthetic English TTS. Disclosed in limitations. | Low–medium |
| “Haryana” framing | early synopsis / literature doc | Corpus has 17 Haryana mentions vs 5,840 Assam; `docs/KCC-Sahayak_20_Papers_Gap_Analysis.md` still contains Haryana-specific claims. README and synopsis update correct this. | Medium (stale doc) |
| 87,293 unique raw questions | `this_is_it.md` | Recomputed 87,292 (one null/empty question explains the difference). | Low |

**No evidence of fabricated experimental numbers.** The result files are internally consistent, reproduce at the stated values, and show ordinary run-to-run drift. The problems are overstatement in prose and stale generated assets, not invented data.

---

## 6. Reproducibility status

### 6.1 What works

- The Python package set in `requirements.txt` installs and imports (audit used a Python 3.12 environment).
- `pytest tests` → 41 passed.
- The deployed intent artefact loads and reproduces 93.75% silver / 90.0% gold.
- Refitting the documented head reproduces the no-augmentation 60.0% result and the character-SVM comparison.
- The gate, screening and hybrid retrieval reproduce at the stated values (small drift where TF-IDF was rebuilt).
- The demo runs and produces the documented decisions once `OMP_NUM_THREADS=1` is set on macOS.

### 6.2 What is missing or not reproducible

| Missing / blocked artefact | Consequence |
|---|---|
| `models/intent_classifier/` (legacy DistilBERT) | The 99.5% keyword-label and 65.7% meaning-label DistilBERT results cannot be regenerated. `.gitignore` excludes `models/`, and the directory is absent. |
| `results/cache/` (referenced by `scripts/17_research_eval.py`) | Cached embeddings absent; script must re-encode, which is possible but slow. |
| `data/gold/gold_queries_review.csv` and `scripts/10_agreement.py` (referenced by `docs/annotation_guidelines.md`) | Referenced files do not exist; the actual review tool is `scripts/18_review_tools.py`. |
| `data/review/answer_key_DO_NOT_SHARE.csv` | Gitignored and absent; `scripts/18_review_tools.py score` cannot run without it. |
| Human labels | No agreement or kappa can be computed. |
| Whisper checkpoint | Voice ASR not re-run; aggregates verified from CSV only. |
| Full encoder comparison | `encoder_comparison.json` is explicitly partial (`"partial": true`); fine-tuning absent. |
| Exact Kaggle checksum/version | Dataset provenance is strong but not byte-verified. |

### 6.3 Environment caveat

On macOS, the combination of PyTorch and FAISS can segfault during hybrid retrieval due to duplicate OpenMP runtimes. `OMP_NUM_THREADS=1` fixes it. This is not documented in the README and should be added, because a reviewer following the README on macOS may see a crash.

### 6.4 Working-tree state

`git status` shows four modified tracked files and an untracked `.venv312/`:

```
 M results/calibration.json
 M results/hybrid_retrieval.json
 M results/model_selection.json
 M scripts/13_hybrid_retrieval.py
?? .venv312/
```

The modifications are small numeric drifts from a re-run (e.g., `combo_lr_C10_w1 silver 0.9355 → 0.9375`; hybrid α=0.4 novel 0.2446 → 0.2454) plus a batch-size/progress-bar change in script 13. This means the committed results and the working-tree results are not identical. The audit used the working tree and notes where the committed values differ. For a submission, commit the regenerated results and document the run, or revert to the committed artefacts and report those.

---

## 7. Answers to the specific authenticity questions

| Question | Answer |
|---|---|
| Where exactly did the original KCC dataset come from? | The data.gov.in KCC transcript catalogue (“Kisan Call Centre – Transcripts of farmers queries and answers”), via the Kaggle mirror by Das Koushik. The exact local download path/date is not recorded. |
| Who published or collected it? | Ministry of Agriculture and Farmers Welfare, Government of India (catalogue); mirrored publicly by Das Koushik on Kaggle. |
| Is it publicly available? | Yes, via data.gov.in / IndiaAI and the Kaggle mirror. The data.gov.in page was not directly fetchable during the audit (HTTP 403), but the catalogue metadata is available through IndiaAI. |
| Original URL? | <https://www.data.gov.in/resource/kisan-call-centre-kcc-transcripts-farmers-queries-answers> and <https://www.kaggle.com/datasets/daskoushik/farmers-call-query-data-qa>. |
| Is it genuinely derived from KCC records? | Strong evidence yes: exact row match, KCC-specific content (Assam districts, AAU, KVKs, local advisors), catalogue description. Not checksum-verified. |
| Are the farmer queries real? | They are real **logged summaries** written by advisors, not verbatim transcripts or audio. |
| Are the advisor responses real? | Yes, they are real logged advisor text, but 8.75% are call-log notes and some are stale, terse or contain PII. |
| Are intent labels manually annotated or LLM-generated? | **LLM-generated**, by one assistant, following written guidelines. No human verification. |
| Are training examples synthetically generated? | The 3,000 training texts are real corpus queries with LLM labels; 1,200 additional rows are LLM paraphrases. |
| Were paraphrasing or augmentation techniques used? | Yes: 400 training queries × 3 registers = 1,200 spoken-style paraphrases, all LLM-generated. |
| Is the evaluation/gold dataset independently created? | Independent of the corpus and training data (0 exact overlap, 0 pairs ≥0.90), but **not** independent of the LLM annotator. |
| Does the project use synthetic speech or actual farmer recordings? | **Synthetic speech only** (Windows SAPI TTS), 50 English queries. No farmer recordings. |
| Are there signs of modification, filtering, generation or fabrication? | Derivation/filtering yes (two-column mirror, cleaning, splits); synthetic labels/gold/paraphrases/speech yes; fabricated rows no evidence. |
| Can provenance be independently verified from the repository? | **No.** The repository contains no URL, checksum, date or licence. External matching establishes a strong chain, but not byte-level certainty. |

---

## 8. Identified issues and recommendations

### 8.1 Critical (must fix before submission/deployment)

1. **Do not claim human verification.** Replace “only partly human-verified” with “not human-verified”; report κ only after the 250-row sheet is completed by at least two annotators.
2. **Do not claim the corpus is PII-free.** Either redact PII or state explicitly that raw answers contain likely personal data and that no redaction was performed.
3. **Correct the corpus-size denominator.** 138,875 is the cleaned corpus; 124,987 is the train+validation retrieval index.
4. **Add a leakage-free evaluation.** Re-split by question identity (grouped split) and report intent and retrieval metrics on that split alongside the existing numbers.
5. **Document the macOS OpenMP crash and the `OMP_NUM_THREADS=1` workaround.**
6. **Restore or remove the legacy DistilBERT claims.** If `models/intent_classifier/` cannot be shipped, mark 99.5%/65.7% as historical, not reproducible, and do not present them as current evidence.

### 8.2 High priority

7. **Complete the human annotation study** with two annotators, report human–human and human–LLM Cohen’s kappa, and adjudicate disagreements.
8. **Collect real Hindi/Hinglish telephone speech** and report ASR WER under noise; the current 1.46% is an upper bound on clean TTS.
9. **Obtain agronomist review** of a stratified sample of auto-answers; report correctness, not just relevance.
10. **Make the safety metric independent.** Have a human judge whether an auto-answer is hazardous, rather than reusing the regex that filters it.
11. **Add privacy engineering**: PII detection/redaction, encryption at rest, retention policy, access control.
12. **Add authentication and role separation** before any multi-user deployment.

### 8.3 Medium priority

13. **Regenerate paper assets** (`scripts/19_make_paper_assets.py`) so call-log share, voice metrics and gate numbers match the current JSON files.
14. **Remove or correct stale Haryana claims** in `docs/KCC-Sahayak_20_Papers_Gap_Analysis.md`.
15. **Commit the regenerated result files** and record the exact environment/seed so the working tree and the report agree.
16. **Run the full encoder comparison**, including fine-tuned MuRIL and IndicBERTv2, or state clearly that only frozen encoders were compared.
17. **Replace fixed thresholds with a calibrated policy** learned from labelled risk data, or at least sweep and report sensitivity.
18. **Add a model/result version stamp** to the Evaluation dashboard so it cannot display metrics from a model that is no longer loaded.

### 8.4 Low priority

19. Add a `curl`-based smoke test for the demo endpoints.
20. Add a `requirements-lock.txt` or pinned versions for exact reproducibility.
21. Document the synthetic-audio generation procedure and commit the TTS voice names.
22. Add a data dictionary for `annotated.csv`, `pool.csv`, `augmented.csv` and `gold_queries.csv`.

---

## 9. Bottom line for a viva or evaluation

You can defend the following claims with evidence:

- A real, working voice/text triage prototype with a FastAPI backend, SQLite queue, advisor console and dashboards.
- 93.75% intent accuracy on 496 held-out corpus queries (95% CI 91.7–95.8), reproduced from the shipped model.
- 90.0% accuracy on an independent 150-query English/Hinglish/Hindi gold set (85.3–94.7), reproduced.
- Spoken-style training text is decisive: without it, Hinglish gold accuracy falls from 90% to 30%.
- The full safety gate auto-answers 27.0% of silver queries at 95.5% accuracy with 2.24% policy violations; thresholds alone auto-answer 92.3% with 71.0% violations.
- Hybrid retrieval improves answer F1 from 0.4548 to 0.4679 overall and from 0.1866 to 0.2454 on novel queries.
- The corpus has 67.51% keyword-label validity, 70.73% near-duplicate test queries, 8.75% call-log-note answers, and 17/138,875 Haryana mentions.

You should **not** claim:

- Human-verified labels, human agreement, or agronomist-validated correctness.
- Real farmer speech or telephone-grade ASR.
- Freedom from PII.
- State-specific (Haryana) results.
- Production readiness, security or privacy compliance.
- That the deployed model beats a character n-gram SVM; it is statistically tied once spoken-style training text exists.
- That “policy violation 2.24%” means real-world safety; it means compliance with a project-defined rule.

---

## 10. Post-audit improvement status (Batch A – Improvement 4), 4 November 2026

This section records work completed **after** the audit above. It is additive:
the audited code paths, datasets, models and original result files are
unchanged, and no original finding is retracted. Everything is classified by
evidence type; no result in this section is human-validated.

### 10.1 Status at a glance

| Workstream | Status | Location | Evidence class |
|---|---|---|---|
| Batch A — grouped, leakage-resistant evaluation | Complete | `experiments/grouped_evaluation/`, `results/improved_evaluation/` | Corpus evaluation with LLM labels; relevance is a weak-label topical proxy |
| Batch B Phase 1 — PII detection/redaction | Complete (synthetic evaluation) | `experiments/privacy/`, `results/privacy_audit/` | Synthetic, same-author test set |
| Batch B Phase 2 — human annotation infrastructure | Infrastructure complete; **annotation PENDING** | `experiments/annotation/`, `results/annotation/` | No human labels exist yet |
| Improvement 1 — answer conflict detection | Complete (synthetic pairs + descriptive corpus scan) | `experiments/conflict_detection/`, `results/conflict_detection/` | Synthetic authored pairs; rule-fire rates |
| Improvement 2 — adversarial safety evaluation | Complete (synthetic adversarial set) | `experiments/safety_evaluation/`, `results/safety_evaluation/` | Synthetic authored cases; production rules unmodified |
| Improvement 3 — multi-reference retrieval evaluation | Complete (corpus evaluation) | `experiments/multi_reference/`, `results/multi_reference/` | Historical advisor answers as references; agreement, not correctness |
| Improvement 4 — out-of-scope detection | Complete (synthetic negatives) | `experiments/out_of_scope/`, `results/out_of_scope/` | Authored synthetic non-agricultural queries; real agricultural test rows |
| Improvement 5 — behavioural regression testing | Complete (authored synthetic probes) | `experiments/behavioural_regression/`, `results/behavioural_regression/` | 168 deterministic synthetic probes over deployed rule-based components; no model loading |

### 10.2 Verified test counts

| Suite | File(s) | Tests |
|---|---|---:|
| Original project | `tests/test_api.py` (10), `test_gate.py` (22), `test_text_utils.py` (9) | **41** |
| Batch A | `tests/unit/test_grouped_evaluation.py` | 11 |
| Pre-Batch-B total | — | **52** |
| Batch B Phase 1 | `tests/unit/test_privacy.py` | 36 |
| Batch B Phase 2 | `tests/unit/test_annotation.py` | 30 |
| Improvement 1 | `tests/unit/test_conflict_detection.py` | 30 |
| Improvement 2 | `tests/unit/test_adversarial_safety.py` | 13 |
| Improvement 3 | `tests/unit/test_multi_reference.py` | 27 |
| Improvement 4 | `tests/unit/test_out_of_scope.py` | 25 |
| Improvement 5 | `tests/unit/test_behavioural_regression.py` | 11 |
| **Current full suite** | `pytest -q` with `OMP_NUM_THREADS=1` | **224 passed** |

**Correction:** intermediate Batch B summaries described “52 original
tests”. That label was wrong: 52 was the pre-Batch-B total (41 original +
11 Batch A). The original suite has always been 41 tests. The mislabel did not
affect any metric or pass/fail result; it is corrected here and in the
Batch B experiment log.

### 10.3 Batch A — grouped evaluation (corpus, LLM labels)

Grouped corpus split (exact-normalised-question + MiniLM cosine ≥ 0.95,
transitive union-find; `StratifiedGroupKFold`): **125,044** index rows,
**13,831** test rows. Leakage after grouping: **0.09%** of test queries have a
≥0.95 neighbour in the index (69.94% ≥0.90; 6.15% below 0.80).

Intent refit (MiniLM + char n-gram TF-IDF → LR), five seeds:

| Metric | Mean ± SD |
|---|---|
| Accuracy | **0.9372 ± 0.0045** |
| Macro F1 | **0.9205 ± 0.0048** |
| Weighted F1 | 0.9358 ± 0.0047 |
| Accuracy without paraphrases | 0.9372 ± 0.0054 |

Novelty buckets (seed 0 counts): near-duplicate 78 → 0.9630 ± 0.0219;
similar 236 → 0.9580 ± 0.0122; novel 186 → **0.9025 ± 0.0107**.

Retrieval (dense, all 13,831 test rows): topic match@1 **0.8519** (95% CI
[0.8462, 0.8576]), Recall@5 0.9508, MRR@10 0.8961, nDCG@5 0.9071, answer
token-F1 0.3213. On the identical 3,000-query sample: dense 0.8403 vs hybrid
(α=0.4) **0.8733** topic match@1; answer F1 0.3220 vs 0.3800. Novel-query topic
match@1 0.7812.

Caveat unchanged from the audit: labels are LLM-generated and relevance is a
weak-label topical proxy, so these numbers describe retrieval of same-topic
logged answers, **not** agronomic correctness. The original row-level reference
(93.55% silver; 70.73% near-duplicate share) is not directly comparable.

### 10.4 Batch B Phase 1 — privacy (synthetic)

87 hand-authored synthetic examples (47 English, 21 Hindi, 19 Hinglish;
70 gold entities; 31 negative/ambiguous), strict exact span+type matching:

| Policy | TP | FP | FN | Precision | Recall | F1 | F1 95% CI |
|---|---:|---:|---:|---:|---:|---:|---|
| balanced | 65 | 1 | 5 | 0.9848 | 0.9286 | **0.9559** | [0.9077, 0.9882] |
| strict | 63 | 1 | 7 | 0.9844 | 0.9000 | 0.9403 | [0.8868, 0.9793] |

Document-level false-positive rate is 0.0 on 31 negative documents. These are
synthetic, same-author results; **no corpus prevalence or recall claim is made**,
and redaction is not integrated into the production gate or backend.

### 10.5 Batch B Phase 2 — annotation infrastructure (PENDING)

The workflow exists and is tested (300-item stratified English sample, two
independent blinded annotators, Cohen/Fleiss kappa, confusion matrices,
disagreement and adjudication tracking, QC, offline UI). **No human has labelled
any item**: `results/annotation/agreement.json` and `qc.json` both report
`status: PENDING`, all pair tables are empty and no adjudication exists.
Therefore:

- human–human kappa: **not computed** (no data);
- human–LLM kappa: **not computed** (no human labels);
- expert verification and adjudicated ground truth: **do not exist**.

The 300 items and their candidate answers are real corpus text, but the labels
are LLM-produced and were used only for stratification. They must not be called
human gold labels.

### 10.6 Improvement 1 — answer conflict detection (synthetic + descriptive)

Deterministic rules over quantities/units/bases/products/intervals/negation;
the module never merges answers and only escalates. Authored synthetic pair set
(**77 pairs**: 52 conflict, 25 no-conflict; 67 core, 10 challenge), bootstrap
2,000 resamples, seed 0:

| TP | FP | FN | TN | Precision | Recall | F1 | F1 95% CI |
|---:|---:|---:|---:|---:|---:|---:|---|
| 47 | 0 | 5 | 25 | 1.0000 | 0.9038 | **0.9495** | [0.8989, 0.9901] |

Core subset F1 1.0000 (n=67); challenge subset F1 0.6154, recall 0.4444 (n=10).
Five challenge misses are reported, not tuned away (brand vs active ingredient,
implicit per-acre vs per-bigha, indirect negation, interval stated once).

Descriptive corpus scan (Batch A grouped metadata, 1,000 sampled multi-answer
groups): 4,918 pairs compared, 1,258 pairs (25.58%) and 382 groups (38.20%)
flagged. These are **rule-fire rates**, not confirmed errors; no expert
adjudication exists.

### 10.7 Improvement 2 — adversarial safety evaluation (synthetic)

Production `src/confidence_gate.py` imported **read-only** (SHA-256 recorded in
`metrics.json`; `modified_for_this_evaluation: false`). 149 authored cases:
81 unsafe, 58 safe, 10 ambiguous (excluded from P/R/F1).

| View | TP | FP | FN | TN | Precision | Recall | F1 | F1 95% CI | Accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|---|---:|
| Content rule (`answer_risk`) | 63 | 23 | 18 | 35 | 0.7326 | 0.7778 | **0.7545** | [0.6790, 0.8208] | 0.7050 |
| Gate, content-only slice | 63 | 24 | 18 | 34 | 0.7241 | 0.7778 | 0.7500 | [0.6750, 0.8182] | 0.6978 |

The four figures requested for verification are **confirmed against
`results/safety_evaluation/metrics.json`**: precision **0.7326**, recall
**0.7778**, F1 **0.7545**, missed unsafe cases **18 of 81** (false-safe rate
0.2222). The gate-slice F1 is **0.7500**, a different measurement on the same
cases; the two must not be quoted interchangeably.

Subsets: core (58) recall 1.0000, F1 0.8136; challenge (81) recall 0.6842, F1
0.7222. Languages: English recall 0.8889, Hinglish 0.9375, **Hindi 0.0000
(0/11)**. False escalations 23 of 58 safe cases (0.3966). Severity of the 18
missed unsafe cases: 13 high, 5 medium.

Documented bypasses (not fixed, because the rules were deliberately not tuned
to this test set): Devanagari units/product names, letter-spaced or
hyphenated/digit-substituted brand names, number words without a Latin unit
token, and non-standard measures (`bottle`, `tank`, `bori`). Because the set is
synthetic and same-author, these scores are **not** real-world safety estimates,
and detecting a pattern is not evidence of agronomic correctness.

### 10.8 Improvement 3 — multi-reference retrieval evaluation (corpus)

Scores a retrieved answer against every distinct answer advisors logged for the
same question group (Batch A grouping, test side only) instead of only the
single answer attached to the row. No reference is generated: pools are
historical answers. Production retrieval is untouched.

**Pool adequacy (measured against pre-registered criteria):** 13,831 grouped
test rows in 4,850 groups; **11,747** distinct informative references in 4,457
groups. 96.6% of test rows have ≥1 reference, **69.9% have ≥2**, 58.2% have ≥5;
469 rows (3.4%) have none and are excluded. The distribution is skewed — 2.64
references per group (median 1) but 175 per row (median 10), with 21.9% of rows
inside groups holding ≥100 references. References are genuinely different
advice rather than restatements: mean max pairwise token-F1 within a group is
0.6201, and 64.2% of sampled multi-reference groups fall below 0.80. Verdict:
**multi-reference evaluation is supported**, with the skew disclosed.

**Batch A reproduction gate (7/7 PASS):** the rebuilt rankings reproduce the
frozen Batch A values before any new metric is reported — dense all-test
own-reference F1 0.321342 vs 0.3213, informative-gold 0.329323 vs 0.3293, topic
match 0.851927 vs 0.8519; 3,000-sample dense 0.321968/0.840333 vs 0.3220/0.8403;
hybrid 0.379990/0.873333 vs 0.3800/0.8733. The gate caught two implementation
bugs during development (sample-answer misalignment and a sparse-matrix length
error) before any result was reported.

**Dense, all 13,831 test rows** (bootstrap 2,000 resamples, seed 0):

| Metric | Rows | Mean |
|---|---:|---:|
| Own-reference F1 (original metric) | 13,831 | 0.3217 |
| Max-reference F1 | 13,831 | 0.5096 |
| Max-other-reference F1 | 9,669 | 0.5788 |
| Random-other-reference F1 (chance control) | 9,669 | 0.2948 |
| Max semantic similarity (not correctness) | 9,669 | 0.7755 |

Paired deltas: max-other − own **+0.2415** (95% CI [0.2362, 0.2465], n=9,768,
80.9% improved, 12.3% worse, d_z 0.91); restricted to informative own answers
**+0.2453** ([0.2402, 0.2501], n=9,029, d_z 1.00); max − own +0.1867
([0.1827, 0.1907], n=13,346). The **chance control is negative**: random-other
− own −0.0393 ([−0.0431, −0.0353]), so the gain is not merely the effect of
having more chances to match, although part of it is mechanical.

**Dense vs hybrid, identical 3,000 queries:** own-reference 0.3222 vs 0.3802
(+0.0581, CI [0.0480, 0.0676], d_z 0.21); max-other (informative own answers)
+0.0244 (CI [0.0120, 0.0370], d_z 0.086). With informative-only screening the
same ordering holds (+0.0532 own, +0.0182 max-other). Hybrid's advantage
**shrinks but does not vanish** when the reference set is widened — it was not an
artefact of one reference answer.

Single-reference “misses” recovered by another advisor answer: of 7,212 rows
with own F1 < 0.5, **3,773 (52.3%)** matched some other reference at ≥0.5.
Subsets: exactly one reference n=3,693 own 0.2954; ≥20 references n=5,721 own
0.3263 / max-other 0.6641; novel bucket n=850 own 0.2330 / max-other 0.5603.

Interpretation limits, unchanged: these are **historical advisor answers, not
gold**. Contradictory advice can share a pool (Improvement 1 shows it does), so
matching any of it is agreement, not agronomic correctness; semantic similarity
is wording closeness; one grouped split seed only; no human relevance labels
exist (Phase 2 PENDING). A known artefact is documented rather than hidden:
1,208 rows (8.7%) have a call-log note as their own “answer”, so the original
metric scores against a non-answer there; those rows are excluded from the
informative-own comparison and reported separately. Full report:
`results/multi_reference/REPORT.md`; raw metrics: `metrics.json`; per-query
scores: `per_query_dense_all.csv`, `per_query_paired_sample.csv`. A rerun after
the module rename reproduced `metrics.json` (excluding its timestamp) and both
CSVs byte-for-byte.

### 10.9 Improvement 4 — out-of-scope detection (synthetic negatives)

Standalone comparison of two rejection strategies over the existing MiniLM
encoder. The production intent classifier, retriever and safety gate are
untouched. In-scope rows are real logged agricultural questions from Batch A's
grouped test split; out-of-scope rows are **180 authored synthetic queries**
(60 question families × English/Hindi/Hinglish, 12 non-agricultural categories).
Families are the split unit (36 train / 12 val / 12 test), so no translation or
paraphrase of a test question is visible during training or threshold selection.

| Approach | Threshold | Precision | Recall | F1 | F1 95% CI | AUROC | False accept (out-of-scope answered) | False reject (agricultural rejected) |
|---|---:|---:|---:|---:|---|---:|---:|---:|
| A. kNN distance (k=5, grouped index) | 0.7670 | 0.8537 | 0.9722 | **0.9091** | [0.8315, 0.9706] | 0.9873 | 0.0278 (1/36) | 0.0417 (6/144) |
| B. MiniLM + logistic regression | 0.8093 | 0.8780 | 1.0000 | **0.9351** | [0.8710, 0.9851] | 0.9927 | 0.0000 (0/36) | 0.0347 (5/144) |

Benchmark: 144 agricultural grouped-test queries (seeded sample) + 36 synthetic
negatives. Thresholds were selected on **validation only** (5,000 agricultural +
36 synthetic) under a fixed policy — maximise out-of-scope recall subject to
≤2% false rejection — giving validation FRR 1.98% for both. The benchmark
sample is optimistic for the false-rejection quantity: on the **full** 13,831-row
agricultural test set it is **6.25%** (kNN, 864 rows) and **3.63%** (LR, 502
rows). Both figures are reported; neither is a deployment estimate.

Synthetic negatives by register: English 12/12 rejected by both detectors, Hindi
12/12 by both, Hinglish kNN 11/12 and LR 12/12. This says nothing about false
rejections for Hindi/Hinglish, because no in-scope Hindi/Hinglish evaluation
data exists — the corpus is English by the project's register detector.

Descriptive sets (excluded from threshold selection and from the benchmark):
16 ambiguous queries → kNN gives 11 `ambiguous`, 4 `out_of_scope`, 1 `in_scope`
(75.0% not accepted); LR gives 7 `ambiguous`, 3 `out_of_scope`, 6 `in_scope`
(56.3% not accepted). 16 mixed-intent queries → kNN 15 `ambiguous`, 1
`in_scope`; LR 10 `ambiguous`, **6 `in_scope`**, so the LR detector can accept a
query that also contains an unrelated request. Neither detector implements
multi-intent handling (Improvement 5), and neither is wired into production.

### 10.10 Metric classification and comparability

| Evidence class | Where it appears |
|---|---|
| **Human-validated / expert-reviewed** | **None.** No human labels, no kappa, no agronomist review, no adjudicated ground truth. |
| **Corpus evaluation with LLM labels** | Original 93.75% silver / 90.0% gold; Batch A grouped intent 0.9372 and grouped retrieval scores. |
| **Synthetic evaluation** | Phase 1 PII metrics; Improvement 1 pair metrics; Improvement 2 adversarial metrics; Improvement 4 out-of-scope metrics (synthetic negatives); original synthetic speech WER. |
| **Descriptive corpus statistics** | Improvement 1 rule-fire rates; original leakage and call-log-note shares. |
| **Pending** | Human annotation, human–human/human–LLM agreement, expert safety review, real Hindi/Hinglish speech, PII prevalence on the corpus. |

The F1 values in §10 are **not comparable with each other**: 0.9559 (PII entity
span detection), 0.9495 (pairwise conflict detection on authored pairs), 0.8136
and 0.7222 (conflict detection subsets) 0.7545 (safety-rule detection) and 0.9351/0.9091 (out-of-scope detection on
authored synthetic negatives) score different tasks on different,
non-overlapping samples. Averaging or combining
them would be meaningless.

### 10.11 Integrity and protection verification

- Full suite: **224 passed** (`OMP_NUM_THREADS=1`, macOS); 25 of them are the
  Improvement 4 focused tests (`pytest tests/unit/test_out_of_scope.py -q`).
- Improvement 5 focused tests: **11 passed** (`pytest tests/unit/test_behavioural_regression.py -q`).
- Improvement 4 snapshot (`batch_b_improvement4_hashes.json`): **379 files** —
  220 original, 50 Batch A, 14 Phase 1, 24 Phase 2, 17 Improvement 1,
  13 Improvement 2, 22 Improvement 3, **18 Improvement 4**, 1 inventory config.
  `--check batch_b_improvement4_hashes` → **PASS** (0 changed, 0 missing,
  0 additions).
- Improvement 5 snapshot (`batch_b_improvement5_hashes.json`): **389 files** —
  all prior groups plus **10 Improvement 5** (4 experiment modules, 1 runner,
  4 results files, 1 focused test). `--check batch_b_improvement5_hashes` →
  **PASS** (0 changed, 0 missing, 0 additions).
- Improvement 5 rerun end-to-end: `metrics.json`, `cases.csv`, `failures.json`
  and `REPORT.md` are **byte-identical** across two runs; the case digest and
  metrics agree (`case_sha256` stable).
- **Older snapshots now report exactly one changed protected file: this audit.**
  The audit is itself a hash-recorded original document and Improvements 3–5
  were required to update it in place. Measured counts: `baseline_hashes.json`
  (270 files) → 1 changed (audit); 308-file snapshot → 1 changed (audit) + 16
  approved additions from Improvements 1–2; 325-file snapshot → 1 changed
  (audit); 339-file snapshot → audit and the protection config changed;
  361-file Improvement 3 snapshot → config changed + the Improvement 4 test
  added; 379-file Improvement 4 snapshot → audit changed + the Improvement 5
  test added. No production source, model, dataset or previous experiment output is
  among the differences.
- **Correction:** the earlier statement here that the 270-file baseline showed
  “0 changed” became inaccurate the moment the audit was edited for Improvement
  3. It is replaced by the measured counts above; the only changed protected file
  is the audit itself, and nothing was silently fixed.
- Pre-existing uncommitted state preserved: `results/calibration.json`,
  `results/hybrid_retrieval.json`, `results/model_selection.json`,
  `scripts/13_hybrid_retrieval.py` (26 insertions / 26 deletions total) and the
  untracked `.venv312/`.
- No production source (`src/`, `backend/`), model, dataset, Batch A output,
  Phase 1/2 output or Improvement 1/2/3 output was modified by Improvement 4.
  The two detectors are standalone and are not imported by `src/` or `backend/`.

### 10.12 Issues discovered during this consistency check

1. **Test-count label (documentation only).** Intermediate summaries called 52
   tests "original"; the correct split is 41 original + 11 Batch A = 52 before
   Batch B, and 224 now (§10.2). No test was skipped or weakened.
2. **Stale “41 tests” row in §3.2.** It was accurate at audit time but no longer
   describes `pytest -q` now that `tests/unit/` exists; annotated rather than
   rewritten, and the current count is reported in §10.2.
3. **Safety F1 has two values.** Rule-level 0.7545 and gate-slice 0.7500 are
   both current and correct for different views; they are labelled explicitly.
   A single number cannot stand for both.
4. **No other numeric drift found.** Phase 1, Phase 2, Improvement 1 and
   Improvement 2 log tables match their saved JSON artefacts; the Batch A
   reports match `grouped_intent.json` and `grouped_retrieval.json`.
5. **Unchanged limitations.** Grouped retrieval used one split seed; Phase 1 has
   no corpus scan; Improvement 1 has no expert adjudication; Improvement 2 is
   synthetic-only; the annotation study remains PENDING.
6. **Improvement 3 module-name collision (fixed).** The first version put
   `metrics.py`/`report.py`/`ranking.py` on `sys.path`; under the full suite
   these shadowed other experiments' modules in `sys.modules` and produced 12
   failures that did not appear when the file ran alone. The modules were renamed
   to `mref_*`; the full suite now passes 224 tests. No assertion was weakened.
7. **Improvement 3 pool-size statistic (corrected before publication).** The
   first report printed one row mixing a row-weighted mean (175) with a per-row
   median (10). Per-row and per-group distributions plus the ≥100-reference skew
   are now reported separately.
8. **Stale protection claim (documentation only, now corrected).** §10.10
   previously reported the 270-file baseline as “0 changed”. That was true only
   before the audit itself was edited for Improvement 3; the check now reports
   the audit as the single changed file. The measured counts and the reason are
   recorded in §10.11 rather than hidden.
9. **Improvement 4 scope of evidence.** Its negatives are author-written, not
   real traffic, and the LR detector accepts 6 of 16 mixed-intent queries. No
   human scope labels and no Hindi/Hinglish in-scope data exist, so the reported
   F1 values describe synthetic-negative detection only.

### 10.13 Remaining research gaps

- Complete the 300-query human annotation (two independent annotators), report
  human–human and human–LLM kappa, adjudicate disagreements; extend to Hindi and
  Hinglish items.
- Independent agronomist review of a stratified sample of auto-answers and
  flagged conflicts, reporting harm severity rather than similarity.
- Real Hindi/Hinglish telephone speech, with WER/CER under noise and downstream
  intent/retrieval/safety effects.
- PII prevalence measured on the real corpus (not only the synthetic set) and
  redaction wired into indexing/display behind an access-controlled design.
- Multi-seed retrieval evaluation (multi-reference answer scoring is now done
  for one grouped split, without human relevance labels).
- Multi-intent detection and missing-context clarification — **Improvement 5
  confirms multi-intent detection is absent**: 17 of 18 authored multi-intent
  probes fail with `single_label_only`, and no production component was modified.
  Out-of-scope detection exists only as a standalone experiment on synthetic
  negatives and is not wired into production.
- Real (not authored) non-agricultural traffic with human scope labels, to
  estimate prevalence and detector precision outside the synthetic set.
- Learned risk classification compared with the regex backstop, with
  coverage–risk curves and validated thresholds.

**Bottom line after Batch A/B:** the system is better evaluated (grouped split,
multiple seeds, conflict and safety stress tests) but the evidence base is still
synthetic or LLM-labelled. The strongest defensible claim remains the
grouped-split intent and retrieval performance on real logged text, with the
explicit caveat that labels are LLM-generated. No claim of human verification,
expert agronomic correctness or real-world safety is supported.

**Addition after Improvement 3:** the single-reference answer metric understates
retrieval usefulness on this corpus. On rows where another advisor answer to the
same question exists, max-other-reference F1 exceeds own-reference F1 by +0.24
(95% CI [0.24, 0.25]) while the random-reference control is below zero, and the
dense-vs-hybrid gap shrinks but survives the wider reference set. This measures
agreement with historical advice, not correctness, and no human relevance or
expert agronomic label exists.

**Addition after Improvement 4:** both out-of-scope detectors separate authored
synthetic non-agricultural queries from real agricultural test queries with high
F1 (kNN 0.9091, LR 0.9351) and low false-accept rates (2.8% and 0%). That is
synthetic-negative performance only: on the full 13,831-row agricultural test
set the false-rejection rates are 6.25% and 3.63%, the negatives are
author-written with no human scope labels, no Hindi/Hinglish in-scope evaluation
data exists, and neither detector is integrated into production.

**Addition after Improvement 5:** a 168-case authored behavioural probe suite
(`experiments/behavioural_regression/`, `results/behavioural_regression/`) runs
the deployed rule-based components (register detection, glossary NER, weak-label
intent, safety gate, conflict detection, PII detection/redaction, out-of-scope
decision, ASR transcript normalisation, data cleaning, API schema) read-only and
records expected-versus-observed behaviour. It is deterministic, synthetic and
explicitly not a production safety or accuracy estimate. Measured: 109/168 passed
(64.88%), 59 failed, 0 crashes. Failures: 6 degraded, 24 false negatives, 10
false positives, 18 missing capability, 1 validation gap. Multi-intent detection
remains absent: 17 of 18 multi-intent cases fail with `single_label_only`.
No production component was modified.

---

## 11. Audit artefact index

| Artefact | Purpose |
|---|---|
| `KCC_Sahayak_Research_Report.pdf` | 65-page academic report with all chapters, tables, figures, references and appendix |
| `KCC_Sahayak_Research_Report.docx` | Editable Word version with automatic TOC field |
| `KCC_Sahayak_Presentation.pptx` | 16-slide summary deck |
| `KCC_Sahayak_Diagrams/*.png` | Rendered architecture, workflow, model, retrieval, gate, provenance, Gantt and results figures |
| `KCC_Sahayak_Diagrams/*.mmd` | Editable Mermaid sources |
| `KCC_Sahayak_Diagrams/make_diagrams.py` | Reproducible diagram generator |
| `KCC_Sahayak_Verification_Audit.md` | This document |
