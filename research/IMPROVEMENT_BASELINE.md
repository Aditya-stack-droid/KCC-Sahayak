# KCC-Sahayak — Improvement Baseline and Repository Audit (Phase 0)

**Status:** audit complete; **Batch A (grouped, leakage-resistant evaluation) implemented and measured**; Batches B–G pending. Measured results and commands: `research/EXPERIMENT_LOG.md`.
**Date:** 3 November 2026
**Branch:** `improvements` (3 commits: `cf6aab3`, `effb1f1`, `ec2b5bc`)
**Scope of this document:** what exists, what was verified, what is missing, what must not change, and the proposed improvement plan.
**Rules honoured:** no file was modified, deleted, overwritten or moved to produce this audit. No dataset, model artefact, result file or research claim was altered. No improvement code has been written yet.

Every number below is either (a) read from an existing repository file, (b) recomputed independently in this audit, or (c) explicitly marked **UNVERIFIED / REPORTED**. Where the prior audit (`KCC_Sahayak_Verification_Audit.md`) and this audit disagree, both values are shown.

---

## 0. Audit method and environment

### 0.1 Method

1. Listed the repository, read `README.md`, `FINAL_REPORT.md`, `KCC_Sahayak_Verification_Audit.md`, `docs/annotation_guidelines.md`, `docs/SYNOPSIS_UPDATE.md`, `docs/VIVA_GUIDE.md`.
2. Read every module in `src/` (10 files), `backend/app.py`, `backend/db.py`, and the principal scripts (`01`, `03`, `04`, `06`, `07`, `10`, `13`, `14`, `16`, `17`, `18`, `22`).
3. Inspected dataset schemas and row counts directly from `data/`, `models/retrieval_index/`, `results/`.
4. Re-ran the unit/API test suite.
5. Recomputed the headline corpus/leakage/PII statistics independently with fresh Python code (not by re-reading the prior audit's numbers).
6. Loaded the deployed model artefact and FAISS index to verify the *actual* deployed configuration rather than trusting the README.
7. Cross-checked every material numeric claim in `paper/generated/*.tex` and `README.md` against `results/*.json`.

### 0.2 Environment (verified)

| Item | Value |
|---|---|
| OS / hardware | macOS, Apple M3, **8 GB RAM** |
| Free disk | **18 GiB** (228 GiB volume, 92% used) — constrains dataset/model downloads |
| Python | 3.12.13 (`.venv312/`); a second `.venv/` with Python 3.14 also exists |
| numpy / pandas | 2.5.4 / 3.1.0 |
| scikit-learn / scipy | 1.9.1 / 1.18.1 |
| torch / transformers | 2.15.0 / 5.22.1 |
| sentence-transformers | 6.2.0 |
| faiss | 1.15.1 (CPU) |
| onnxruntime | 1.31.1 (already installed) |
| Missing | `rank_bm25`, `presidio_analyzer`, `statsmodels`, `datasets` |
| Test suite | **41 passed in 7.04 s** (`OMP_NUM_THREADS=1 pytest tests -q`) |

### 0.3 Working-tree state (must be preserved, not "cleaned")

`git status` already shows pre-existing modifications and untracked files that predate this audit:

```
 M results/calibration.json
 M results/hybrid_retrieval.json
 M results/model_selection.json
 M scripts/13_hybrid_retrieval.py
?? .venv/  .venv312/  _report_build/  KCC_Sahayak_Research_Report.docx ...
```

The committed and working-tree result files are **not identical** (small numeric drift from a re-run). Any new work must not "fix" this by overwriting files; it is recorded here as a reproducibility finding.

---

## 1. Existing project architecture

### 1.1 Runtime flow (as implemented, not as advertised)

```
text or audio
  -> [audio only] src/asr.py: Whisper-small + agricultural decoder prompt
       (re-decodes as Hindi if Whisper labels the clip Urdu)
  -> src/lang.py: register heuristic (hi / hinglish / en)
  -> paraphrase-multilingual-MiniLM-L12-v2 (384-d, L2-normalised)   [one encoder]
       |-> src/intent_model.py: [embedding || char_wb 2-5 TF-IDF] -> LogisticRegression
       |                         -> temperature scaling (T=0.75)
       `-> src/retrieval.py: HybridRetriever
             dense: FAISS IndexFlatIP over 124,987 logged questions
             sparse: word TF-IDF (1-2 gram), fused with alpha=0.4
             + in-process "advisor memory" of resolved answers
  -> src/ner.py: gazetteer crop / pest-symptom extraction
  -> src/confidence_gate.py: 6-rule policy -> AUTO_ANSWER | ESCALATE
       (high-stakes type, time-sensitive type, intent confidence,
        retrieval similarity, topic match, answer screening)
  -> backend/app.py: log to SQLite, queue escalations, advisor resolves,
       resolved answer optionally added to retrieval memory
  -> frontend/index.html: demo, advisor console, ops, evaluation, system pages
```

### 1.2 Components and sizes

| Area | Files | Role |
|---|---|---|
| Inference | `src/pipeline.py` (216 L), `intent_model.py`, `retrieval.py`, `confidence_gate.py`, `answer_quality.py`, `ner.py`, `lang.py`, `asr.py` | end-to-end pipeline and its parts |
| Labelling | `src/weak_labels.py`, `src/glossary.py` | keyword weak labels, entity/category glossaries |
| API / persistence | `backend/app.py` (273 L), `backend/db.py` (182 L) | FastAPI, one SQLite table `queries`, advisor queue |
| UI | `frontend/index.html` + `assets/` | HTML/CSS/JS demo (Chart.js) |
| Offline work | `scripts/01`–`22` (4,319 L total incl. src/backend) | prepare, train, index, evaluate, annotate, paper assets |
| Tests | `tests/test_gate.py`, `test_api.py`, `test_text_utils.py` | 41 tests, stub pipeline, no model download |
| Research | `paper/main.tex` (198 L), `paper/generated/`, `results/`, report PDF/DOCX/PPTX, diagrams | claims and assets |

### 1.3 Verified deployed configuration (`models/`)

Loaded directly with `joblib` / `faiss`:

| Item | Verified value |
|---|---|
| `models/intent_embed_lr/model.joblib` | keys: `classes, emb_weight, embed_model, lr, temperature, tfidf` |
| Intent model | LogisticRegression, **C=10**, `n_features_in_ = 18,404` (384 dense + 18,020 char n-grams) |
| Char features | `TfidfVectorizer(analyzer="char_wb", ngram_range=(2,5), min_df=2, sublinear_tf=True)` |
| Embedding weight | 1.0 |
| Calibration | temperature **0.75** |
| Classes (10) | Animal Husbandry, Cultural Practices, Fisheries, Government Scheme, Market Price, Nutrient Management, Other, Plant Protection, Varietal Information, Weather |
| `models/retrieval_index/kcc_index.faiss` | **IndexFlatIP**, `ntotal=124,987`, `d=384` |
| Retrieval metadata | 124,987 rows (`qid, questions, answers, query_type, crops, symptoms`) |
| Hybrid fusion | `alpha=0.4` (`hybrid.json`) |
| Informative-answer mask | **114,051 / 124,987** kept (91.25%); 10,936 (8.75%) excluded as call-log notes |

The README's description of the deployed head is therefore accurate. The fallback path in `pipeline.py` points at `models/intent_classifier/`, which **does not exist**, so if `model.joblib` were missing the pipeline would raise rather than fall back.

### 1.4 Backend privacy/security surface (verified by grep)

`backend/app.py` contains **no** `CORSMiddleware`, authentication dependency, API-key check, JWT/OAuth, rate limiter or `Authorization` handling. Routes are open: `/api/health`, `/model-info`, `/samples`, `/query`, `/transcribe`, `/queue`, `/queue/{id}/resolve`, `/queries/{id}/feedback`, `/queries/recent`, `/ops/summary`, `/stats`. `db.py` has no encryption, retention policy, role separation, audit table or access control. All query text, candidate answers and advisor answers are stored in plaintext SQLite at `data/app.db`.

---

## 2. Existing models and algorithms

| Component | Implementation | Notes / limitations |
|---|---|---|
| Speech | `openai-whisper` `small`, optional agricultural vocabulary prompt; Urdu→Hindi re-decode; light transcript normalisation | CPU; no VAD, no diarisation, no confidence, no alternative ASR backend |
| Language ID | `src/lang.py` heuristics: Devanagari ratio, Hinglish function-word and farm-noun counts | Display/analytics only; not used by the gate |
| Intent | MiniLM embedding + char n-gram TF-IDF → multinomial LR; temperature scaling fitted on 5-fold out-of-fold logits | Single-label, forced choice among 10 classes; no "out of scope" option |
| Retrieval | FAISS `IndexFlatIP` dense cosine + word TF-IDF cosine, `fused = 0.4*dense + 0.6*tfidf`; call-log notes removed; optional topic re-ranking | Exact (not approximate) index; one reference answer per query; no reranker; no BM25; no conflict logic |
| Answer screening | `src/answer_quality.py`: typo-tolerant note/referral detection + generic-reference phrases | Rule-based English vocabulary; difflib matching |
| Entity extraction | Glossary longest-match for crops and pests/symptoms | No learned NER, no location/quantity/date extraction |
| Gate | Ordered rules in `src/confidence_gate.py`: high-stakes type → time-sensitive type → intent ≥0.70 → similarity ≥0.55 → topic match → informative answer → no dosage/chemical text | First rule wins; no structured multi-reason log; dosage/chemical regex is English/unit-centric |
| Weak labels | Keyword glossary over 9 categories + `Other` | Superseded by LLM labels but still used for corpus statistics and the historical baseline |
| Advisor memory | In-process numpy array of embeddings + list of Q/A; rebuilt from `status='resolved'` rows at startup | No review, expiry, provenance or conflict check before entering memory |

---

## 3. Existing datasets and artefacts

### 3.1 Verified counts

| Dataset / artefact | Path | Verified count | Status |
|---|---|---:|---|
| Raw KCC Q/A | `data/raw/kcc_mapping_full.csv` | **139,542 rows**, 87,292 unique question strings, **14,969 duplicate groups**, 52,249 extra rows in those groups, 74 null answers | real advisor-log text; no URL/licence/checksum in repo |
| Cleaned corpus | `data/processed/full_labeled.csv` | **138,875 rows** | derived; keyword labels + gazetteer entities |
| Splits | `train/val/test.csv` | **111,100 / 13,887 / 13,888** | random **row-level** split, seed 42 |
| LLM meaning labels | `data/annotation/annotated.csv` | **3,496** (3,000 train, 496 test) | one LLM annotator; 0 human verification |
| LLM paraphrases | `data/annotation/augmented.csv` | **1,200** (400 sources × 3 registers) | synthetic; shares author with gold set |
| Gold set | `data/gold/gold_queries.csv` | **150** (50 EN / 50 Hinglish / 50 Hindi; 15 per class) | LLM-written; 0 exact overlap with raw corpus (verified) |
| Human review sheet | `data/review/review_sheet.csv` | **250 rows, 0 completed labels** | study not performed |
| Retrieval index | `models/retrieval_index/` | 124,987 × 384 | derived from train+val only |
| Synthetic speech | `demo/voice_eval/*.wav` | 50 English TTS clips | synthetic, not farmer audio |

Label distribution (annotated set, verified): Plant Protection 790, Cultural Practices 540, Nutrient Management 458, Animal Husbandry 441, Government Scheme 336, Fisheries 274, Varietal Information 264, Other 179, Weather 151, Market Price 63.

### 3.2 Provenance and licence status

- The prior audit traced the corpus to the data.gov.in "Kisan Call Centre — Transcripts of farmers queries and answers" catalogue, mirrored on Kaggle as "Farmers Call Query (KCC) Data" (Das Koushik). The chain is strongly evidenced (row match) but **not checksum-verified**, and the repository itself records no source URL, download date or licence file.
- The government catalogue is described as Open Government Licence – India. **This audit did not independently confirm the licence text or the mirror's terms.** Any redistribution decision is therefore **PENDING legal/owner confirmation**.
- All labels, gold queries, paraphrases and speech are **synthetic supervision**, not human or expert ground truth. The repository's own README/paper say so; the paper still contains two claims that contradict this (§5.3).

### 3.3 PII-like content in raw answers (independently recomputed)

Regex scan over the 139,468 non-null raw answers:

| Pattern | Rows matching | Occurrences |
|---|---:|---:|
| Indian mobile-like 10-digit (`[6-9]\d{9}`) | **1,379** | 1,598 |
| Email-like | **87** | 103 |
| URL (`http://`, `https://`, `www.`) | **72** | 81 |
| "aadhaar"/"aadhar" word | **637** | 695 |
| 12-digit Aadhaar-like string | **6** | 6 |
| IFSC-like (`^[A-Z]{4}0[A-Z0-9]{6}$`) | **0** | 0 |

**Discrepancy with the prior audit:** it reports 663 email-bearing rows and 1,016 Aadhaar mentions; this audit finds 87 and 637. Different regexes and possibly different columns were scanned. Neither count is human-adjudicated, and mobile-like matches include likely false positives (IDs, quantities). **Conclusion:** PII presence is established; exact volume is **not** established and must be resolved by a labelled adjudication sample in Phase 3. Redaction is therefore justified regardless of the exact count.

---

## 4. Existing evaluation methodology

| Script | What it measures | Split / data | Known issue |
|---|---|---|---|
| `04_evaluate.py` | intent accuracy, retrieval type-match, risk–coverage sweep | 3,000 random rows of `test.csv` | row-level leakage; uses **keyword** labels |
| `07_evaluate_gold_and_baselines.py` | majority, keyword rules, TF-IDF+SVC, DistilBERT on weak-label test and gold; TF-IDF vs MiniLM retrieval | 3,000 test rows + 150 gold | labels are LLM; DistilBERT artefact absent; single reference answer |
| `10_train_embedding_classifier.py` | MiniLM-LR, TF-IDF-SVC, old DistilBERT, keyword rules; saves a head and calibration | 3,000 train / 496 silver / 150 gold | saves a **different, incompatible joblib schema** (no TF-IDF block) than `16` |
| `13_hybrid_retrieval.py` | word-FIDF build, alpha sweep, answer token-F1 | 2,737 informative-answer test rows | one reference answer; no Recall@k/MRR/nDCG |
| `14_selective_and_screening_eval.py` | coverage/selective accuracy, gate ablation, answer screening | silver 496 + gold 150 + 3,000 test | "violation" is defined by the same regex that filters |
| `16_model_selection.py` | candidate comparison, 5-seed Hinglish re-spelling, temperature scaling, `--save` | train+aug / silver / gold | selection score uses benchmark labels; no grouped split |
| `17_research_eval.py` | CIs, McNemar, ablations, learning curve, novelty buckets, corpus audit | same sets | reuses the leaky split; no multiplicity correction |
| `18_review_tools.py` | 250-row blind sheet, human–LLM and human–human kappa | 150 gold + 100 silver | **never run**; answer key gitignored/absent; no adjudication workflow |
| `22_retrieval_baselines.py` | KisanQRS-style / lexical / dense / hybrid baselines | 3,000 test rows | **output file absent** — script not run in the shipped tree |

**Leakage mechanism (verified in `scripts/01_prepare_data.py`):** `train_test_split` splits **rows**, not question identities, and the corpus contains 14,969 duplicate question strings. Therefore identical questions appear in train and test with different answers.

**Independently recomputed leakage (this audit):**

| Measure | Value |
|---|---:|
| Test question strings exactly present in train | **44.98%** |
| Test queries with dense similarity ≥ 0.95 to the index | **70.73%** |
| ≥ 0.90 | **83.53%** |
| < 0.80 (novel) | **4.93%** |

These confirm the prior audit's figures. Note that the similarity distribution is computed from the saved `results/test_predictions.csv`, where the index is train+val; it is a valid leakage measure for that split but is not a grouped-split estimate.

**Statistical practice today:** bootstrap 95% CIs on accuracy (2,000 resamples), exact McNemar tests, five seeds only for the Hinglish spelling perturbation, fixed seeds elsewhere. No paired bootstrap, effect sizes, Holm correction, or multiple-seed model comparison. No grouped split anywhere.

---

## 5. Existing reported results and their verification status

### 5.1 Results that reproduce / are stored consistently

| Result | Stored value (file) | Status |
|---|---|---|
| Intent accuracy, silver n=496 | **0.9355**, CI 0.9153–0.9556 (`research_eval.json`) | stored; README says 93.5; prior audit refit says 93.75 → **drift** |
| Intent accuracy, gold n=150 | **0.9000**, CI 0.8533–0.9467 | stored; prior audit reproduced |
| Gold by register | EN 0.94 / Hinglish 0.90 / Hindi 0.86 | stored; consistent |
| Character TF-IDF + SVM | 0.9254 silver / 0.8733 gold | stored |
| McNemar vs production (char SVM) | p = 0.4049 silver / 0.4807 gold | stored; conclusion = statistically tied |
| Embedding-only LR | 0.9052 silver / 0.7400 gold; p = 0.0059 / <0.001 | stored |
| Word TF-IDF + SVM | 0.9133 silver / 0.8133 gold; p = 0.0708 / 0.0146 | stored |
| No spoken-style augmentation, gold | 0.6000 (Hinglish 0.30) | stored |
| Hinglish random re-spelling (5 seeds) | 0.856 mean | stored in `model_selection.json` |
| Gate @ defaults, silver | coverage **0.2641**, selective acc **0.9542**, violations **0.0229** | stored |
| Gate @ defaults, gold | coverage **0.2333**, selective acc **0.9429**, violations **0.0286** | stored |
| Thresholds only, silver | coverage **0.9254**, selective acc **0.9695**, violations **0.7168** | stored |
| Answer screening (3,000) | coverage 0.2923 → 0.1647; risky-text share 0.4276 → 0.0000 | stored |
| Hybrid retrieval (n=2,737, novel 143) | dense 0.4545 → hybrid 0.4679; novel 0.1866 → 0.2454 (α=0.4) | stored |
| Calibration | T=0.75; silver ECE 0.0177→0.0237; OOF 0.0381→0.0060 | stored |
| Keyword-label validity | 0.6751 agreement | stored |
| Call-log notes in index | 8.75% (10,936 / 124,987) | recomputed from mask |
| Novelty buckets (silver) | <0.80: 0.8771 (n=179); 0.80–0.90: 0.9535 (172); ≥0.90: 0.9862 (145) | stored |
| Synthetic speech | WER 0.0146, intent after ASR 0.94, n=50 English TTS | stored; not re-run |
| Unit/API tests | 41 passed | **verified this audit** |

### 5.2 Results that are missing or not reproducible

| Missing artefact | Consequence |
|---|---|
| `models/intent_classifier/` (legacy DistilBERT) | 99.5% keyword-label / 65.7% meaning-label DistilBERT numbers cannot be regenerated (65.7% is stored in `research_eval.json`; 99.5% only in old files) |
| `results/retrieval_baselines.json` | `scripts/22` exists but was never run in this tree; KisanQRS-style comparison is unsubstantiated |
| `results/cache/` | `17_research_eval.py` expects cached embeddings; absent, so it must re-encode |
| `data/review/answer_key_DO_NOT_SHARE.csv` | gitignored and absent; `18_review_tools.py score` cannot run |
| `results/human_agreement.json` | listed in `results/README.md` as trusted output but does not exist |
| Encoder fine-tuning | `encoder_comparison.json` is explicitly partial; MuRIL/IndicBERTv2 fine-tunes never run |
| Whisper checkpoint | voice evaluation not re-runnable without downloading the model |
| Kaggle file checksum/version | provenance strong but not byte-verified |

### 5.3 Claim inconsistencies discovered (for the Phase 10 claim checker)

These are **flag-only** findings. No document was edited.

| Claim | Location | Evidence | Issue |
|---|---|---|---|
| "only partly human-verified" | `paper/main.tex` | 0/250 review labels | **unsupported**; "partly" overstates zero |
| "No personal data about farmers is used; the corpus is public" | `paper/main.tex` | 1,379 mobile-like, 87 email, 637 Aadhaar-word rows | **unsupported**; public ≠ non-personal |
| Corpus "124,987 … after cleaning" | `paper/main.tex` | cleaned 138,875; index 124,987 | denominator conflation |
| Call-log share 8.2% | `paper/generated/numbers.tex` | 8.75% | stale generated asset |
| Voice metrics `n/a` | `paper/generated/numbers.tex` | `voice_eval.json` has WER 1.46%, acc 94% | stale |
| Hybrid n = 2,756 | `paper/generated/numbers.tex` | `hybrid_retrieval.json` n=2,737 | stale |
| Screening coverage 17.2%, risky 41% | `paper/generated/numbers.tex` | 16.47%, 42.76% | stale |
| Gate table 92.3/97.4/72.3; 26.0/96.1/2.3 | `paper/generated/table_gate.tex` | 92.54/96.95/71.68; 26.41/95.42/2.29 | stale; also omits the topic-match row |
| Gold thresholds-only 70.0/95.2/65.7 | `paper/generated/table_gate.tex` | 68.0/95.1/64.71 | stale |
| Silver 93.5% | README/paper | `research_eval.json` 93.55; `model_selection.json` 93.75; prior audit refit 93.75 | drift between evaluation paths |
| PII counts (663 email, 1,016 Aadhaar) | prior audit | this audit: 87, 637 | unresolved methodology difference |
| Retrieval baseline exists | docs/scripts | output file absent | missing evidence |
| `models/` rebuildable from scripts | README | `scripts/10 --save` writes an **embedding-only** joblib, while the deployed artefact is embedding+char (saved by `16 --save`) | ambiguous reproduction path; the README does not say `16 --save` is the one that produced the deployed head |

**Not fabricated:** the result files are internally consistent and the principal metrics reproduce within run-to-run drift. The problems are stale generated assets, denominator/label conflation and overstated provenance, not invented experiments.

---

## 6. Identified weaknesses (prioritised)

### Critical

1. **Severe train–test leakage.** Row-level split over 14,969 duplicate-question groups; 44.98% exact test-in-train overlap and 70.73% near-duplicates. Every random-row metric is an upper bound, not a generalisation estimate.
2. **Synthetic supervision presented near human ground truth.** 3,496 labels, 150 gold queries and 1,200 paraphrases come from one LLM; 0 human labels exist; gold and augmentation share an author (prior audit: 22% of gold queries have ≥0.90 similarity to augmentation — **reported, not re-verified here**).
3. **Circular safety metric.** `answer_risk`/`_DOSE` both filters answers and defines "policy violations". The 2.29% figure measures compliance with a project-defined regex, not harm. No agronomist review.
4. **Unredacted PII in data and logs.** Raw answers contain phone/email/Aadhaar-like content; retrieved text is displayed and stored unredacted; advisor answers and queries persist in plaintext SQLite. Paper claims the opposite.
5. **No authentication, authorisation, rate limiting, CORS policy, encryption or retention** in the backend; advisor and farmer roles are indistinguishable.

### High

6. **Retrieval evaluation is single-reference.** 14,969 duplicate-question groups mean one logged answer is not a unique correct target; token-F1 against it is noisy and biased. Only top-1 is scored; no Recall@k, MRR or nDCG.
7. **Speech evidence is synthetic English TTS only.** 1.46% WER on clean synthetic audio is an upper bound; Hindi/Hinglish telephone speech is untested.
8. **Closed-set classifier.** Forced 10-class prediction for any input (cricket scores → some agricultural class). No out-of-scope rejection, no multi-intent, no missing-context clarification.
9. **Safety rules are English/dosage-regex-centric.** No systematic test for Hindi/Hinglish instructions, unit variants, obfuscated product names, paraphrases or negation ("do not spray pesticide").
10. **Claims drift from artefacts** (§5.3) and `results/README.md` itself labels several files untrusted while the paper still derives numbers from old generated assets.
11. **Advisor memory has no review gate.** Any resolved answer enters retrieval immediately; no second-advisor approval, expiry, provenance or conflict check.
12. **Reproducibility gaps:** unpinned `requirements.txt`, no lockfile, no seed manifest, working-tree/committed drift, macOS OpenMP crash caveat undocumented (prior audit; not re-tested here), distinct incompatible save schemas in `10` vs `16`.

### Medium

13. No statistical multiplicity control, paired bootstrap, effect sizes or multiple-seed comparison for the main model table.
14. No retrieval reranker, BM25 baseline, HNSW/quantisation comparison, or latency benchmark.
15. No explainability output (top n-grams, dense/sparse contributions, rule trace).
16. No dataset datasheet, retention policy, or security log.
17. Documentation still carries stale Haryana framing in parts of `docs/` and `src/glossary.py`'s docstring (README/synopsis already correct this).
18. No CI, no claim-consistency check, no environment lock; `models/` gitignored so artefact provenance lives outside version control.

---

## 7. Files that must remain untouched (freeze list)

**Never edit, delete, move or overwrite:**

- `data/raw/**`, `data/processed/**`, `data/annotation/**`, `data/gold/gold_queries.csv`, `data/review/**`, `data/app.db`
- `models/**` (all current artefacts, including `models/retrieval_index/*` and `models/intent_embed_lr/model.joblib`)
- `results/**` (all existing JSON/CSV/PNG)
- `src/**`, `backend/**`, `frontend/**`, `scripts/**`
- `tests/test_api.py`, `tests/test_gate.py`, `tests/test_text_utils.py`
- `paper/**` (source, generated assets, PDF), `docs/**`
- `README.md`, `FINAL_REPORT.md`, `KCC_Sahayak_Verification_Audit.md`, `this_is_it.md`
- `KCC_Sahayak_Research_Report.pdf/.docx`, `KCC_Sahayak_Presentation.pptx`, `KCC_Sahayak_Diagrams/**`
- `requirements.txt`, `.gitignore`, `run_app.bat`

**Allowed additive locations:** `research/`, `experiments/**`, `tests/{unit,integration,behavioural,regression}/**` (new files only), `results/{improved_evaluation,improved_retrieval,improved_safety,privacy_audit,speech_evaluation,baseline_comparison,statistical_analysis}/**`.

**Integration rule (if any):** a new optional adapter may be added as a *new* module (e.g. `experiments/.../gate_v2.py` plus a thin `src/`-free wrapper) and wired into the backend only via a default-off configuration flag in a *new* file. Until explicitly approved, the backend and `src/` are not touched at all, so the existing behaviour and the 41 tests remain exactly as they are.

---

## 8. Improvement-to-file mapping (proposed)

| # | Improvement | New files (proposed) | Original preserved |
|---|---|---|---|
| 1 | Human annotation of a new 300/register gold set | `experiments/human_annotation/{make_sheets.py,score_agreement.py,adjudicate.py,README.md}`, `data/annotation_v2/` templates | `data/gold/gold_queries.csv`, `scripts/18_review_tools.py` untouched |
| 2 | Grouped, leakage-resistant evaluation + buckets | `experiments/grouped_evaluation/{build_groups.py,make_split.py,evaluate.py,README.md}` | `scripts/01`, `17` untouched; new split written outside `data/processed/` |
| 4 | Answer consolidation / conflicts | `experiments/answer_consolidation/{consolidate.py,conflict_rules.py,report.py}` | corpus and index untouched |
| 5 | Conflict-aware gate rule | `experiments/conflict_detection/{detector.py,tests}` | `src/confidence_gate.py` untouched; optional wrapper |
| 6 | Multi-reference retrieval metrics | `experiments/improved_retrieval/{metrics.py,evaluate.py,bm25.py,rerank.py}` | `results/hybrid_retrieval.json` untouched |
| 7 | PII detection/redaction | `experiments/pii_redaction/{recognizers.py,redact.py,audit.py,test_recognizers.py}` | raw data untouched; redacted copies only |
| 8 | Independent safety evaluation | `experiments/safety_evaluation/{build_stress_set.py,evaluate.py,taxonomy.py}` + human-review pack | `scripts/14` untouched |
| 9 | Adversarial regex stress tests | `experiments/safety_evaluation/regex_stress.py`, `tests/behavioural/` | `src/confidence_gate.py` untouched |
| 10 | Out-of-scope detection | `experiments/out_of_scope/{build_oos_set.py,knn_reject.py,evaluate.py}` | classifier untouched |
| 11 | Multi-intent classification | `experiments/multi_intent/{build_multilabel.py,train.py,evaluate.py,risk.py}` | single-label model untouched |
| 12 | Missing-context / clarification | `experiments/multi_intent/context.py` (or `experiments/clarification/`) | gate untouched |
| 13 | Behavioural pytest suite | `tests/behavioural/test_*.py`, `tests/unit/test_*.py` | existing tests untouched |
| 14 | ASR evaluation + telephone simulation | `experiments/speech_evaluation/{asr_backends.py,evaluate_asr.py,simulate_channel.py,downstream.py}` | `src/asr.py` untouched |
| 15 | Classifier baselines | `experiments/classifier_baselines/{run.py,models.py}` | deployed head untouched |
| 16 | Retrieval baselines + reranker | `experiments/retrieval_baselines/{run.py,metrics.py}` | index untouched |
| 17 | Optional local LLM-RAG | `experiments/llm_rag/{retrieve_then_generate.py,safety.py}` | retrieval pipeline untouched |
| 18 | Enhanced gate + decision logs | `experiments/safety_gate/{gate_v2.py,evaluate.py}` | original gate and tests untouched |
| 19 | Advisor review workflow | `experiments/advisor_review/{review_db.py,workflow.py,test_workflow.py}` | `data/app.db`, `backend/db.py` untouched |
| 20 | Privacy/security | `research/DATASET_DATASHEET.md`, `experiments/privacy_ops/{retention.py,audit_log.py}`, deployment notes | live DB untouched |
| 21 | Backend auth/RBAC/rate limiting | proposal + optional adapter + tests | `backend/app.py` untouched unless approved |
| 26 | Claim consistency checker | `experiments/statistical_analysis/claims_checker.py` (or `tools/`) | documents untouched; report only |
| 27 | Statistical framework | `experiments/statistical_analysis/{stats.py,run.py}` | existing results untouched |
| 28 | Latency benchmarks | `experiments/latency_benchmarks/{bench.py,REPORT.md}` | no code changes |
| 29 | ONNX/INT8/HNSW | `experiments/model_optimisation/{export_onnx.py,bench_index.py}` | original model/index untouched |
| 30 | Explainability | `experiments/explainability/{intent.py,retrieval.py,gate.py}` | modules untouched |
| 31–33 | Research framing, advisor-assist, datasheets | `research/{EXPERIMENT_LOG.md,DATASET_DATASHEET.md,LIMITATIONS.md,IMPROVEMENT_SUMMARY.md}` | paper untouched; new docs only |

All new outputs go to `results/improved_*`, `results/privacy_audit/`, etc. **No existing result file is overwritten.**

---

## 9. Dependency requirements (proposed, cheapest first)

| Purpose | Package | Size / cost | Notes |
|---|---|---|---|
| BM25 baseline | `rank_bm25` | tiny, pure Python | not installed |
| PII (optional second opinion) | `presidio-analyzer`, `presidio-anonymizer` | small; spaCy model optional | Presidio's Indian-identifier and Roman-Hindi coverage is limited → custom recognizers are required regardless |
| BERTScore (optional) | `bert-score` | pulls a transformer checkpoint; default `roberta-large` ≈1.4 GB | **avoid default** on 18 GB free disk; use a small multilingual checkpoint and document it, or use embedding cosine instead |
| Stats | `statsmodels` (optional) | small | McNemar/effect sizes can be implemented with `scipy` + bootstrap, no new dependency needed |
| Cross-encoder reranker | `sentence-transformers` (already present) + `cross-encoder/ms-marco-MiniLM-L-6-v2` | ≈90 MB | feasible for ≤3k queries on CPU; measure before claiming |
| Rate limiting (Phase 9) | `slowapi` | tiny | only if backend work is approved |
| Auth (Phase 9) | `python-jose[cryptography]`, `passlib[bcrypt]` | small | proposal + tests first |
| ASR alternative | `faster-whisper` | ≈500 MB model + CTranslate2 | needs licence/disk check; English TTS only today |
| Load testing | `locust` (optional) | small | optional |
| ONNX | `onnx`, `onnxruntime` (already installed) | small | export the existing head only, never replace it |
| Fuzzy matching | `rapidfuzz` (optional) | tiny | `difflib` works but is slower |

**No paid API will be introduced.** Any LLM-RAG experiment will use a locally available model only; if none is available, the experiment stays **PENDING** rather than calling a paid service.

---

## 10. Computational requirements and constraints

- **Hardware:** Apple M3, 8 GB unified RAM, CPU-only; no CUDA. Free disk 18 GiB.
- **Feasible:** MiniLM encoding, LR/SVM/SetFit on ≤5k rows, FAISS Flat/HNSW over 125k × 384, small cross-encoder reranking of a 3k-query eval set, Presidio, ONNX export of the LR head, faster-whisper small, latency micro-benchmarks.
- **Not feasible without external resources:** fine-tuning MuRIL-base / IndicBERTv2 on CPU within a reasonable time; downloading multi-GB LLMs; full Whisper-large evaluation; processing a large speech corpus.
- **Encoding reuse:** existing `models/retrieval_index/embeddings.npy` covers the 124,987 train+val rows. A grouped split that moves groups across splits will need embeddings for the remaining ~13.9k cleaned queries (or one full 138,875-query encode). This is a one-off CPU job expected in the tens of minutes; **runtime will be measured, not assumed**. Results will be cached under `results/improved_evaluation/cache/`.
- **macOS OpenMP:** prior audit reports a PyTorch+FAISS hybrid-search segfault unless `OMP_NUM_THREADS=1`. Not re-tested here. New scripts will set/require it and document it rather than alter `src/retrieval.py`.
- **Memory discipline:** batch size ≤128 for encoding, no simultaneous loading of multiple transformers, float32 embeddings.

---

## 11. External dataset requirements (all currently UNVERIFIED)

| Resource | Intended use | What must be checked before use |
|---|---|---|
| Digital Green agricultural QA / human-preference data | external evaluation / human preference | official source URL, licence, schema, language mix, whether real farmer queries, size |
| Farmer.Chat benchmark (real agricultural speech) | ASR + downstream evaluation | official release, licence, audio format, language distribution, transcript quality, download size, ethics |
| GramVaani / Vistaar | telephony speech | access terms, consent/ethics, language, size |
| MUCS 2021 | Indic speech | licence, domains, recording conditions |
| AI4Bharat (IndicSUPERB / Shrutilipi / IndicVoices) | Indic ASR baselines | per-dataset licence, size, agricultural relevance |
| KCC data.gov.in catalogue | provenance/licence confirmation | licence text, export version, checksum, district/month fields (absent from local mirror) |

**No external dataset has been downloaded.** Disk space (18 GiB) and licence terms must be checked first; any download needs explicit approval. None of these datasets may be described as "real farmer data" until its provenance is verified.

---

## 12. Human / expert involvement requirements (cannot be automated)

| Activity | Requirement | If unavailable |
|---|---|---|
| New gold set, 300 queries × 3 registers | 2 independent annotators, ≥900 queries each, blind to each other and to LLM labels | build sheets/scoring/adjudication code; mark **PENDING** |
| Human–human and human–LLM kappa | the same 2 annotators + adjudicator | infrastructure only; **no kappa reported** |
| Safety/agronomist review | qualified agricultural reviewers judge correctness/harm on a stratified sample | build review pack; **no correctness claim** |
| Real Hindi/Hinglish speech | consenting speakers / verified public corpus + ethics approval | synthetic telephone simulation only; label as synthetic |
| Advisor pilot | real advisors, time/correction measures | offline retrieval-usefulness measures only; **no time-saved claim** |

I will not create, simulate or infer human labels. Any file that would require them will be created empty/templated with an explicit `PENDING` status.

---

## 13. Risks of breaking existing functionality

| Risk | Mitigation |
|---|---|
| Editing `src/` changes pipeline behaviour and breaks 41 tests | zero edits to `src/`, `backend/`, `scripts/`; all new code in `experiments/` |
| New index overwrites `models/retrieval_index/` | new artefacts written only under `experiments/**/artifacts/`, never `models/` |
| New results overwrite trusted JSON | strict output-path allowlist (`results/improved_*`, etc.); assert-on-exists guard in scripts |
| Rebuilding TF-IDF/FAISS drifts from committed numbers | reuse existing artefacts read-only; compare against stored baselines; document drift |
| pandas 3 / numpy 2 API differences | use the existing venv; add `requirements-improvements.txt`, do not replace `requirements.txt` |
| Memory exhaustion with cross-encoders/large models | batched evaluation, small checkpoints, measure first |
| OpenMP segfault on macOS | `OMP_NUM_THREADS=1` in new scripts; do not patch `src` |
| `models/` gitignored → new artefacts untracked | write a regeneration manifest with hashes; never move existing artefacts |
| Backend adapter accidentally becomes required | default-off flag in a new file; existing app must run unchanged without it |
| Claim checker "fixing" prose | report-only, never rewrites; flags for human review |
| LLM-judge reported as ground truth | judge outputs always paired with a human-scored subset or marked unvalidated |
| Existing dirty working tree mistaken for our changes | record baseline hashes; only add files, do not stage/commit unless asked |

---

## 14. Phased implementation plan (proposed order)

Numbering follows the brief; note that items **22–25 are absent from the brief**, so the sequence jumps from 21 to 26.

### Batch A — Trustworthy evaluation (offline, no human data)
- Group exact/near-duplicate questions (MiniLM cosine ≥ 0.95 configurable), write group ids and statistics.
- Build group-aware split (StratifiedGroupKFold / GroupShuffleSplit), preserve class balance, report leakage before/after.
- Evaluation buckets: near-duplicate / similar / novel; accuracy, macro/weighted P/R/F1, confusion matrices, bootstrap CIs, ≥5 seeds where practical.
- Re-evaluate the **existing deployed artefact** on the grouped split (read-only) and compare with the old split.
- Outputs: `experiments/grouped_evaluation/`, `results/improved_evaluation/`.

### Batch B — Retrieval validity
- Answer consolidation: group by question identity, embed answers, cluster, detect numeric/unit/product disagreement, produce consensus/conflict/unresolved sets and a human-review report. Never auto-merge contradictory advice.
- Multi-reference metrics: max token-F1 across references, semantic similarity, Recall@k, MRR, nDCG; keep the original single-reference metric for comparison.
- Baselines: BM25, dense, hybrid, cross-encoder rerank, hybrid+rerank; latency measured.
- Outputs: `experiments/answer_consolidation/`, `experiments/retrieval_baselines/`, `results/improved_retrieval/`.

### Batch C — Privacy and independent safety
- PII recognizers (mobile, email, URL, Aadhaar-like, bank/IFSC, names where reliable) + adversarial tests + audit report; redacted copies only.
- Independent safety test set (synthetic, clearly labelled), severity taxonomy, regex stress tests in EN/Hinglish/Roman-Hindi, learned risk classifier (TF-IDF+LR, embedding+LR) vs regex, coverage-risk curves, threshold sensitivity.
- Optional enhanced gate wrapper (conflict, PII, OOS, multi-intent risk, missing context, learned risk) with structured decision logs; original gate preserved as fallback.
- Outputs: `experiments/pii_redaction/`, `experiments/safety_evaluation/`, `results/privacy_audit/`, `results/improved_safety/`.

### Batch D — Classifier robustness and baselines
- Out-of-scope set + kNN-distance / class-distance / extra-class approaches.
- Multi-label classifier + riskiest-intent rule; clarification/missing-context detector.
- Behavioural pytest suite (typos, Roman-Hindi, negation, units, multi-intent, OOS).
- Classifier baselines on the **same grouped split**: TF-IDF+LR, char-SVM, MiniLM+LR, fine-tuned MiniLM (small, feasible), SetFit; MuRIL/IndicBERTv2 only if resources permit (otherwise marked not run).
- Statistical framework: paired bootstrap, effect sizes, Holm, McNemar, seed reporting.
- Outputs: `experiments/out_of_scope/`, `experiments/multi_intent/`, `experiments/classifier_baselines/`, `experiments/statistical_analysis/`, `results/baseline_comparison/`, `results/statistical_analysis/`, `tests/behavioural/`.

### Batch E — Annotation, advisor workflow, documentation
- Two-annotator blind sheets, kappa, disagreement report, confusion matrices, adjudication; 300/register target; status **PENDING** until real humans participate.
- Advisor review workflow (`PENDING_REVIEW` → second advisor → safety checks → indexed; expiry; audit log) in a **separate** SQLite DB; original workflow untouched.
- Datasheets, limitations, experiment log, claim-consistency checker (report-only), CI hook where practical.
- Outputs: `experiments/human_annotation/`, `experiments/advisor_review/`, `research/*.md`, `results/improved_evaluation/annotation/`.

### Batch F — Speech, security, performance, explainability
- ASR pluggable interface; evaluate existing Whisper-small vs faster-whisper vs an Indic model **if** audio/licence permit; telephone simulation (8 kHz, noise/SNR); downstream intent/retrieval/gate impact. Otherwise infrastructure + **PENDING**.
- Privacy/security: retention config, safe deletion, audit log, access-control design; auth/RBAC/rate-limit/CORS proposal with default-off adapter and tests; SQLite encryption compatibility study.
- Latency benchmarking, ONNX/INT8 export of the head, Flat vs HNSW comparison (recall/latency/memory), explainability module.
- Outputs: `experiments/speech_evaluation/`, `experiments/privacy_ops/`, `experiments/latency_benchmarks/`, `experiments/model_optimisation/`, `experiments/explainability/`, `results/speech_evaluation/`.

### Batch G — Research framing
- Let the results decide the central claim. The current evidence already shows the deployed model is **statistically tied** with a char n-gram SVM, so the paper will not claim superiority.
- Candidate framing: agricultural query triage, safety-aware retrieval, privacy-aware assistance, answer-conflict detection, dataset auditing, human-in-the-loop decision support, cross-lingual robustness.
- Advisor-assist evaluation design only; real pilot **PENDING**.
- Update `EXPERIMENT_LOG.md`, `LIMITATIONS.md`, `DATASET_DATASHEET.md`, `IMPROVEMENT_SUMMARY.md`; never rewrite the original paper.

---

## 15. Improvements that can be completed fully offline

No humans, no external data, no paid services:

1. Grouped split, leakage audit, novelty buckets, bootstrap CIs, multi-seed evaluation of the **existing** model.
2. Answer consolidation, conflict clusters, numeric/unit/product disagreement reports (human review of the report is pending, but the artefacts are produced).
3. Multi-reference retrieval metrics + BM25/cross-encoder/hybrid baselines + latency.
4. PII recognizers + adversarial tests + audit report + redacted derived copies.
5. Synthetic safety stress set, regex stress tests, learned risk classifier, coverage-risk curves, enhanced gate wrapper and decision logs.
6. Out-of-scope detection, multi-intent classifier, clarification detector, behavioural test suite.
7. Classifier baselines on the grouped split (small/frozen encoders; fine-tuned MiniLM feasible on this hardware).
8. Statistical analysis, claim checker, datasheets, experiment log, improved-gate comparison.
9. ONNX/INT8 export of the LR head, Flat vs HNSW benchmark, explainability outputs, offline latency micro-benchmarks.
10. Advisor-review workflow code + tests on a **separate** database.

**Explicitly PENDING (cannot be completed by me):** human annotations and kappa, expert/agronomist safety review, real farmer speech, ethics approval, advisor pilot, licence confirmation for external datasets, and any hardware-dependent fine-tune (MuRIL/IndicBERTv2) that exceeds this machine.

---

## 16. Audit gaps in this document

Recorded so they are not mistaken for verified facts:

- The prior audit's claims about a macOS OpenMP segfault, 22% gold↔augmentation similarity, 663 email rows, 1,016 Aadhaar mentions, and the 93.75% silver refit were **not** reproduced here; they are marked as reported.
- Hybrid retrieval, Whisper, and a full model refit were **not** executed in this audit; only stored artefacts and the deployed head/index were loaded.
- File-level licence/consent status of the raw corpus is **unconfirmed**.
- `results/test_predictions.csv` is generated from the keyword-labelled legacy evaluation; its `retrieval_similarity` column is nonetheless a valid dense-similarity measure for the leakage calculation.

---

## 17. Approval gate

Batch A was approved and implemented on 3 November 2026 (see
`research/EXPERIMENT_LOG.md`); no original file was modified. The remaining
batches require approval of:

1. the freeze list in §7,
2. the proposed output layout and default-off integration policy,
3. the batch order in §14 (A → G),
4. the dependency additions in §9,
5. the explicit PENDING status of all human/expert/external-data work.

On approval, implementation starts with **Batch A** and follows the per-stage protocol: inspect → implement in a separate location → unit tests → run → save raw outputs → report → compare to baseline → document limitations → update `EXPERIMENT_LOG.md`.
