# KCC-Sahayak — Final Comprehensive Project Audit (v2)

**Repository:** `KCC-Sahayak`, branch `improvements`, HEAD `293103c` ("Add Batch B research
and safety improvements"), remote `origin` = `Aditya-stack-droid/KCC-Sahayak`.
**Audit date:** 4 November 2026.
**Relationship to earlier work:** this is a **new** document created on the user's request;
it does **not** overwrite `KCC_Sahayak_Verification_Audit.md`, which remains the audit of
record for the original system and Batch A/B. This file extends that audit with a final,
independent, project-wide pass and one correction (§7).

**Scope rule observed:** no production source (`src/`, `backend/`), model, dataset,
previous experiment output or pre-existing uncommitted change was modified. Only the two
new report files were written.

---

## 1. Repository inventory

**Git state (measured, not assumed):**

- Branch `improvements`; HEAD `293103c`; `main` is `cf6aab3`.
- 8 modified tracked files, 20 untracked paths (see §6).
- Tracked files: **361**.

**Tracked files by extension:** 85 `.py`, 51 `.wav`, 43 `.csv`, 40 `.txt`, 40 `.json`,
34 `.png`, 28 `.md`, 7 `.tex`, 6 `.mmd`, 5 `.pdf`, 4 `.jpg`, 3 `.html`, 2 `.jsonl`,
1 `.pptx`, 1 `.npz`, 1 `.js`, 1 `.joblib`, 1 `.css`, 1 `.bat`, 1 `.ps1`.

**Python inventory:** **110 `.py` files** (excluding `.venv/`, `.venv312/`, `.git/`):

- `src/` — 10: `pipeline`, `retrieval`, `intent_model`, `confidence_gate`,
  `answer_quality`, `ner`, `lang`, `asr`, `glossary`, `weak_labels`.
- `backend/` — 2: `app.py`, `db.py`.
- `scripts/` — 23 numbered scripts (`01`–`22`, `00` is PowerShell).
- `tests/` — 11 files (`test_api`, `test_gate`, `test_text_utils` + 8 under `tests/unit/`).
- `experiments/` — grouped evaluation (Batch A), privacy + annotation (Batch B), conflict
  detection, safety evaluation, multi-reference, out-of-scope, behavioural regression,
  and the `batch_b` verification tooling.
- `KCC_Sahayak_Diagrams/`, `_report_build/` — asset generators (untracked).

**Local (gitignored) artefacts present but not committed:** `models/retrieval_index/`
(FAISS index, embeddings, TF-IDF, metadata, informative mask, `hybrid.json`) and
`models/intent_embed_lr/model.joblib`. `models/` is `.gitignore`d, so a fresh clone lacks
them and must rebuild via scripts 03/16/13.

`models/intent_classifier/` (legacy DistilBERT) is **absent**; its historical 99.5%/65.7%
figures cannot be regenerated from this snapshot.

---

## 2. Files inspected

**Read in full or in part during this audit (~40 files):**

- Core: all 10 `src/` modules; `backend/app.py`, `backend/db.py`.
- Tests/tooling: `experiments/behavioural_regression/{br_cases,br_probes,run_evaluation}.py`;
  `experiments/batch_b/{common,verify_phase_outputs}.py`.
- Result artefacts: `research_eval.json`, `selective_eval.json`, `hybrid_retrieval.json`,
  `model_selection.json`, `calibration.json`, `answer_screening.json`,
  `annotation_summary.json`, `corpus_geography.json`, `voice_eval.json`,
  `improved_evaluation/{grouped_intent,grouped_retrieval}.json`,
  `conflict_detection/metrics.json`, `annotation/agreement.json`,
  `behavioural_regression/{metrics,REPORT}.md`, headline keys of `privacy_audit/metrics.json`,
  `safety_evaluation/metrics.json`, `multi_reference/metrics.json`,
  `out_of_scope/metrics.json`.
- Docs: `README.md`, `FINAL_REPORT.md`, `KCC_Sahayak_Verification_Audit.md`, `this_is_it.md`.
- Front end: `frontend/assets/js/app.js` (helpers and dynamic-render paths).
- Config: `requirements.txt`, `run_app.bat`, `experiments/batch_b/inventories/protection.json`,
  protection manifests.

**Machine-checked across the whole tree:** `compileall` over `src`, `backend`, `scripts`,
`experiments`, `tests`, `KCC_Sahayak_Diagrams`, `_report_build`; grep scans for absolute
paths, secrets, `eval`/`exec`/`pickle`/`yaml.load`/`subprocess`/`shell=True`/bare `except`,
environment-variable use, and temporary-file handling.

---

## 3. Code-quality findings

- **Syntax/imports:** `python -m compileall` succeeds for every module (exit 0). No syntax
  errors. `tests` import cleanly (224 collected).
- **Hardcoded local paths:** none in tracked source. The only `/Users/...` string is inside
  `.venv312/bin/vba_extract.py` (third-party). Scripts use `Path(__file__).resolve()` roots.
- **Risky constructs:** no bare `except:`; no `eval`/`exec`; no `pickle.load` on untrusted
  data (`joblib` is used only for locally produced artefacts); the only `subprocess` call is
  in `scripts/15_voice_eval.py` (Windows PowerShell SAPI, constant command, no user input);
  no `shell=True`.
- **Environment variables:** only `scripts/15` reads `LOCALAPPDATA`/`PATH` (Windows-only).
- **Temporary files:** `backend/app.py` writes the audio upload to a `NamedTemporaryFile`
  and `unlink`s it in a `finally` block — no leak on the normal or error path.
- **Unicode:** SQLite writes use `json.dumps(..., ensure_ascii=False)`; `__main__` probes
  reconfigure stdout to UTF-8. Devanagari handling is correct in `lang`/`asr`.
- **Exception handling:** the API catches broad exceptions only at the audio boundary
  (→ HTTP 422) and via a global handler that returns the exception **type** only (no stack
  trace).
- **Small defects found** (see §8): a whitespace-only API validation gap and an
  attribute-context escaping gap in the front end.

---

## 4. Pipeline integrity findings

Traced the source path Input → ASR → register → intent → retrieval → NER → gate →
answer/escalation, and **executed the real pipeline** (not a stub) with the local models:

| Query | Register | Type | Conf. | Sim. | Decision |
|---|---|---|---:|---:|---|
| "which tomato variety is good for me?" | en | Varietal Information | 1.00 | 0.845 | AUTO_ANSWER (`ok`) |
| "gehu mein peela rust aa gaya hai kya spray karun" | hinglish | Plant Protection | 0.97 | 0.236 | ESCALATE (`high_stakes`) |
| "आज मंडी में सरसों का भाव क्या चल रहा है?" | hi | Market Price | 1.00 | 0.661 | ESCALATE (`time_sensitive`) |
| "control measure for aphid infestation in mustard" | en | Plant Protection | 1.00 | 0.956 | ESCALATE (`high_stakes`) |

These match the documented demo behaviour. Backend reported the expected intent backend
(MiniLM + char n-gram TF-IDF → LR, T=0.75), hybrid retrieval, and a 124,987-row corpus.

**Stage reasoning:**

- **Input/output compatibility:** the pipeline always returns at least one retrieved
  candidate (empty retrieval falls back to a zero-similarity placeholder), so
  `db.insert_query` reading `retrieved_top3[0]["similarity"]` is safe.
- **Confidence/threshold handling:** the gate compares the **calibrated** intent
  probability and the **dense** cosine (not the fused score), so the 0.55 threshold keeps a
  single meaning across languages. Thresholds default to 0.70/0.55.
- **Staleness:** Weather/Market Price escalate unconditionally before any confidence check.
- **Call-log filtering:** `informative_mask` filters candidates at query time; the complete
  FAISS/TF-IDF index still contains every row (documented wording nuance).
- **Dosage safety:** `answer_risk` screens retrieved text for unit+pesticide/fertiliser
  language even for non-high-stakes query types.
- **Contradictory answers / out-of-scope / multi-intent:** these are **standalone
  experiments** (Improvements 1/4/5), **not wired into** `src/pipeline.py` or the gate.
  Source evidence: `backend/app.py` imports only `confidence_gate` and `pipeline`; the gate
  does not import the conflict or out-of-scope modules. Multi-intent handling is absent
  (Improvement 5 confirms 17/18 probes fail with `single_label_only`).
- **Register/NER:** `detect_register` is display/analytics only; the NER gazetteer is
  English-only and feeds the UI/DB, not the gate.

**Distinction:** the Safety gate, intent head and hybrid retriever are production-integrated
(verified by execution). Conflict detection, out-of-scope detection, PII redaction and
multi-intent detection are experimental-only.

---

## 5. Security and privacy findings

- **Secrets/credentials:** none found in source, config, tests, or front end.
- **PII:** the raw corpus contains regex/keyword matches consistent with personal data
  (≈1,379 phone-like, 663 email, 72 URL, 1,016 Aadhaar-mention rows in answers). No
  redaction is applied to the corpus or to live logs; `experiments/privacy/` implements
  detection/redaction but is **not integrated** into the backend or gate.
- **Sensitive logging:** SQLite stores full query text, suggested answer and advisor answer
  unencrypted, with no retention/deletion controls.
- **API security:** no authentication or authorization; any reachable client can query,
  resolve queue items and post feedback. No rate limiting, CORS or TLS; error responses
  leak only the exception type.
- **Unsafe file handling / path traversal:** the audio upload suffix is taken from the
  client filename but is used only as a temp-file suffix under the system temp dir (not as a
  path), and the temp file is deleted in `finally`. No traversal path was found.
- **Injection:** all dynamic front-end text is escaped with `esc()` (text-content context).
  One **low-severity** gap: `data-use="${esc(c.answer)}"` places the escaped value inside a
  quoted HTML **attribute**, and `esc()` (built on `textContent`→`innerHTML`) does **not**
  escape `"`. A retrieved/advisor answer containing a double quote could break out of the
  attribute (see §8, D2). No server-side injection (stdlib `sqlite3` is parametrised).
- **Deserialisation:** no untrusted deserialisation; `joblib`/`np.load`/`faiss.read_index`
  load only repository-local artefacts.
- **No destructive security testing was performed.**

---

## 6. Model and dataset findings

- **Model-loading paths:** `models/intent_embed_lr/model.joblib`,
  `models/retrieval_index/{kcc_index.faiss,embeddings.npy,corpus_metadata.csv,tfidf.joblib,
  tfidf_matrix.npz,informative_mask.npy,hybrid.json}`. All present locally; all gitignored.
- **Embedding dimension/tokenisation:** one MiniLM encoder feeds both intent and retrieval;
  dense vectors are L2-normalised and searched with `IndexFlatIP` (cosine). Consistent with
  the documented design.
- **Train/inference preprocessing consistency:** both use the same encoder and the same
  char_wb 2–5 TF-IDF vectoriser for intent; retrieval TF-IDF is word 1–2 gram — the
  documented design split.
- **Label/class mappings:** 10 classes from `lr.classes_`; the production head stores its
  own classes with the artefact.
- **Calibration:** T=0.75; working-tree `calibration.json` records silver ECE 0.0177→0.0237
  and out-of-fold 0.0381→0.0060 (committed: 0.0232 and 0.0051).
- **Retrieval index consistency:** index rows 124,987 = train+val; test excluded (no index
  leakage).
- **Dataset schemas:** raw is two-column text; processed adds `qid`, `query_type`, `crop`,
  `symptom`; annotation adds `label`/`pool`; gold adds `style`. `qid` is a sequential row
  index, **not** a stable identity — correctly flagged in the existing guide.
- **Missing values / duplicates:** 1 empty raw question (explains 87,292 vs 87,293 unique
  questions); zero empty questions after cleaning; zero exact duplicate pairs; 14,969
  duplicate-question groups in raw.
- **Train/test separation:** **row-level, not question-level** → 44.98% exact-question
  overlap; Batch A's grouped split reduces near-duplicate leakage to 0.09%.
- **Reproducibility:** deterministic scripts with fixed seeds; four uncommitted numeric
  drifts document residual run-to-run sensitivity (unpinned dependencies).

---

## 7. Correction to the earlier audit (§10.11 of the original)

The original audit states the Improvement 5 protection snapshot
(`batch_b_improvement5_hashes.json`, 389 files) verifies with **0 changed**. At audit time
that was true. It is now **stale**: the snapshot reports **1 changed protected file**,
`results/out_of_scope/metrics.json` (sha256 `ff6b9642…` → `8e8b7435…`, size unchanged at
11,987 bytes).

Cause: the file carries a `generated_utc` timestamp (`2026-11-04T12:36:06+00:00`) and was
regenerated **after** the snapshot was written (`improvement5_hashes.json` created
12:32:51 UTC; the file mtime is 18:06 local / 12:36 UTC). The size is identical and the only
non-deterministic field is the timestamp, so this is a **timestamp-only regeneration**, not
a substantive change to any metric. It is recorded here rather than hidden, and older
snapshots fail as expected (audit edited + approved new improvement files added).

---

## 8. Defects discovered

| ID | File / component | Problem | Severity | Type |
|---|---|---|---|---|
| D1 | `results/out_of_scope/metrics.json` vs `batch_b_improvement5_hashes.json` | Protected-file hash drift from a non-deterministic `generated_utc` timestamp | Low | Integrity |
| D2 | `backend/app.py` `QueryRequest` | Whitespace-only text of length ≥2 (e.g. two spaces) passes `min_length=2`, then `req.text.strip()` yields empty input that reaches the pipeline | Low | Validation gap (failures G26) |
| D3 | `frontend/assets/js/app.js` (line 265) | `esc()` is text-content-safe but `data-use="${esc(c.answer)}"` is an attribute context; `"` is not escaped → a crafted answer could break the attribute | Low | HTML-injection risk |
| D4 | `results/model_selection.json` (working tree) vs `results/research_eval.json` | Internal numeric inconsistency: selected-model silver 0.9375 in the regenerated `model_selection.json` vs 0.9355 in the committed `research_eval.json` | Informational | Documentation/drift |
| D5 | `src/answer_quality.py` `is_informative` | Digit/symbol-only answers return `True` (documented edge case) | Low | Known gap |
| D6 | `backend/db.py` `summary(days)` | `days` filters only `by_day`; totals/breakdowns are all-time | Low | Known quirk |
| D7 | `src/pipeline.py` advisor memory | `add_memory` mutates `mem_emb`/`mem_items` without the pipeline lock; no dedup; unbounded | Low/Medium | Known gap |

**Capability gaps (not defects — behaviour is correct-to-spec but the capability is
absent):** multi-intent detection (17/18 probes), Hindi/Devanagari evidence extraction
(Improvement 4 lexicon is English-only), Devanagari/low-resource dosage and product-name
detection in `answer_risk` (Improvement 2 Hindi recall 0/11), typo tolerance in the NER
gazetteer, and PII redaction not integrated into production.

---

## 9. Defects fixed

**None.** No code defect was fixed during this audit. Rationale, per the fix policy:

- D2 and D3 are the only clearly reproducible code defects, but both live in
  **production-protected** files (`backend/app.py`; `frontend/assets/js/app.js`, which is
  inside the frozen `baseline_hashes.json`). A fix would alter production behaviour beyond
  the "safe, isolated" scope of the policy and would invalidate a frozen protection
  manifest, which this audit was instructed not to disturb. They are therefore reported,
  not implemented.
- D1 cannot be "fixed" without re-running an experiment (which would regenerate the
  timestamp again); it is an integrity note.
- The remaining items are documented capability gaps, not defects, and the instructions
  forbade new features or restarting prior implementations.

Every production and prior-experiment file is therefore byte-identical to how it was found
(the four pre-existing uncommitted modifications are preserved untouched).

---

## 10. Behavioural failure analysis (Phase 8)

Independently re-ran `experiments/behavioural_regression/run_evaluation.py`: **168 cases,
109 passed, 59 failed, 0 crashes**; `metrics.json`, `cases.csv`, `failures.json` and
`REPORT.md` reproduced **byte-for-byte** (case-set SHA-256 `ed71cef9…`). The 59 failures
are unchanged from the previously reported set. Classification:

| Category | Scope | Cases | Failed | Failure types |
|---|---|---:|---:|---|
| A | Typo / spelling robustness | 24 | 12 | degraded 1, false_negative 11 |
| B | Hindi / Roman-Hindi / Hinglish | 24 | 11 | degraded 5, false_negative 5, missing_capability 1 |
| C | Negation / prohibitions | 18 | 10 | false_positive 10 |
| D | Numbers, units, dosage, ambiguity | 23 | 7 | false_negative 7 |
| E | Ambiguous / underspecified | 9 | 0 | — |
| F | Multi-intent | 18 | 17 | missing_capability 17 |
| G | Empty / malformed / oversized input | 27 | 1 | validation_gap 1 |
| H | Cross-component | 25 | 1 | false_negative 1 |
| **Total** | | **168** | **59** | degraded 6, FN 24, FP 10, missing-capability 18, validation-gap 1 |

**Representative failures and root causes** (full records in `failures.json`):

- **A — typo robustness (12).** e.g. `whea rust control` → `{crop: [], symptom: [rust]}`;
  `wheatt rustt` → `{}`. Root cause: the NER gazetteer is exact-match (`src/ner.py`); no
  spelling tolerance. Genuine code limitation; severity low (NER is display-only).
- **B — Hindi (11).** `gehu me ratua` → register `en` (exact-match word lists miss
  alternate spellings); `गेहूं में पीला रतुआ` → no entities; `gehu ka bhav kya hai` →
  `Other` (weak-label rules are English keywords). Root causes: register/NER/weak-label
  lexicons are Latin/English-centric. This is the most consequential cluster for a
  "voice-first multilingual" claim.
- **C — negation (10, all false positives).** `Do not spray chlorpyrifos.`,
  `spray mat karein` → `answer_risk == True`. Root cause: `answer_risk` has no negation
  handling; it flags the chemical/unit regardless of prohibition. Severity **high** for
  safety framing (could over-escalate, and mislabels negated advice as risky).
- **D — numbers/units (7).** `Apply two bottles`, `Apply 1 bottle dawa 1 tank`, `Apply 50
  dawa` → not flagged; `1 sachet`/`2 bori`/`2 chammach` conflict pairs → not detected. Root
  cause: `_DOSE` requires a Latin digit+unit token; container/household units are
  unsupported. Genuine code limitation; severity medium.
- **E — ambiguous (0).** All pass.
- **F — multi-intent (17).** All single-label; root cause: capability absent. Not a defect;
  a documented gap.
- **G — validation (1).** `{"text": "  "}` → accepted. See D2.
- **H — cross-component (1).** `Call ०९८७६५४३२१० for details` → no PHONE. Root cause: the
  privacy recogniser does not handle Devanagari digits with a trunk prefix. Genuine
  limitation; severity low.

**Preserved as regression tests; no assertion was weakened.** Expected values are the
study's own correctness definitions and are synthetic, not human labels.

---

## 11. Research metric verification

| Reported result | Source | Verified here | Which value |
|---|---|---|---|
| Production intent, silver n=496 | `research_eval.json` | Yes (read) | 0.9355 [0.9153, 0.9556] |
| Production intent, gold n=150 | `research_eval.json` | Yes | 0.90 [0.8533, 0.9467]; EN 0.94 / Hing 0.90 / HI 0.86 |
| No-augmentation gold | `research_eval.json` | Yes | 0.60 |
| Hinglish perturbed | `research_eval.json` | Yes | 0.856 ± 0.0233 |
| McNemar char SVM | `research_eval.json` | Yes | p=0.4049 silver / 0.4807 gold |
| Gate silver (full) | `selective_eval.json` | Yes | 0.2641 / 0.9542 / 0.0229 |
| Gate gold (full) | `selective_eval.json` | Yes | 0.2333 / 0.9429 / 0.0286 |
| Thresholds-only silver | `selective_eval.json` | Yes | 0.9254 / 0.9695 / 0.7168 |
| Answer screening | `answer_screening.json` | Yes | coverage 0.2923→0.1647; risky 0.4276→0.0 |
| Hybrid retrieval | `hybrid_retrieval.json` | Yes | committed 0.4514→0.4673 (novel 0.1887→0.2446); working tree 0.4545→0.4679 (novel 0.1866→0.2454) |
| Calibration | `calibration.json` | Yes | committed silver 1.77→2.32, oof 3.78→0.51; working tree silver 1.77→2.37, oof 3.81→0.60 |
| Voice | `voice_eval.json` | Arithmetic | WER 0.0146, n=50, acc 0.94/0.94 |
| Keyword-vs-meaning agreement | `annotation_summary.json` | Yes | 0.6751 |
| Geography | `corpus_geography.json` | Yes | 17 / 5,840 of 138,875 |
| Batch A grouped intent | `grouped_intent.json` | Yes | 0.9372 ± 0.0045; novel 0.9025 |
| Batch A grouped retrieval | `grouped_retrieval.json` | Yes | topic@1 0.8519; hybrid 0.8733 |
| Privacy PII (synthetic) | `privacy_audit/metrics.json` | Yes | balanced F1 0.9559 [0.9077, 0.9882] |
| Conflict detection | `conflict_detection/metrics.json` | Yes | P 1.0 / R 0.9038 / F1 0.9495 |
| Adversarial safety | `safety_evaluation/metrics.json` | Headline | content-rule F1 0.7545; gate-slice 0.7500 |
| Multi-reference | `multi_reference/metrics.json` | Headline | max-other 0.5788 vs own 0.3217 |
| Out-of-scope | `out_of_scope/metrics.json` | Headline | kNN 0.9091; LR 0.9351 |
| Behavioural regression | `behavioural_regression/` | Re-ran | 109/168 (0.6488), byte-identical |
| Full test suite | `tests/` | Re-ran | 224 passed |

**Consistency checks performed:** CSVs/JSON/Markdown agree for the behavioural suite
(byte-identical reproduction); `grouped_intent.json`/`grouped_retrieval.json` match their
report Markdown; the Safety F1 is reported as **two** labelled values (rule 0.7545,
gate-slice 0.7500) rather than conflated; the F1 values across improvements score different
tasks on non-overlapping samples and are **not** comparable. No case of a fabricated or
internally impossible number was found.

**Evidence classes (kept explicit):**

- **Human-validated / expert-reviewed:** none.
- **Corpus evaluation with LLM labels:** silver/gold intent, Batch A grouped metrics.
- **Synthetic evaluation:** PII spans, conflict pairs, adversarial safety, out-of-scope
  negatives, behavioural probes, synthetic speech WER.
- **Descriptive statistics:** leakage, call-log share, geography, conflict rule-fire rates.

**Known documentation inaccuracies that remain in the *original* sources (not corrected in
place, listed for the record):** `paper/main.tex` uses 124,987 as the "cleaned" corpus size
(correct: 138,875); `paper/generated/numbers.tex` is stale (call-log 8.2%, voice `n/a`);
the paper's "only partly human-verified" and "no personal data" claims are unsupported;
`README.md` quotes thresholds-only as 92.3%/72.3% (stored 92.54%/71.68%).

---

## 12. Final test results

| Check | Result |
|---|---|
| `python -m compileall` (all source dirs) | **PASS** (exit 0) |
| Full suite `OMP_NUM_THREADS=1 pytest -q` | **224 passed** |
| Per-file counts | api 10, gate 22, text_utils 9, grouped 11, privacy 36, annotation 30, conflict 30, adversarial safety 13, multi_reference 27, out_of_scope 25, behavioural 11 |
| Behavioural focused (`test_behavioural_regression.py`) | **11 passed** |
| Behavioural evaluation rerun | **109/168 passed, 0 crashes, byte-identical outputs** |
| Real pipeline smoke test | **PASS** (decisions match docs) |
| Import/compile checks | **PASS** |
| Reproducibility (behavioural suite) | **PASS** (metrics/cases/failures/REPORT identical) |
| Protected hashes | **1 changed** (see §13) |
| Pre-existing uncommitted changes | **Preserved** (unmodified) |

---

## 13. Protection / hash verification

- **Improvement 5 snapshot** (`batch_b_improvement5_hashes.json`, **389 files**): checked →
  **1 changed** (`results/out_of_scope/metrics.json`, timestamp-only; §7), 0 missing, 0
  unexpected additions. Group counts: 220 original, 50 Batch A, 14 Phase 1, 24 Phase 2,
  17 Improvement 1, 13 Improvement 2, 22 Improvement 3, 18 Improvement 4, 10 Improvement 5,
  1 config.
- **Older snapshots fail as expected**, because the audit file and config legitimately
  changed and later improvement files were approved additions: Improvement 3 snapshot → 2
  changed (audit, config) + 2 additions (Improvement 4/5 tests); Improvement 4 snapshot → 3
  changed (audit, config, the regenerated out-of-scope metrics) + 1 addition; the 308-file
  `batch_b_protection_hashes` → 1 changed (audit) + 16 additions (Improvement 1 files).
- **No production source (`src/`, `backend/`), model, dataset, Batch A output, or earlier
  improvement output changed** as a result of this audit. The only files written by this
  audit are the two new report files.
- **Pre-existing uncommitted state preserved:** `results/calibration.json`,
  `results/hybrid_retrieval.json`, `results/model_selection.json`,
  `scripts/13_hybrid_retrieval.py` (plus the pre-existing edits to the audit and Batch B
  docs). No `git reset`/`checkout`/`clean`/`restore`/`add`/commit was run.

---

## 14. Remaining issues

1. **No human verification** of any label; no kappa; the annotation study is PENDING. (§4 B22,
   §8/§10.)
2. **Hindi/Devanagari weakness** across register, NER, weak-label intent, dosage screening
   and PII (Devanagari digits). The "voice-first multilingual" claim is under-supported.
3. **Negation not handled** by `answer_risk` (10 false positives) — a safety-framing concern.
4. **Multi-intent detection absent** (17/18 probes).
5. **Container/household-unit dosage** and non-Latin units not screened or compared.
6. **Whitespace-only API input** (≥2 spaces) reaches the pipeline as empty text (D2).
7. **Attribute-context escaping** in the front end (D3).
8. **No answer-correctness/agronomist evaluation**; policy violations are rule matches.
9. **Corpus leakage/representativeness** (row split; Assam-heavy English logs).
10. **No auth/rate-limit/encryption/retention**; PII redaction not integrated; advisor
    memory not thread-safe/deduplicated.
11. **Stale paper assets and denominator/PII prose in `paper/main.tex`.**
12. **Legacy DistilBERT artefact absent**; its figures remain historical only.
13. **Unpinned dependencies and result drift** (four uncommitted modified files).

---

## 15. Production-readiness assessment

**Not production-ready.** The system is a competent research prototype: the end-to-end
pipeline runs, the safety gate behaves as specified, and the headline evaluation numbers are
reproducible from committed artefacts (224 tests pass; behavioural outputs reproduce
byte-for-byte). But deployment is blocked by: no human-verified labels or agronomist
review; no answer-correctness measurement; severe leakage on the original split; a
non-representative corpus; synthetic-English-only speech; unhandled negation and
Hindi/Devanagari gaps in safety screening; unintegrated PII redaction; and the complete
absence of authentication, authorization, rate limiting, encryption, retention and
monitoring. The most defensible claim remains **"selective, auditable triage with honest
limitations"** — not "high-accuracy automated advice" and not "safe".

---

*This document and `FINAL_REPORT_v2.md` were produced by the final audit. The originals
(`FINAL_REPORT.md`, `KCC_Sahayak_Verification_Audit.md`) are unchanged. Correct any number
by fixing the underlying artefact and re-running the relevant script — never by hand-editing
prose.*
