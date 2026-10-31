# KCC-Sahayak

A voice-first, safety-gated triage system for Kisan Call Centre (KCC) farmer queries,
plus an honest audit of the public KCC corpus it is built on.

> **Read this first.** An earlier version of this project reported 99% accuracy. That
> number measured agreement with keyword rules, not understanding, and has been
> withdrawn. All figures below come from `results/*.json`; every caveat is listed in
> [Limitations](#limitations).

## What it does

```
voice or text (English / Hinglish / Devanagari Hindi)
   -> Whisper speech-to-text (voice only)
   -> ONE multilingual sentence encoder
        |-> intent: sentence embedding + char n-gram TF-IDF -> logistic regression (temperature-calibrated)
        `-> retrieval: dense (FAISS) + TF-IDF fusion over ~125k logged answers
              (+ answers human advisors have written since deployment)
   -> crop / pest entity tagging
   -> SAFETY GATE  ->  AUTO-ANSWER  or  ESCALATE to a human advisor
```

The gate auto-answers only when **all** of these hold, otherwise it escalates and says why:
1. the predicted category is not high-stakes (plant protection, nutrient management, government scheme);
2. it is not time-sensitive (weather, market price) - a logged answer is stale by construction;
3. intent confidence >= 0.70 and retrieval similarity >= 0.55;
4. the retrieved text is real advice, not a call-log note such as "explained in details";
5. the retrieved text contains no dosage or chemical instruction.

Escalated queries go to the **Advisor Console**; an advisor's answer is returned to the farmer and
added to the retrieval memory.

## Run the demo

Double-click **`run_app.bat`** (first start takes about a minute while models load), or:

```bash
.venv/Scripts/python.exe -m uvicorn app:app --app-dir backend --port 8420
```

Open <http://localhost:8420>. Pages: Live Demo (voice or text), Advisor Console, Operations
(live log statistics), Evaluation (the honest numbers), System, About.
Voice input needs `ffmpeg` (installed with `winget install Gyan.FFmpeg`).

Try: *"which tomato variety is good for me?"* (auto-answered), *"gehu mein peela rust aa gaya hai
kya spray karun"* (escalated: pest advice), *"आज मंडी में सरसों का भाव क्या चल रहा है?"*
(escalated: prices go stale).

## Results

| | Result |
|---|---|
| Intent accuracy, held-out corpus calls (n=496, labelled by meaning) | **93.5%** (95% CI 92-96) |
| Intent accuracy, independent gold set (n=150) | **90.0%** (85-95): English 94%, Hinglish 90%, Hindi 86% |
| Hinglish with random re-spelling | 85.6% |
| Same model **without** spoken-style training text, gold set | 60.0% |
| Gate: share auto-answered / accuracy of those / policy violations (held-out) | 26.0% / 96.1% / 2.3% |
| Confidence thresholds alone (no safety rules) | 92.3% auto-answered, **72.3% policy violations** |
| Answer F1 vs advisor's real answer: dense -> hybrid (novel queries) | 0.452 -> 0.468 (0.189 -> 0.247) |

Honest comparisons:
- A plain **character n-gram SVM is statistically indistinguishable** from the deployed model
  (McNemar p = 0.40 held-out, 0.48 gold). The deployed model is significantly better than word
  TF-IDF on gold (p = 0.015) and than embeddings alone (p < 0.001).
- The sentence embedding's value is **zero-shot cross-lingual transfer**: with no Hindi training
  data it reaches 68% on Hindi, versus 4% for character n-grams. Once Hindi training text exists,
  character n-grams are competitive.
- Even this model scores 98.6% on near-duplicate test queries but 87.7% on novel ones.

Corpus audit (`results/research_eval.json`): keyword labels agree with labels by meaning only
**67.5%** of the time; **70.7%** of the original test queries have a near-duplicate in training;
**8.2%** of logged answers are call-log notes rather than advice; only **17 of 138,875** queries
mention a Haryana place while about 5,840 mention Assam. The old DistilBERT scored 99.5% against
keyword labels but 65.7% against labels by meaning.

## Limitations

- **Labels are LLM-assisted and not yet human-verified.** All 3,496 training/test labels and the
  150-query gold set were produced by one LLM assistant following `docs/annotation_guidelines.md`.
  Use `scripts/18_review_tools.py` to measure human agreement (Cohen's kappa).
- The gold set and the augmentation paraphrases share an author, so Hindi/Hinglish figures are
  likely optimistic. At n=50 per language intervals are about +/-8 points.
- **The corpus is overwhelmingly Assamese, not Haryanvi.** Data.gov.in's Haryana export is
  CAPTCHA-gated and was not obtained. Nothing here is Haryana-specific.
- The data is English advisor-log text with no audio. Speech was tested on synthetic speech only.
- "Policy violation" is a rule-based compliance measure defined by us, not independent evidence of
  harm; no agronomist has judged answer correctness. Auto-answers are terse log text.
- Roughly three quarters of queries are escalated; that is the cost of the safety rules.

## Reproduce

```bash
python -m venv .venv && .venv/Scripts/python.exe -m pip install -r requirements.txt
```

| Script | Purpose |
|---|---|
| `01_prepare_data.py` | clean the raw corpus, keyword labels, splits |
| `03_build_retrieval_index.py` | dense FAISS index |
| `06_build_gold_set.py` | write the 150-query gold set |
| `07_evaluate_gold_and_baselines.py` | first baseline/gold comparison |
| `08`-`09`, `11`-`12` | sample, annotate, merge, augment (labels in `data/annotation/`) |
| `10`, `16_model_selection.py --save` | compare candidates, calibrate and save the deployed model |
| `13_hybrid_retrieval.py` | TF-IDF side + fusion weight |
| `14_selective_and_screening_eval.py` | selective accuracy, gate ablation |
| `15_voice_eval.py` | synthetic-speech end-to-end test |
| `17_research_eval.py` | CIs, McNemar, ablations, learning curve, corpus audit |
| `18_review_tools.py` | human annotation sheet and agreement |
| `19_make_paper_assets.py` | numbers/tables/figures for `paper/main.tex` |

`pytest tests` runs the unit tests. The paper (`paper/main.tex`) takes every number from the
result files; compile with `pdflatex main.tex` inside `paper/`.

## Layout

```
backend/    FastAPI app, SQLite query log + advisor queue
frontend/   demo UI (HTML/CSS/JS, Chart.js)
src/        pipeline, gate, retrieval, answer-quality filter, language detection, entities
scripts/    numbered experiment scripts (above)
data/       raw corpus, processed splits, annotation/, gold/, review/
models/     deployed intent head, retrieval indexes (generated)
results/    all evaluation outputs (JSON/CSV)
paper/      LaTeX source, generated tables/figures
tests/      unit tests
docs/       annotation guidelines, literature survey
```
