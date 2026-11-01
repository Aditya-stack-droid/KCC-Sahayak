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
4. the closest logged question really addresses the question: same topic, and every crop the farmer names (and any
   comparison they ask for, e.g. "tomato or cucumber") appears in it. Otherwise: "no solid answer in the corpus";
5. the retrieved text is real advice, not a call-log note such as "explained in details";
6. the retrieved text contains no dosage or chemical instruction.

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

## Optional: Gemini second opinion

Next to the KCC-corpus answer, the demo can show what Google's Gemini suggests for the same question. It is **display only**:
labelled "AI-generated, not verified", never sent automatically, and it never changes the safety-gate decision. The prompt
forbids exact pesticide/fertiliser doses and points the farmer to a KVK or the Kisan Call Centre (1800-180-1551).

1. Create a key at https://aistudio.google.com/apikey.
2. Copy `.env.example` to `.env` and paste the key after `GEMINI_API_KEY=` (`.env` is git-ignored; never commit a key).
3. Restart the server. Without a key the feature simply says it is not set up and nothing is sent anywhere.

The text of the question is sent to Google when this is on. Optional `GEMINI_MODEL` overrides the default model name.

## Data source (cite this)

The corpus (`data/raw/kcc_mapping_full.csv`, 139,542 raw question/answer rows; 138,875 queries after cleaning) was
downloaded from a public Hugging Face mirror of the Government of India's Kisan Call Centre (KCC) logs:

- **Mirror used:** `hisham1404/kcc_call_center_query_embedded`, file `kcc_ollama_nomic-embed-text_mapping.csv`
  (https://huggingface.co/datasets/hisham1404/kcc_call_center_query_embedded). Only the `questions` and `answers` columns are used.
- **Original publisher:** Kisan Call Centre transcripts, Department of Agriculture & Farmers Welfare, via the Open Government
  Data Platform India: https://www.data.gov.in/resource/kisan-call-centre-kcc-transcripts-farmers-queries-answers
- The mirror's own provenance and processing were **not independently verified** against the data.gov.in export. For a paper,
  cite both and say so.
- Not real farmer data: the 1,200 spoken-style training paraphrases, the 150-query gold set, the labels and the speech
  test audio were produced by us (see `docs/annotation_guidelines.md`).

## Results

| | Result |
|---|---|
| Intent accuracy, held-out corpus calls (n=496, labelled by meaning) | **93.5%** (95% CI 92-96) |
| Intent accuracy, independent gold set (n=150) | **90.0%** (85-95): English 94%, Hinglish 90%, Hindi 86% |
| Hinglish with random re-spelling | 85.6% |
| Same model **without** spoken-style training text, gold set | 60.0% |
| Gate: share auto-answered / accuracy of those / policy violations (held-out) | 24.6% / 95.1% / 2.5% |
| Confidence thresholds alone (no safety rules) | 92.5% auto-answered, **71.7% policy violations** |
| Answer F1 vs advisor's real answer: dense -> hybrid (novel queries) | 0.451 -> 0.467 (0.189 -> 0.245) |

Honest comparisons:
- Encoder comparison (scripts/21, same data and head; silver / gold): MiniLM (deployed) 93.5 / 89.3, LaBSE 93.1 / 90.0, MuRIL 91.9 / 84.7, IndicBERTv2 93.3 / 91.3. None differs significantly from the deployed encoder (McNemar p >= 0.17, gold n=150).
- Retrieval vs a KisanQRS-style baseline (simplified re-implementation of its retrieval idea, not its LSTM): answer F1 0.459 vs 0.467, a small gain; the clear benefit of our screening is call-log notes returned falling from 4.6% to 0.2%.
- A plain **character n-gram SVM is statistically indistinguishable** from the deployed model
  (McNemar p = 0.40 held-out, 0.48 gold). The deployed model is significantly better than word
  TF-IDF on gold (p = 0.015) and than embeddings alone (p < 0.001).
- The sentence embedding's value is **zero-shot cross-lingual transfer**: with no Hindi training
  data it reaches 68% on Hindi, versus 4% for character n-grams. Once Hindi training text exists,
  character n-grams are competitive.
- Even this model scores 98.6% on near-duplicate test queries but 87.7% on novel ones.

Corpus audit (`results/research_eval.json`): keyword labels agree with labels by meaning only
**67.5%** of the time; **70.7%** of the original test queries have a near-duplicate in training;
**8.7%** of logged answers are call-log notes rather than advice; only **17 of 138,875** queries
mention a Haryana place while about 5,840 mention Assam. The old DistilBERT scored 99.5% against
keyword labels but 65.7% against labels by meaning.

## Limitations

- **Labels are LLM-assisted and not yet human-verified.** All 3,496 training/test labels and the
  150-query gold set were produced by one LLM assistant following `docs/annotation_guidelines.md`.
  Use `scripts/18_review_tools.py` to measure human agreement (Cohen's kappa).
- The gold set and the augmentation paraphrases share an author, so Hindi/Hinglish figures are
  likely optimistic. At n=50 per language intervals are about +/-8 points.
- **Scope: the public multi-state KCC corpus, not one state.** It is mostly Assamese in content, and
  no state-specific (e.g. Haryana) results are claimed. The pipeline itself is state-independent.
- The data is English advisor-log text with no audio. Speech was tested on synthetic speech only.
- "Policy violation" is a rule-based compliance measure defined by us, not independent evidence of
  harm; no agronomist has judged answer correctness. Auto-answers are terse log text.
- Roughly three quarters of queries are escalated; that is the cost of the safety rules.

## Quick start (fresh clone from GitHub)

```bash
python -m venv .venv
```

```bash
.venv/Scripts/python.exe -m pip install -r requirements.txt
```

```bash
.venv/Scripts/python.exe scripts/setup_models.py
```

```bash
.venv/Scripts/python.exe -m uvicorn app:app --app-dir backend --port 8420
```

Then open http://localhost:8420. `setup_models.py` rebuilds the two large retrieval files that are too big for
GitHub (about 10-20 minutes, once). Everything else the app needs is in the repository. On Windows,
`run_app.bat` does the setup check and starts the site. Needs about 3 GB of free RAM and Python 3.10+.
Microphone input also needs `ffmpeg` on PATH.

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
