# KCC-Sahayak: Final Report

**Status:** working system and honest evaluation complete; human verification of labels,
real farmer speech and agronomist review are still outstanding (see Limitations).

An earlier version of this report claimed 99.37% intent accuracy. **That claim is withdrawn.**
It measured agreement with keyword rules, not understanding. Everything below is measured
against labels assigned by meaning, on an independent test set, with baselines and
significance tests. Full detail: [`README.md`](README.md), [`paper/main.tex`](paper/main.tex),
`results/*.json`.

## What was built
A voice-first triage system for Kisan Call Centre queries: Whisper speech-to-text, one
multilingual sentence encoder feeding (a) a calibrated intent classifier and (b) hybrid
dense + TF-IDF retrieval over ~125,000 logged answers, a crop/pest tagger, and a **safety
gate** that auto-answers only low-risk, non-time-sensitive queries whose retrieved answer is
real advice with no dosage or chemical instruction, escalating everything else to a human
advisor whose answers are fed back into retrieval. A FastAPI + web UI exposes it (live demo,
advisor console, operations, evaluation pages).

## Headline results
| | |
|---|---|
| Intent accuracy, held-out calls (n=496) | 93.5% (95% CI 92-96) |
| Intent accuracy, independent gold set (n=150) | 90.0% (85-95); EN 94 / Hinglish 90 / Hindi 86 |
| Without spoken-style training text (gold) | 60.0% |
| Gate: auto-answered / accuracy of those / policy violations | 24.6% / 95.1% / 2.5% |
| Thresholds only, no safety rules | 92.5% auto-answered, 71.7% policy violations |
| Encoders (MiniLM / LaBSE / MuRIL / IndicBERTv2), gold | 89.3 / 90.0 / 84.7 / 91.3, none significantly different |
| Retrieval answer F1: KisanQRS-style vs ours | 0.459 vs 0.467 (small); call-log notes returned 4.6% -> 0.2% |

**What to say about overall accuracy:** about 90% intent accuracy on an independent
three-language test set (93.5% on held-out calls); "whole-system accuracy" is not a single
number because the system deliberately refuses to answer about three quarters of queries.

## Findings about the public KCC corpus
- keyword labels agree with meaning-based labels only 67.5% of the time;
- on the original test set a TF-IDF baseline (99.8%) matched the fine-tuned transformer (99.5%);
- 70.7% of original test queries have a near-duplicate in training;
- 8.7% of logged answers are call-log notes ("explained in details"), not advice;
- only 17 of 138,875 queries mention a Haryana place; the corpus is overwhelmingly Assamese.

## Honest comparisons
A character n-gram SVM is statistically indistinguishable from the deployed model
(McNemar p = 0.40 / 0.48). The embedding's value is zero-shot cross-lingual transfer
(68% vs 4% on Hindi with no Hindi training data). Spoken-style training text is the decisive
ingredient.

## Limitations
LLM-assisted labels (one annotator, not yet human-verified); gold set and augmentation share
an author; n=50 per language (about +/-8 points); Assam-dominated English advisor logs, no
audio; synthetic speech only; "policy violation" is a rule-based measure we defined; no
agronomist review; no user study; no comparison to a retrieval-augmented LLM.

## Next steps (need people, not code)
1. Two human annotators fill `data/review/review_sheet.csv`; run `scripts/18_review_tools.py score`.
2. Record real Hindi/English spoken queries; evaluate ASR.
3. Agronomist review of auto-answers.
4. Optional: re-run on a state-specific KCC export, since the pipeline is state-independent.
