# Updated synopsis text (Review 2 follow-up)

Use this to update the Project Synopsis (.docx) and the Review 2 slides. It reflects what was actually
built, not the original plan. Numbers marked **[fill]** come from `results/*.json` once the final
evaluation run has finished.

## Title (drop "in Haryana")
**KCC-Sahayak: A Voice-First, Confidence-Gated Triage Framework for Kisan Call Centre Query Resolution**

## Abstract (revised)
The Kisan Call Centre (KCC) routes every farmer query to a live human agent, so cost scales linearly
with staff while years of call transcripts go unused. KCC-Sahayak is a voice-first NLP pipeline that
transcribes farmer speech (Whisper), classifies the query type (multilingual sentence embedding plus
character n-gram features), extracts crop and pest entities (rule-bootstrapped NER), and retrieves
matched historical advisor answers (FAISS dense search fused with TF-IDF). Unlike prior retrieval
systems, human handoff is a first-class safety objective: pesticide, fertiliser, scheme, price and
weather queries are always escalated, and an answer is auto-sent only if the retrieved question has
the same topic as the query, the answer is real advice rather than a call-log note, and it contains
no dosage or chemical instruction. Advisor answers re-enter the retrieval index. We evaluate on a
public multi-state KCC corpus (about 139,000 queries) with meaning-based labels, an independent
English/Hinglish/Hindi gold set, bootstrap confidence intervals, significance tests and a gate
ablation. The full stack uses open-source models and can be re-run on any state's KCC export.

## What changed from the original synopsis, and why (say this in the review)
| Original plan | What was done | Reason |
|---|---|---|
| Scoped to Haryana (Hindi/Haryanvi) | Public multi-state KCC corpus; English, Hinglish and Hindi tested | The official Haryana export is CAPTCHA-protected; the public corpus has almost no Haryana rows. Scope reduced to the available data; the pipeline is state-independent. |
| Fine-tune MuRIL / IndicBERT | Multilingual MiniLM encoder + char-n-gram head. MuRIL, IndicBERTv2 and LaBSE were compared (`results/encoder_comparison.json`) | MuRIL, IndicBERTv2 and LaBSE were compared on the same data; none is significantly better than MiniLM (MuRIL is lower on the independent set, IndicBERTv2 about 2 points higher but within noise at n=150). MiniLM kept for its size and speed. End-to-end fine-tuning of MuRIL/IndicBERTv2 was not completed (out of memory on an 8 GB CPU laptop); only frozen encoders were compared. |
| Labels from KCC query types | Labels assigned by meaning (LLM-assisted, human verification pending) | Keyword-rule labels agree with meaning only 67.5%; the first model's 99% was an artefact of that. |
| FAISS retrieval | Hybrid dense + TF-IDF retrieval, call-log notes filtered, topic-aware ranking | Pure dense retrieval returned "unable to resolve" notes and off-topic answers. |
| Whisper on Haryanvi/telephone audio | Synthetic English speech only: WER 1.5% (3.4% without the hint) with a farming-vocabulary hint | Real recordings still to be collected. |

## Results slide (replace the placeholder after the final run)
- Intent accuracy, independent 3-language gold set (n=150): **90%** (95% CI 85-95): English 94, Hinglish 90, Hindi 86
- Intent accuracy, held-out corpus calls (n=496): **93.5%** (95% CI 92-96)
- Safe gate: auto-answers **24.0%** of queries, **95.0%** correct on those, **2.5%** policy violations (held-out)
- Without the safety rules: answers **92.5%** but **72.3%** of those break policy
- Encoder comparison, same data and head for every encoder (silver / gold accuracy, embedding + char n-grams): MiniLM (deployed) 93.5 / 89.3; LaBSE 93.1 / 90.0; MuRIL 91.9 / 84.7; IndicBERTv2 93.3 / 91.3. No encoder is significantly different from the deployed one (McNemar p >= 0.17, gold n=150), so the choice is not evidence that MuRIL/IndicBERT are worse or better here. End-to-end fine-tuning was **not completed**: it crashed out of memory on this 8 GB CPU laptop, so only the frozen-encoder comparison is reported.
- Retrieval vs KisanQRS-style baseline (answer F1 vs advisor answer, n=2,737): KisanQRS-style 0.459 vs ours 0.468 (+0.009, small); the clear gain is that call-log notes returned fall from 4.6% to 0.0%. The KisanQRS-style row is a simplified re-implementation of its retrieval idea, not their LSTM system.

## Fixes to the synopsis and slides
- USNs: Aditya Hubly and Ankur P B have different USNs in the .docx and the .pptx. Confirm and correct.
- Chow reference: 1970 (slide 10 row says 2024).
- Kavitha and Anandaraja: 2018 (slide 10 row says 2023).
- Rows 4 and 7 (Godara, Sahu) justified Haryana; rewrite them as background on KCC query patterns and
  extension access in general.
- Add rows: MuRIL/IndicBERT comparison result and the KisanQRS-style baseline result.
