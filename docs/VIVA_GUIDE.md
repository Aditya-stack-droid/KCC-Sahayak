# Viva guide: KCC-Sahayak

## 30-second pitch
Kisan Call Centre advisors answer every farmer query by hand. KCC-Sahayak is a **triage layer**: it
understands a spoken or typed query in English, Hinglish or Hindi, finds the closest verified answer
from about 125,000 past calls, and **automatically answers only when it is safe to**. Everything
risky (pesticides, fertiliser doses, schemes, prices, weather) goes to a human advisor, and the
advisor's answer teaches the system. It is not a replacement for advisors.

## 3-minute demo (http://localhost:8420, or double-click `run_app.bat`)
1. **Live Demo -> English chip:** *"which tomato variety is good for me?"* -> answered automatically.
   Point at the five stages and the confidence bars.
2. Type *"gehu mein peela rust aa gaya hai kya spray karun"* (Hinglish) -> **escalated: pest advice is high-stakes**.
3. Type *"आज मंडी में सरसों का भाव क्या चल रहा है?"* -> **escalated: prices go stale**.
4. **Advisor Console:** open the pending query, write an answer, send it. Re-ask the question.
5. **Operations:** live counts. **Evaluation:** the honest numbers (say this part out loud).
6. Optional: tap the microphone and speak a question.

## The numbers to quote
| | |
|---|---|
| Intent accuracy, independent 3-language test set (n=150) | **90%** (95% CI 85-95): English 94, Hinglish 90, Hindi 86 |
| Intent accuracy, held-out corpus calls (n=496) | **93.5%** (92-96) |
| Safe-gate result | answers about a quarter of queries, ~95% accurate on those, ~2% policy violations |
| Without safety rules (thresholds only) | answers ~92% of queries but ~72% of them break safety policy |

"Overall accuracy of the whole model" is **not a single number**, because the system deliberately
refuses to answer about three quarters of queries. Say that, then give the table.

## Likely questions, and honest answers
**Q. Why isn't the accuracy 99%?**
An earlier version scored 99%, but that measured agreement with the keyword rules that created its
labels. Those rules agree with labels by meaning only 67.5% of the time. Scored against meaning, that
model got 65.7%. I rebuilt the labels, the test sets and the baselines.

**Q. How were the labels made?**
3,496 queries were labelled by meaning using written guidelines (`docs/annotation_guidelines.md`),
with an LLM assistant doing the labelling. They are **not yet human-verified**; a review sheet and an
agreement script (Cohen's kappa) are included. That is the main limitation.

**Q. Is your model better than a simple baseline?**
Not significantly, against a character n-gram SVM (McNemar p = 0.40 and 0.48). It is clearly better than
word TF-IDF and than embeddings alone. The embedding's value is **zero-shot Hindi**: 68% with no
Hindi training data, versus 4% for character n-grams. Training on spoken-style text was the decisive
factor: without it accuracy on the independent set drops from 90% to 60%.

**Q. Which data did you use, and is it state-specific?**
The public multi-state KCC corpus (about 139,000 queries with advisor answers). It has no state column
and is mostly Assamese in content, so I report results on this corpus and make no state-specific claim.
The pipeline is state-independent and can be re-run on any KCC export.

**Q. What is new?**
Not voice or retrieval on their own (Farmer.Chat, AgriLLM and KisanQRS exist). The contributions are
the **audit** of the KCC corpus (label validity, test-set leakage, call-log notes, geography), the
**content-aware safety gate** (it checks the retrieved answer for dosages and never reuses stale
weather/price answers), and a full ablation showing what each rule buys.

**Q. Why does the gate answer so few queries?**
That is the cost of safety. Thresholds alone answer ~92% but 72% of those violate policy; with the full
gate, 2% do. In a real deployment advisors still handle the rest, but the routine, low-risk quarter is automated.

**Q. What are the weaknesses?**
One annotator (an LLM) and no human verification yet; my gold set and paraphrases share an author
(Hindi/Hinglish likely optimistic); only 50 queries per language (about +/-8 points); synthetic speech
only; data is English advisor logs; "violation" is a rule-based measure I defined; no agronomist has
judged the answers; auto-answers are terse log text.

**Q. Did the speech part work?**
On synthetic English speech, Whisper small had a ~3% word error rate and intent accuracy fell only ~2
points. It mishears farming words ("mandi" -> "monday"), so I added a vocabulary hint (see
`results/voice_eval.json`). Real farmer speech and Hindi speech are untested.

**Q. Why not MuRIL or IndicBERT, as your synopsis said?**
I compared MiniLM, LaBSE, MuRIL and IndicBERTv2 on the same data and head. None is significantly different from
the deployed MiniLM (gold 89.3 / 90.0 / 84.7 / 91.3, n=150, McNemar p >= 0.17), so I kept the smaller, faster one.
This does not show Indic models are worse; the test set is too small to separate them.

**Q. Did you compare with KisanQRS?**
Only a simplified version of its retrieval idea (equal-weight lexical + semantic), not its LSTM layer. Answer F1 was
0.459 vs 0.468 for mine, a small gain. The clear benefit is that my screening drops call-log notes from 4.6% to 0.0%.

**Q. Why not an LLM?**
Generative models can invent pesticide doses. This system only returns text a human advisor actually
wrote and escalates the dangerous categories. I did not compare against an LLM+RAG baseline (listed as future work).

**Q. What would you do next?**
Human verification with two annotators; real recorded speech; an agronomist reviewing auto-answers;
a state-specific KCC export; an LLM+RAG comparison.

## Do NOT say
- "99% accurate" or "the whole model is X% accurate".
- "Haryana-specific".
- "Human-labelled". (Say "LLM-assisted, verification in progress".)
