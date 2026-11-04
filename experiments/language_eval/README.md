# Language capability report

Per-language capability check reusing the committed 150-query gold set (50 each en / hinglish / hi)
and the shipped components. No training.

Run: `.venv312/bin/python experiments/language_eval/run.py --overwrite`

| Language | intent acc. | register detection | entity coverage |
|---|---:|---:|---:|
| English | 94% | 1.00 | 0.58 |
| Hinglish | 90% | 1.00 | 0.02 |
| Hindi (Devanagari) | 86% | 1.00 | 0.00 |
| Roman-Hindi | not measured | 1.00 | 0.00 |

Intent accuracy is quoted from `results/research_eval.json`. Register detection is script-based
and near-perfect; it says nothing about understanding. **Entity coverage collapses for Hinglish
and Devanagari (0.02 / 0.00)** because the NER gazetteer is English-only.

Dosage/risk detection recall on authored per-language probes: v1 (shipped) 0.0 on Devanagari vs
v2 (candidate, `answer_screening`) 1.0; unchanged (1.0) elsewhere.

Limitations: n=50 per register, LLM-authored, not human-verified; entity coverage is not
precision/recall (no entity ground truth); Roman-Hindi intent accuracy is not separately measured.
