# Language capability report

**Evidence class:** reuse of the committed gold set and shipped components; no training.
Small, LLM-authored sample (n=50 per register) — a capability check, not a benchmark.

## Intent accuracy (from `results/research_eval.json`)

| Language | Accuracy | Source |
|---|---:|---|
| en | 0.94 | results/research_eval.json main['production (emb+char, aug)'] |
| hinglish | 0.9 | results/research_eval.json main['production (emb+char, aug)'] |
| hi | 0.86 | results/research_eval.json main['production (emb+char, aug)'] |
| roman_hindi | not measured | no separate Roman-Hindi gold queries were authored |

## Register detection and entity coverage (recomputed)

| Language | n | register detection | entity coverage |
|---|---:|---:|---:|
| en | 50 | 1.0 | 0.58 |
| hi | 50 | 1.0 | 0.0 |
| hinglish | 50 | 1.0 | 0.02 |

## Dosage/risk detection recall on authored per-language probes

| Language | n | v1 (shipped) | v2 (candidate) |
|---|---:|---:|---:|
| en | 2 | 1.0 | 1.0 |
| hinglish | 2 | 1.0 | 1.0 |
| roman_hindi | 2 | 1.0 | 1.0 |
| devanagari | 2 | 0.0 | 1.0 |

## Limitations

- Gold set is small (n=50 per register) and LLM-authored, not human-verified.
- Entity coverage counts any crop/pest hit; there is no entity ground truth, so no precision/recall.
- Roman-Hindi has no separate gold queries; its intent accuracy is not measured here.
- Devanagari register detection is script-based and near-perfect; it says nothing about understanding.
