# Answer-quality evaluation framework

Separates the dimensions of answer quality that the project can and cannot measure.
Aggregates committed artefacts; runs no model and **invents no score**.

Run: `.venv312/bin/python experiments/answer_quality_eval/framework.py --overwrite`

| Dimension | Status | Evidence |
|---|---|---|
| Retrieval relevance | measured | grouped retrieval: topic@1 0.8519, recall@5 0.9508, MRR@10 0.8961, nDCG@5 0.9071 |
| Answer completeness | proxy | mean own token-F1 0.3217 (multi-reference) |
| Semantic similarity | proxy | mean max embedding similarity (wording closeness, not correctness) |
| Factual consistency | proxy | mean max-other token-F1 0.5749 (agreement with other historical answers) |
| Crop/entity consistency | unavailable | no per-row entity ground truth |
| Dosage consistency | proxy | conflict-detection rules (rule match, not truth) |
| Unit consistency | proxy | conflict-detection basis/dimension rules |
| Safety compliance | measured | answer-screening v1/v2 F1 on authored cases |
| Escalation correctness | proxy | gate coverage / selective accuracy / violation rate |
| Unsupported-claim rate | proxy | share of rows with own token-F1 < 0.2 (0.3648) |
| **Agronomic correctness** | **unavailable** | **no agronomist/expert labels exist** |

The framework is designed so verified reference labels can be added later and promoted to
`measured`. Similarity, agreement and token overlap are **not** agricultural correctness.
