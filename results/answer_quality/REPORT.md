# Answer-quality evaluation framework

**Evidence class:** aggregation of committed artefacts; no new model run, no expert labels.
Each dimension is labelled **measured**, **proxy**, or **unavailable**; unavailable
dimensions are not scored.

| Dimension | Status | What it measures | Key metric | Source |
|---|---|---|---|---|
| retrieval_relevance | measured | topical retrieval quality on the grouped test split (weak-label relevance) | recall_at_1=0.8519; recall_at_5=0.9508; mrr_at_10=0.8961; ndcg_at_5=0.9071 | results/improved_evaluation/grouped_retrieval.json |
| answer_completeness | proxy | token overlap between the retrieved answer and the row's own logged answer | mean_own_f1=0.3217; rows=13815 | results/multi_reference/per_query_dense_all.csv |
| semantic_similarity | proxy | max embedding similarity to any reference answer (wording closeness, NOT correctness) | mean_max_semantic=0.6989 | results/multi_reference/per_query_dense_all.csv |
| factual_consistency | proxy | agreement with other historical answers to the same question (agreement, not truth) | mean_max_other_f1=0.5749 | results/multi_reference/per_query_dense_all.csv |
| crop_entity_consistency | unavailable | no per-row crop/entity ground truth for retrieved answers | — | — |
| dosage_consistency | proxy | rule-fired dosage conflicts between answers to the same question (rule match, not truth) | pair_precision=1.0; pair_recall=0.9038 | results/conflict_detection/metrics.json |
| unit_consistency | proxy | basis/dimension conflicts detected by the same rules | basis_f1=0.9091; dimension_f1=1.0 | results/conflict_detection/metrics.json |
| safety_compliance | measured | content safety rule agreement on authored cases (v1 shipped vs v2 candidate) | v1_f1=0.5882; v2_f1=0.9714 | results/answer_screening/metrics.json |
| escalation_correctness | proxy | coverage / selective intent accuracy / rule-defined violation rate of the gate | coverage=0.2641; selective_accuracy=0.9542; violation_rate=0.0229 | results/selective_eval.json |
| unsupported_claim_rate | proxy | share of rows whose retrieved answer barely overlaps its own logged answer (own_f1 < 0.2) | share=0.3648; definition=own_f1 < 0.2; a low-support proxy, not a factual check | results/multi_reference/per_query_dense_all.csv |
| agronomic_correctness | unavailable | no agronomist or expert labels exist; correctness cannot be measured from the corpus | — | — |

## Headline

- **retrieval_topic_match_at_1**: 0.8519
- **answer_own_token_f1**: 0.3217
- **answer_max_other_token_f1**: 0.5749
- **unsupported_claim_rate_proxy**: 0.3648
- **agronomic_correctness**: unavailable

## Warnings

- Similarity, agreement and token overlap are NOT agricultural correctness.
- No agronomist has reviewed any answer; no expert labels exist.
- Proxy metrics use historical advisor answers as references, which may be contradictory or outdated.
