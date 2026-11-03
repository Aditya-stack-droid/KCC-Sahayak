# Grouped retrieval evaluation (Batch A)

Grouped corpus split: 125,044 index rows (train+val), 13,831 test rows. Relevance = weak-label topic match (proxy, not correctness).

## Leakage check after grouping

- Test queries with a >=0.95 neighbour in the grouped index: **0.09%**
- Test queries with a >=0.90 neighbour: 69.94%
- Test queries with max similarity <0.80 (novel): 6.15%

## Dense retrieval (all grouped test rows)

| Metric | Value |
|---|---:|
| Topic match @1 | 0.8519 (95% CI [0.8462, 0.8576]) |
| Recall @5 | 0.9508 |
| MRR @10 | 0.8961 |
| nDCG @5 | 0.9071 |
| Answer token-F1 @1 (all) | 0.3213 |
| Answer token-F1 @1 (informative gold) | 0.3293 |

## Dense vs hybrid (same 3,000-query sample)

| System | Topic match @1 | Recall @5 | MRR @10 | Answer F1 |
|---|---:|---:|---:|---:|
| Dense | 0.8403 | 0.9450 | 0.8881 | 0.3220 |
| Hybrid (alpha=0.4) | 0.8733 | 0.9630 | 0.9132 | 0.3800 |

## By novelty bucket (dense, all test rows)

| Bucket | N | Topic match @1 | Recall @5 | Answer F1 |
|---|---:|---:|---:|---:|
| near_duplicate | 9673 | 0.8702 | 0.9568 | 0.3319 |
| similar | 3308 | 0.8168 | 0.9368 | 0.3134 |
| novel | 850 | 0.7812 | 0.9365 | 0.2325 |

## Original row-level reference (not directly comparable)

- Near-duplicate share (>=0.95): 0.7073
- Topic match on near-duplicates: 0.9807
- Topic match on novel queries: 0.7365

## Limitations

- Relevance is a weak-label topical proxy, not human relevance or agronomic correctness.
- Only one grouped split seed was run for retrieval because the index rebuild is expensive.
- The hybrid evaluation uses a fixed 3,000-query sample; dense metrics use all grouped test rows.
- Token-F1 is still single-reference; multi-reference evaluation is Batch B.
- The corpus metadata has no district/date field, so temporal or district-held-out evaluation is not possible.
