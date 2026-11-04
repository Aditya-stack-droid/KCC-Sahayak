# Dataset leakage audit

**Evidence class:** descriptive recomputation from the committed splits; no model, no training.

**Sizes:** train 111,100, val 13,887, test 13,888, retrieval index 124,987.

## Row-split leakage (this split)

| Measure | Count | Share of test |
|---|---:|---:|
| Test question exactly in train | 6,247 | 44.98% |
| Test question exactly in val | 3,053 | 21.98% |
| Test question in index (train+val) | 6436 | 46.34% |
| Test answer appears in index | 5,004 | 36.03% |
| Full test record appears in index | 0 | 0.00% |

Test-internal duplicate questions: 1,948.

## Grouped split (Batch A reference)

- Near-duplicate (>=0.95) share: 0.00094
- Near-duplicate (>=0.90) share: 0.699371
- Novel (<0.80) share: 0.061456

## Interpretation

- Row-level metrics overstate generalisation because the same question (often an exact string) appears in train and test.
- These are descriptive counts on the shipped splits, not model measurements.
- The grouped split (Batch A) removes near-duplicate leakage by construction and is the leakage-resistant protocol.
