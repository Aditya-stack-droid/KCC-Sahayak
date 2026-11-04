# Leakage audit

Descriptive recomputation of row-level leakage from the committed splits
(`data/processed/{train,val,test}.csv`); no model, no embeddings.

Run: `.venv312/bin/python experiments/leakage_audit/audit.py --overwrite`

Findings (see `results/leakage_audit/REPORT.md`):

| Measure | Value |
|---|---:|
| Test question exactly in train | 6,247 (44.98%) |
| Test question in index (train+val) | 46.34% |
| Test answer appears in index | 36.03% |
| Full test record in index | 0.00% |
| Test-internal duplicate questions | 1,948 |
| Grouped split near-duplicate (>=0.95), Batch A | 0.09% |

Interpretation: row-level metrics overstate generalisation because the same question (often
an exact string) appears on both sides. The grouped split (Batch A) is the leakage-resistant
protocol. These are descriptive counts, not model measurements.
