# Experiment log

**Rules:** append-only; every number comes from an executable run; original
result files are never overwritten; failed or partial runs are recorded, not
hidden; human/expert work that has not happened is marked PENDING.

---

## 2026-11-03 — Batch A: grouped, leakage-resistant evaluation

**Status:** COMPLETED (fully offline).
**Purpose:** replace the row-level split with question-group splits and measure
how much of the reported performance depended on exact/near-duplicate leakage.
**Original files:** read-only. No file under `data/`, `models/`, `src/`,
`backend/`, `scripts/`, `results/*.json` (pre-existing), `paper/` or `docs/`
was modified.

### Environment

- macOS, Apple M3, 8 GB RAM, CPU only; Python 3.12.13 (`.venv312`).
- sentence-transformers 6.2.0, faiss 1.15.1, scikit-learn 1.9.1, pandas 3.1.0.
- `OMP_NUM_THREADS=1` for all runs (macOS PyTorch/FAISS OpenMP caution).
- Tests: **52 passed** (41 original + 11 new; no original test altered).

### Commands

```bash
OMP_NUM_THREADS=1 .venv312/bin/python experiments/grouped_evaluation/01_prepare_embeddings.py
OMP_NUM_THREADS=1 .venv312/bin/python experiments/grouped_evaluation/02_build_groups.py
OMP_NUM_THREADS=1 .venv312/bin/python experiments/grouped_evaluation/03_make_grouped_split.py
OMP_NUM_THREADS=1 .venv312/bin/python experiments/grouped_evaluation/04_evaluate_grouped_intent.py
OMP_NUM_THREADS=1 .venv312/bin/python experiments/grouped_evaluation/05_evaluate_grouped_retrieval.py
OMP_NUM_THREADS=1 .venv312/bin/python experiments/grouped_evaluation/06_write_report.py
```

### Protocol

- **Grouping:** exact normalised question match + MiniLM cosine ≥ 0.95
  (FAISS k-NN, k=64), merged transitively with union-find.
- **Corpus split:** 10-fold `StratifiedGroupKFold` (seed 0); fold 0 = test,
  fold 1 = validation, remainder = train.
- **Annotated split:** 7-fold `StratifiedGroupKFold` on the LLM meaning labels;
  fold 0 = test; seeds 0–4. Paraphrases follow their source query's side.
- **Intent model:** documented deployed recipe — MiniLM embedding + `char_wb`
  2–5 TF-IDF → LogisticRegression (C=10, embedding weight 1.0), refit on the
  grouped training side; with and without paraphrases.
- **Retrieval:** fresh FAISS `IndexFlatIP` and word TF-IDF matrix over the
  grouped train+val side; dense and hybrid (α=0.4) scoring.
- **Relevance proxy:** retrieved weak keyword label == test weak label, the same
  proxy the original evaluation used. This is topical, not agronomic.

### 1. Grouping and leakage

| Measure | Original row split | Grouped split |
|---|---:|---:|
| Corpus rows | 138,875 | 138,875 |
| Question groups | — | **48,674** (largest 1,623; p95 7) |
| Test questions exactly in train | 44.98% | 0% by construction |
| Test query shares a group with train | **70.82%** | 0% |
| Test query has ≥0.95 dense neighbour in index | 71.38% (recomputed) / 70.73% (stored) | **0.09%** (13 of 13,831) |
| Test query has ≥0.90 neighbour in index | 83.32% | 69.94% |
| Novel test queries (<0.80) | 4.75% | 6.15% |

The 0.09% residual ≥0.95 share after grouping is disclosed: the k-NN candidate
cap (k=64) and the transitive 0.95 threshold can leave a small number of pairs
unmerged. It is not zero, but two orders of magnitude below the original 70.7%.

### 2. Intent classification (mean ± sd over 5 grouped splits)

| Model | Accuracy | Macro F1 | Weighted F1 |
|---|---:|---:|---:|
| Refit, embedding + char TF-IDF + LR (with paraphrases) | **0.9372 ± 0.0045** | 0.9205 ± 0.0048 | 0.9358 ± 0.0047 |
| Refit, same recipe without paraphrases | **0.9372 ± 0.0054** | — | — |
| Shipped artefact on the same test rows | 0.9924 (0.984–0.996) | — | — |

Novelty buckets (refit with paraphrases, mean over seeds):

| Bucket | N (seed 0) | Accuracy |
|---|---:|---:|
| Near-duplicate (≥0.90) | 78 | 0.9630 ± 0.0219 |
| Similar (0.80–0.90) | 236 | 0.9580 ± 0.0122 |
| Novel (<0.80) | 186 | 0.9025 ± 0.0107 |

Independent-phrasing gold set (refit, per seed): 0.8933, 0.8933, 0.8800,
0.8733, 0.8800 — **≈0.884 mean**, versus 0.90 on the original split.

**Important:** 83.6–87.8% of the grouped-test rows were in the shipped
artefact's original training pool, so its 0.992 is a memorisation upper bound,
not a generalisation estimate. The refit rows are the primary estimate.

### 3. Retrieval (grouped index: 125,044 rows; test: 13,831)

| Metric | Grouped dense | Original (leaky) reference |
|---|---:|---:|
| Topic match @1 | **0.8519** (95% CI 0.8462–0.8576) | 0.933 |
| Recall @5 | 0.9508 | — |
| MRR @10 | 0.8961 | — |
| nDCG @5 | 0.9071 | — |
| Answer token-F1 @1 (all) | 0.3213 | 0.4545 (dense) |
| Topic match, near-duplicate bucket | 0.8702 | 0.9807 |
| Topic match, novel bucket | 0.7812 | 0.7365 |

Dense vs hybrid on the same fixed 3,000-query grouped sample:

| System | Topic match @1 | Recall @5 | MRR @10 | Answer token-F1 |
|---|---:|---:|---:|---:|
| Dense | 0.8403 | 0.9450 | 0.8881 | 0.3220 |
| Hybrid (α=0.4) | **0.8733** | 0.9630 | 0.9132 | **0.3800** |

### Findings

1. **The reported retrieval number was inflated by leakage.** Top-1 topic match
   falls from 0.933 to 0.8519, and near-duplicate-bucket match from 0.9807 to
   0.8702. The original figure described performance on a split where 70.7% of
   test queries had a ≥0.95 neighbour in the index.
2. **Intent accuracy is largely robust to the enforced group split** on this
   labelled pool: 0.9372 ± 0.0045 grouped versus 0.9355 stored. Under the
   grouped split, near-duplicate test rows score 0.9630 and novel rows 0.9025;
   the gap is about six points, much smaller than the retrieval gap. The
   annotated pool's duplicate structure is weaker than the corpus's, which is
   *consistent with* its 496-row test having been sampled separately, but the
   original annotated train↔test overlap was not re-measured in this batch, so
   this explanation is a hypothesis, not a measured decomposition.
3. **Paraphrase augmentation did not change in-domain accuracy** under the
   grouped split (0.9372 with vs 0.9372 without). The previously reported
   augmentation benefit was on the independent-phrasing gold set; a grouped
   no-augmentation gold comparison was not run in this batch and is not claimed.
4. **Hybrid retrieval still helps** under the stricter split: +3.3 points topic
   match and +5.8 points answer token-F1 over dense on the same sample. This is
   a paired comparison on one sample, not a significance test.
5. **Novel-query retrieval remains the weak point**: topic match 0.7812 and
   answer F1 0.2325, despite Recall@5 of 0.9365 — the right topic is often in
   the top 5 but not at rank 1, and the single logged answer is a weak target.

### Limitations

- Labels, gold queries and paraphrases remain LLM-generated; leakage is removed,
  annotator bias is not. Human verification is PENDING.
- The gold set shares its author with the paraphrases; its ≈0.884 remains an
  optimistic estimate.
- Retrieval relevance is the weak-label topic proxy, not factual correctness.
- One corpus split seed for retrieval; five seeds for intent.
- The 0.09% residual ≥0.95 neighbour share is a known artifact of the k=64 cap.
- No district/date metadata exists, so temporal and district-held-out evaluation
  remains impossible without new data.
- The grouped annotated test is ~500 rows per seed; confidence intervals remain
  wide for per-class and per-bucket estimates.

### Artefacts

- `experiments/grouped_evaluation/artifacts/` — 213 MB corpus embeddings
  (124,987 reused + 13,888 encoded), groups, splits, grouped FAISS index.
- `results/improved_evaluation/grouped_intent.json`
- `results/improved_evaluation/grouped_retrieval.json`
- `results/improved_evaluation/GROUPED_INTENT_REPORT.md`
- `results/improved_evaluation/GROUPED_RETRIEVAL_REPORT.md`
- `results/improved_evaluation/REPORT.md` (combined)
- Logs: `experiments/grouped_evaluation/logs/`

### Next

Batch B: answer consolidation and conflict detection, multi-reference
retrieval metrics, BM25/cross-encoder baselines. Human annotation, expert
safety review and real speech remain PENDING.
