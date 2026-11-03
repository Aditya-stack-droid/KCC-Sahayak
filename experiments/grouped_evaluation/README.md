# Grouped evaluation (Batch A)

Standalone, leakage-resistant evaluation for KCC-Sahayak. It reads the original
corpus, annotations, model artefacts and results, and writes **only** new files
under `experiments/grouped_evaluation/artifacts/` and
`results/improved_evaluation/`. No original file is modified.

## Why

The original split is row-level: 44.98% of test questions appear verbatim in
train and 70.73% have a >=0.95 near-duplicate there. Every random-row metric is
therefore an upper bound on performance for unseen farmer questions.

## Protocol

1. **Grouping** (`02_build_groups.py`): identical normalised questions are
   merged; then a FAISS k-NN scan (k=64) over MiniLM embeddings merges any pair
   with cosine >= 0.95, transitively, via union-find.
2. **Splits** (`03_make_grouped_split.py`):
   - corpus: 10-fold `StratifiedGroupKFold` (seed 0), fold 0 = test, fold 1 =
     validation, rest = train;
   - annotated set: 7-fold `StratifiedGroupKFold`, fold 0 = test, seeds 0-4;
   - paraphrases are attached to their source query's side.
3. **Intent** (`04_evaluate_grouped_intent.py`): refit the documented deployed
   recipe on the grouped train side, with and without paraphrases; evaluate on
   the grouped test side with bootstrap CIs, per-class metrics, confusion
   matrix and novelty buckets. The shipped artefact is also scored, with the
   share of test rows it saw during training reported explicitly.
4. **Retrieval** (`05_evaluate_grouped_retrieval.py`): rebuild the dense index
   and TF-IDF matrix on the grouped train+val side; evaluate dense and hybrid
   retrieval (alpha=0.4) with topic match, Recall@k, MRR, nDCG and the original
   single-reference token-F1.

## Run

```bash
OMP_NUM_THREADS=1 .venv312/bin/python experiments/grouped_evaluation/01_prepare_embeddings.py
OMP_NUM_THREADS=1 .venv312/bin/python experiments/grouped_evaluation/02_build_groups.py
OMP_NUM_THREADS=1 .venv312/bin/python experiments/grouped_evaluation/03_make_grouped_split.py
OMP_NUM_THREADS=1 .venv312/bin/python experiments/grouped_evaluation/04_evaluate_grouped_intent.py
OMP_NUM_THREADS=1 .venv312/bin/python experiments/grouped_evaluation/05_evaluate_grouped_retrieval.py
```

`01` reuses the shipped embeddings by `qid` and encodes only the 13,888 rows
the shipped index does not cover.

## What this does and does not fix

- Removes **split leakage** (exact and >=0.95 near-duplicate overlap).
- Does **not** remove annotator bias: labels remain LLM-generated. Human
  verification remains PENDING.
- Relevance for retrieval is the weak-label topic match, the same proxy the
  original evaluation used. It is not agronomic correctness.
