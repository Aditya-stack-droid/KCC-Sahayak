"""Improvement 3: multi-reference retrieval evaluation.

Additive experiment package. It reads the original corpus, the Batch A grouped
split/index artefacts and the existing embedding model, and writes only under
``experiments/multi_reference/artifacts/`` and ``results/multi_reference/``.

Nothing here changes production retrieval behaviour, and no reference answer is
invented: every reference is a historical advisor answer already present in
``data/processed/full_labeled.csv``.
"""
