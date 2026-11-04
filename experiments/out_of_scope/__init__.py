"""Improvement 4: out-of-scope query detection.

Additive experiment package. It reads the original corpus, the Batch A grouped
split and the existing multilingual encoder, and writes only under
``experiments/out_of_scope/`` and ``results/out_of_scope/``.

Nothing here changes the production intent classifier, retrieval pipeline or
safety gate. The synthetic out-of-scope queries are authored for this study,
are marked ``synthetic=True`` everywhere, and are never presented as real
farmer data.
"""
