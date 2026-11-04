"""Batch B Improvement 5 — behavioural regression testing.

Additive only: exercises existing production and Batch B components read-only,
and writes exclusively under ``results/behavioural_regression/``. No model is
trained and no production file, dataset or previous result is modified.
"""

__all__ = ["br_cases", "br_probes", "br_report", "run_evaluation"]
