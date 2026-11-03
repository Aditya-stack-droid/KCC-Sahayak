"""Batch B Improvement 1 — deterministic answer conflict detection.

Additive only: reads existing corpus/artefacts and writes exclusively under
``experiments/conflict_detection/`` and ``results/conflict_detection/``.
The production safety gate is imported read-only and never modified.
"""

__all__ = ["units", "products", "conflicts", "gate"]
