"""Batch B Phase 2 — human annotation infrastructure.

Additive only: this package reads the existing corpus and writes exclusively
under ``experiments/annotation/`` and ``results/annotation/``.  It never edits
the original project, Batch A or Phase 1 outputs.
"""

__all__ = ["schema", "sampling", "store", "agreement", "quality", "ui", "report"]
