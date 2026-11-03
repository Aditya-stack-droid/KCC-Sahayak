"""Shared helpers for Batch B.

Batch B is additive: no original file is written. This module only defines
paths, hashing helpers and small JSON utilities used by the phase scripts.
"""
from __future__ import annotations

import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BATCH_B = ROOT / "experiments" / "batch_b"
MANIFESTS = BATCH_B / "manifests"
BATCH_A_RESULTS = ROOT / "results" / "improved_evaluation"

# Seeds are fixed here so every Batch B experiment can record them.
SEEDS = {
    "annotation_split": [0, 1, 2, 3, 4],
    "retrieval_sample": 42,
    "bootstrap": 0,
}

PHASES = [
    "Phase 0 - baseline verification",
    "Phase 1 - PII detection and redaction",
    "Phase 2 - human annotation infrastructure",
    "Phase 3 - independent safety evaluation",
    "Phase 4 - answer conflict detection",
    "Phase 5 - out-of-scope and multi-intent detection",
    "Phase 6 - extended retrieval evaluation",
    "Phase 7 - behavioural and regression testing",
    "Phase 8 - documentation and comparison",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def rel(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def write_json(path: Path, obj, overwrite: bool = False) -> Path:
    """Write JSON, refusing to clobber an existing file unless asked."""
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")
    return path


def environment() -> dict:
    versions = {}
    for name in ("numpy", "pandas", "sklearn", "scipy", "sentence_transformers",
                 "faiss", "joblib", "torch", "transformers", "pytest"):
        try:
            mod = __import__(name)
            versions[name] = getattr(mod, "__version__", "unknown")
        except Exception:
            versions[name] = "not importable"
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "versions": versions,
    }
