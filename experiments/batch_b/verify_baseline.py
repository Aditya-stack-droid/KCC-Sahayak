#!/usr/bin/env python
"""Create and verify the Batch B baseline manifest.

The manifest records SHA-256 hashes and sizes for:

* Batch A outputs, code, artefacts and research notes (immutable baseline);
* protected original data, models, source, backend, scripts, documents and
  existing tests.

Usage:
    python experiments/batch_b/verify_baseline.py --init
    python experiments/batch_b/verify_baseline.py --check

``--check`` exits non-zero if any listed file is missing or changed, or if a
new file appears in an immutable directory. It never writes to the repository
(only ``--init`` writes the manifest).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import MANIFESTS, ROOT, environment, rel, sha256_file, utc_now  # noqa: E402

MANIFEST_PATH = MANIFESTS / "baseline_hashes.json"

EXCLUDE_PARTS = {"__pycache__", ".pytest_cache", ".DS_Store", ".git"}
EXCLUDE_SUFFIXES = (".pyc", ".pyo")

PROTECTED_DIRS = [
    "data",
    "models",
    "src",
    "backend",
    "scripts",
    "experiments/grouped_evaluation",
    "results/improved_evaluation",
]
PROTECTED_TREES = ["paper", "docs", "frontend"]
PROTECTED_FILES = [
    "README.md",
    "FINAL_REPORT.md",
    "KCC_Sahayak_Verification_Audit.md",
    "this_is_it.md",
    "requirements.txt",
    ".gitignore",
    "run_app.bat",
    "tests/test_api.py",
    "tests/test_gate.py",
    "tests/test_text_utils.py",
    "tests/unit/test_grouped_evaluation.py",
    "research/IMPROVEMENT_BASELINE.md",
    "research/EXPERIMENT_LOG.md",
]
IMMUTABLE_DIRS = [
    "data",
    "models",
    "src",
    "backend",
    "scripts",
    "experiments/grouped_evaluation",
    "results/improved_evaluation",
]
BATCH_A_PREFIXES = (
    "experiments/grouped_evaluation/",
    "results/improved_evaluation/",
)


def excluded(path: Path) -> bool:
    if any(part in EXCLUDE_PARTS for part in path.parts):
        return True
    return path.name.endswith(EXCLUDE_SUFFIXES)


def group_for(relative: str) -> str:
    if relative.startswith(BATCH_A_PREFIXES):
        return "batch_a"
    if relative in {
        "research/IMPROVEMENT_BASELINE.md",
        "research/EXPERIMENT_LOG.md",
        "tests/unit/test_grouped_evaluation.py",
    }:
        return "batch_a"
    return "protected_original"


def collect_files() -> dict[str, Path]:
    files: dict[str, Path] = {}

    def add(path: Path) -> None:
        if path.is_file() and not excluded(path):
            files[rel(path)] = path

    for directory in PROTECTED_DIRS:
        base = ROOT / directory
        if base.exists():
            for path in base.rglob("*"):
                add(path)
    for directory in PROTECTED_TREES:
        base = ROOT / directory
        if base.exists():
            for path in base.rglob("*"):
                add(path)
    for pattern in ("results/*", "demo/*", "demo/voice_eval/*"):
        for path in ROOT.glob(pattern):
            add(path)
    for relative in PROTECTED_FILES:
        add(ROOT / relative)
    return files


def build_manifest() -> dict:
    files = collect_files()
    entries = {}
    for relative, path in sorted(files.items()):
        entries[relative] = {
            "sha256": sha256_file(path),
            "size": path.stat().st_size,
            "group": group_for(relative),
        }
    return {
        "schema": 1,
        "created_at": utc_now(),
        "phase": "batch_b_phase0",
        "environment": environment(),
        "immutable_dirs": IMMUTABLE_DIRS,
        "files": entries,
    }


def init(force: bool = False) -> int:
    if MANIFEST_PATH.exists() and not force:
        print(f"manifest already exists: {rel(MANIFEST_PATH)} (use --force to rebuild)")
        return 1
    manifest = build_manifest()
    MANIFESTS.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    groups = {}
    for entry in manifest["files"].values():
        groups[entry["group"]] = groups.get(entry["group"], 0) + 1
    print(f"wrote {rel(MANIFEST_PATH)}")
    print(f"files: {len(manifest['files'])}  {groups}")
    return 0


def check() -> int:
    if not MANIFEST_PATH.exists():
        print(f"missing manifest: {rel(MANIFEST_PATH)}; run --init first")
        return 2
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    entries = manifest["files"]
    changed, missing, unreadable = [], [], []

    for relative, meta in sorted(entries.items()):
        path = ROOT / relative
        if not path.exists():
            missing.append(relative)
            continue
        try:
            if path.stat().st_size != meta["size"]:
                changed.append((relative, "size", meta["size"], path.stat().st_size))
                continue
            digest = sha256_file(path)
        except OSError as exc:
            unreadable.append((relative, str(exc)))
            continue
        if digest != meta["sha256"]:
            changed.append((relative, "sha256", meta["sha256"][:12], digest[:12]))

    # New files in immutable directories are also a violation.
    current = set(collect_files())
    listed = set(entries)
    additions = []
    for directory in manifest.get("immutable_dirs", []):
        prefix = directory.rstrip("/") + "/"
        for relative in sorted(current - listed):
            if relative == directory or relative.startswith(prefix):
                additions.append(relative)

    print(f"manifest: {rel(MANIFEST_PATH)}")
    print(f"checked:  {len(entries)} files")
    print(f"changed:  {len(changed)}")
    print(f"missing:  {len(missing)}")
    print(f"unexpected additions in immutable dirs: {len(additions)}")
    for relative, kind, old, new in changed[:20]:
        print(f"  CHANGED {relative} ({kind}: {old} -> {new})")
    for relative in missing[:20]:
        print(f"  MISSING {relative}")
    for relative in additions[:20]:
        print(f"  ADDED   {relative}")
    for relative, error in unreadable[:20]:
        print(f"  ERROR   {relative}: {error}")

    ok = not (changed or missing or additions or unreadable)
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--init", action="store_true", help="write the manifest")
    group.add_argument("--check", action="store_true", help="verify against the manifest")
    parser.add_argument("--force", action="store_true", help="rebuild an existing manifest")
    args = parser.parse_args()
    return init(force=args.force) if args.init else check()


if __name__ == "__main__":
    raise SystemExit(main())
