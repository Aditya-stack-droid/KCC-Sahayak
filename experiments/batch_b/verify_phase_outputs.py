#!/usr/bin/env python
"""Create and verify the Batch B protection baseline.

This is additive: it does not modify ``baseline_hashes.json`` or any original
file.  A snapshot combines

* every file already recorded in ``manifests/baseline_hashes.json``
  (the original project and Batch A), and
* the Batch B Phase 1 (privacy) and Phase 2 (annotation) code, data, tests and
  result artefacts.

Usage
-----
    python experiments/batch_b/verify_phase_outputs.py --snapshot
    python experiments/batch_b/verify_phase_outputs.py --check
    python experiments/batch_b/verify_phase_outputs.py --snapshot after_phase3

``--check`` exits non-zero if a listed file is missing or changed, or if a new
file appears inside a protected directory.  Annotation shards, adjudication and
export files are exempt from the "new file" rule because they are the expected
future human data for Phase 2; listings in the manifest itself are still checked
when they exist.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import MANIFESTS, ROOT, environment, rel, sha256_file, utc_now  # noqa: E402

BASELINE = MANIFESTS / "baseline_hashes.json"
DEFAULT_NAME = "batch_b_protection_hashes"

PHASE_PATTERNS = [
    "experiments/privacy/**/*",
    "results/privacy_audit/**/*",
    "experiments/annotation/**/*",
    "results/annotation/**/*",
    "experiments/conflict_detection/**/*",
    "results/conflict_detection/**/*",
    "tests/unit/test_privacy.py",
    "tests/unit/test_annotation.py",
    "tests/unit/test_conflict_detection.py",
]

IMMUTABLE_PREFIXES = [
    "data/",
    "models/",
    "src/",
    "backend/",
    "scripts/",
    "experiments/grouped_evaluation/",
    "results/improved_evaluation/",
    "experiments/privacy/",
    "results/privacy_audit/",
    "experiments/annotation/",
    "results/annotation/",
    "experiments/conflict_detection/",
    "results/conflict_detection/",
]

ADDITION_EXEMPT_PREFIXES = [
    "experiments/annotation/data/annotations/",
    "experiments/annotation/data/adjudication.jsonl",
    "experiments/annotation/data/exports/",
]

EXCLUDE_PARTS = {"__pycache__", ".pytest_cache", ".DS_Store", ".git"}
EXCLUDE_SUFFIXES = (".pyc", ".pyo")


def excluded(path: Path) -> bool:
    if any(part in EXCLUDE_PARTS for part in path.parts):
        return True
    return path.name.endswith(EXCLUDE_SUFFIXES)


def group_for(relative: str, baseline_groups: dict[str, str] | None = None) -> str:
    if relative.startswith(("experiments/grouped_evaluation/", "results/improved_evaluation/")):
        return "batch_a"
    if relative.startswith(("experiments/privacy/", "results/privacy_audit/")) or relative == "tests/unit/test_privacy.py":
        return "batch_b_phase1"
    if relative.startswith(("experiments/annotation/", "results/annotation/")) or relative == "tests/unit/test_annotation.py":
        return "batch_b_phase2"
    if relative.startswith(("experiments/conflict_detection/", "results/conflict_detection/")) or relative == "tests/unit/test_conflict_detection.py":
        return "batch_b_improvement1"
    if baseline_groups and relative in baseline_groups:
        return baseline_groups[relative]
    return "protected_original"


def collect_files() -> dict[str, Path]:
    if not BASELINE.exists():
        raise SystemExit(f"missing {rel(BASELINE)}; run experiments/batch_b/verify_baseline.py --init")
    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
    files: dict[str, Path] = {}
    for relative in baseline["files"]:
        path = ROOT / relative
        if path.is_file() and not excluded(path):
            files[relative] = path
    for pattern in PHASE_PATTERNS:
        for path in ROOT.glob(pattern):
            if path.is_file() and not excluded(path):
                files[rel(path)] = path
    return files


def build_manifest(name: str) -> dict:
    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
    baseline_groups = {
        relative: meta.get("group", "protected_original")
        for relative, meta in baseline["files"].items()
    }
    files = collect_files()
    entries = {
        relative: {
            "sha256": sha256_file(path),
            "size": path.stat().st_size,
            "group": group_for(relative, baseline_groups),
        }
        for relative, path in sorted(files.items())
    }
    groups: dict[str, int] = {}
    for entry in entries.values():
        groups[entry["group"]] = groups.get(entry["group"], 0) + 1
    return {
        "schema": 1,
        "name": name,
        "created_at": utc_now(),
        "phase": "batch_b_protection",
        "environment": environment(),
        "baseline_manifest": rel(BASELINE),
        "immutable_prefixes": IMMUTABLE_PREFIXES,
        "addition_exempt_prefixes": ADDITION_EXEMPT_PREFIXES,
        "groups": groups,
        "files": entries,
    }


def manifest_path(name: str) -> Path:
    return MANIFESTS / f"{name}.json"


def snapshot(name: str, force: bool = False) -> int:
    path = manifest_path(name)
    if path.exists() and not force:
        print(f"snapshot already exists: {rel(path)} (use --force to rebuild)")
        return 1
    manifest = build_manifest(name)
    MANIFESTS.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"wrote {rel(path)}")
    print(f"files: {len(manifest['files'])}  groups: {manifest['groups']}")
    return 0


def check(name: str, allow_additions: bool = False) -> int:
    path = manifest_path(name)
    if not path.exists():
        print(f"missing snapshot: {rel(path)}; run --snapshot {name} first")
        return 2
    manifest = json.loads(path.read_text(encoding="utf-8"))
    entries = manifest["files"]
    changed, missing, unreadable = [], [], []

    for relative, meta in sorted(entries.items()):
        file_path = ROOT / relative
        if not file_path.exists():
            missing.append(relative)
            continue
        try:
            if file_path.stat().st_size != meta["size"]:
                changed.append((relative, "size", meta["size"], file_path.stat().st_size))
                continue
            digest = sha256_file(file_path)
        except OSError as exc:
            unreadable.append((relative, str(exc)))
            continue
        if digest != meta["sha256"]:
            changed.append((relative, "sha256", meta["sha256"][:12], digest[:12]))

    current = collect_files()
    additions = []
    exempt = []
    for relative in sorted(set(current) - set(entries)):
        if any(relative.startswith(prefix) for prefix in manifest.get("addition_exempt_prefixes", [])):
            exempt.append(relative)
        elif any(relative.startswith(prefix) for prefix in manifest.get("immutable_prefixes", [])):
            additions.append(relative)

    print(f"snapshot: {rel(path)}")
    print(f"checked:  {len(entries)} files  groups: {manifest.get('groups', {})}")
    print(f"changed:  {len(changed)}")
    print(f"missing:  {len(missing)}")
    print(f"unexpected additions in protected dirs: {len(additions)}")
    if exempt:
        print(f"expected future annotation files (not failures): {len(exempt)}")
    for relative, kind, old, new in changed[:20]:
        print(f"  CHANGED {relative} ({kind}: {old} -> {new})")
    for relative in missing[:20]:
        print(f"  MISSING {relative}")
    for relative in additions[:20]:
        print(f"  ADDED   {relative}")
    for relative, error in unreadable[:20]:
        print(f"  ERROR   {relative}: {error}")

    if additions and allow_additions:
        print("NOTE: additions are allowed for this run (approved new improvement files); "
              "changed/missing files are still failures.")
    ok = not (changed or missing or unreadable or (additions and not allow_additions))
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--snapshot", nargs="?", const=DEFAULT_NAME, metavar="NAME",
                        help=f"write a snapshot (default {DEFAULT_NAME})")
    action.add_argument("--check", nargs="?", const=DEFAULT_NAME, metavar="NAME",
                        help=f"verify against a snapshot (default {DEFAULT_NAME})")
    parser.add_argument("--force", action="store_true", help="rebuild an existing snapshot")
    parser.add_argument("--allow-additions", action="store_true",
                        help="permit new approved files while still failing on changed/missing files")
    args = parser.parse_args()
    if args.snapshot is not None:
        return snapshot(args.snapshot, force=args.force)
    return check(args.check, allow_additions=args.allow_additions)


if __name__ == "__main__":
    raise SystemExit(main())
