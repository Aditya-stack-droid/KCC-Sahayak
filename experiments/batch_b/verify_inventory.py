#!/usr/bin/env python
"""Data-driven protection inventory.

Why this exists
---------------
``verify_phase_outputs.py`` hard-codes the Protected phases and is itself part of
the pre-Improvement-1 snapshot, so extending its scan list would report its own
file as "changed" and invalidate that snapshot. This tool keeps the scan
definition in a JSON config instead, so protection coverage can grow without
editing code that is already inside a frozen manifest.

The manifest is built from:

* every file in the original/Batch A ``baseline_hashes.json``,
* every file under the configured roots (Phase 1, Phase 2, Improvement 1,
  Improvement 2),
* the explicitly listed files (focused tests and this config).

``--check`` exits non-zero on any missing or changed file, or on an unexpected
new file inside a protected directory. ``--allow-additions`` is only for a run
that intentionally adds approved files; changed/missing files still fail.

Usage
-----
    python experiments/batch_b/verify_inventory.py --snapshot
    python experiments/batch_b/verify_inventory.py --check
    python experiments/batch_b/verify_inventory.py --check --allow-additions
    python experiments/batch_b/verify_inventory.py --snapshot other_name --force
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from common import MANIFESTS, environment, rel, sha256_file, utc_now  # noqa: E402

DEFAULT_CONFIG = HERE / "inventories" / "protection.json"
EXCLUDE_PARTS = {"__pycache__", ".pytest_cache", ".DS_Store", ".git"}
EXCLUDE_SUFFIXES = (".pyc", ".pyo")


def excluded(path: Path) -> bool:
    if any(part in EXCLUDE_PARTS for part in path.parts):
        return True
    return path.name.endswith(EXCLUDE_SUFFIXES)


def load_config(path: Path) -> dict:
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    for key in ("name", "baseline_manifest", "roots", "files", "immutable_prefixes"):
        if key not in config:
            raise SystemExit(f"config missing '{key}': {path}")
    return config


def explicit_files(config: dict) -> list[dict]:
    """Normalise the config's explicit file list to {path, group} entries."""
    entries = []
    for item in config["files"]:
        if isinstance(item, str):
            entries.append({"path": item, "group": "protected_original"})
        else:
            entries.append({"path": item["path"], "group": item.get("group", "protected_original")})
    return entries


def group_for(relative: str, config: dict, baseline_groups: dict[str, str]) -> str:
    for item in explicit_files(config):
        if relative == item["path"]:
            return item["group"]
    for root in config["roots"]:
        if relative.startswith(root["prefix"]):
            return root["group"]
    return baseline_groups.get(relative, "protected_original")


def collect(config: dict) -> tuple[dict[str, Path], dict[str, str]]:
    baseline_path = ROOT / config["baseline_manifest"]
    if not baseline_path.exists():
        raise SystemExit(f"missing baseline manifest: {config['baseline_manifest']}")
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    baseline_groups = {
        relative: meta.get("group", "protected_original")
        for relative, meta in baseline["files"].items()
    }
    files: dict[str, Path] = {}
    for relative in baseline["files"]:
        path = ROOT / relative
        if path.is_file() and not excluded(path):
            files[relative] = path
    for root in config["roots"]:
        for path in (ROOT / root["prefix"]).glob("**/*"):
            if path.is_file() and not excluded(path):
                files[rel(path)] = path
    for item in explicit_files(config):
        relative = item["path"]
        path = ROOT / relative
        if path.is_file() and not excluded(path):
            files[relative] = path
    return files, baseline_groups


def manifest_path(name: str) -> Path:
    return MANIFESTS / f"{name}.json"


def build(config: dict, name: str) -> dict:
    files, baseline_groups = collect(config)
    entries = {
        relative: {
            "sha256": sha256_file(path),
            "size": path.stat().st_size,
            "group": group_for(relative, config, baseline_groups),
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
        "phase": "batch_b_protection_full",
        "environment": environment(),
        "config": rel(DEFAULT_CONFIG),
        "baseline_manifest": config["baseline_manifest"],
        "immutable_prefixes": config["immutable_prefixes"],
        "addition_exempt_prefixes": config.get("addition_exempt_prefixes", []),
        "groups": groups,
        "files": entries,
    }


def snapshot(config: dict, name: str, force: bool) -> int:
    path = manifest_path(name)
    if path.exists() and not force:
        print(f"snapshot already exists: {rel(path)} (use --force to rebuild)")
        return 1
    manifest = build(config, name)
    MANIFESTS.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"wrote {rel(path)}")
    print(f"files: {len(manifest['files'])}  groups: {manifest['groups']}")
    return 0


def check(config: dict, name: str, allow_additions: bool) -> int:
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

    current, _ = collect(config)
    additions, exempt = [], []
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
        print("NOTE: additions are allowed for this run (approved new files); "
              "changed/missing files are still failures.")
    ok = not (changed or missing or unreadable or (additions and not allow_additions))
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--snapshot", nargs="?", const="", metavar="NAME",
                        help="write a snapshot (default: config name)")
    action.add_argument("--check", nargs="?", const="", metavar="NAME",
                        help="verify against a snapshot (default: config name)")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--force", action="store_true", help="rebuild an existing snapshot")
    parser.add_argument("--allow-additions", action="store_true",
                        help="permit new approved files while still failing on changed/missing files")
    args = parser.parse_args()

    config = load_config(Path(args.config))
    if args.snapshot is not None:
        return snapshot(config, args.snapshot or config["name"], args.force)
    return check(config, args.check or config["name"], args.allow_additions)


if __name__ == "__main__":
    raise SystemExit(main())
