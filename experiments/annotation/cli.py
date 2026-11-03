#!/usr/bin/env python
"""Command line workflow for the Batch B human annotation study.

Examples
--------
    python experiments/annotation/cli.py build-sample
    python experiments/annotation/cli.py make-ui --annotator A001 A002
    python experiments/annotation/cli.py import --annotator A001 sheet.json
    python experiments/annotation/cli.py status
    python experiments/annotation/cli.py validate
    python experiments/annotation/cli.py agreement
    python experiments/annotation/cli.py adjudicate --list
    python experiments/annotation/cli.py report --overwrite

The tool never creates labels itself and never resolves a disagreement.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.annotation import agreement as agreement_mod  # noqa: E402
from experiments.annotation import quality as quality_mod  # noqa: E402
from experiments.annotation import report as report_mod  # noqa: E402
from experiments.annotation import sampling, schema  # noqa: E402
from experiments.annotation import store as store_mod  # noqa: E402
from experiments.annotation import ui as ui_mod  # noqa: E402

DEFAULT_OUT = ROOT / "results" / "annotation"


def _store(args) -> store_mod.Store:
    return store_mod.Store(Path(args.data_dir))


def cmd_build_sample(args) -> int:
    store = _store(args)
    targets = [store.items_path, store.key_path, store.summary_path]
    existing = [str(path) for path in targets if path.exists()]
    if existing and not args.overwrite:
        print("sample already exists; refusing to replace a frozen sample:")
        for path in existing:
            print(f"  {path}")
        print("pass --overwrite only if you intentionally restart the study")
        return 1

    items, key, summary = sampling.build_sample(n=args.n, seed=args.seed)
    store_mod.write_jsonl(store.items_path, items, overwrite=args.overwrite)
    store_mod.write_jsonl(store.key_path, key, overwrite=args.overwrite)
    store_mod.write_json(store.summary_path, summary, overwrite=args.overwrite)
    print(json.dumps({
        "n_items": len(items),
        "n_candidates": sum(len(item["candidates"]) for item in items),
        "realised_strata": summary["realised_strata"],
        "shortfall_by_stratum": summary["shortfall_by_stratum"],
        "existing_llm_labels": summary["sample_label_counts"],
        "items": str(store.items_path),
        "key": str(store.key_path),
    }, indent=2))
    return 0


def cmd_make_ui(args) -> int:
    store = _store(args)
    written = []
    for annotator in args.annotator:
        try:
            path = ui_mod.write_ui(store, annotator, overwrite=args.overwrite)
        except FileExistsError as exc:
            print(f"error: {exc}")
            return 1
        written.append(str(path))
    print(json.dumps({"written": written,
                      "note": "send each file only to its annotator; key.jsonl stays with the researcher"},
                     indent=2))
    return 0


def cmd_status(args) -> int:
    store = _store(args)
    items = store.load_items()
    latest = store.latest()
    annotators = sorted({record["annotator"] for record in latest.values()})
    status = {
        "items": len(items),
        "annotators_with_files": annotators,
        "per_annotator": {
            annotator: {
                "items": sum(1 for (ann, _) in latest if ann == annotator),
                "coverage": round(
                    sum(1 for (ann, _) in latest if ann == annotator) / len(items), 4
                ) if items else 0.0,
            }
            for annotator in annotators
        },
        "adjudication_records": len(store.load_adjudications()),
        "revisions": sum(1 for count in store.revision_counts().values() if count > 1),
        "next": "two independent sheets are required before any agreement value exists",
    }
    print(json.dumps(status, indent=2))
    return 0


def cmd_validate(args) -> int:
    store = _store(args)
    candidates = store.candidate_ids()
    records = store.load_records()
    errors = []
    for record in records:
        item_id = str(record.get("item_id", ""))
        messages = schema.validate_record(record, candidates.get(item_id, []))
        if messages:
            errors.append({"annotator": record.get("annotator"), "item_id": item_id,
                           "errors": messages})
    print(json.dumps({
        "records": len(records),
        "invalid": len(errors),
        "errors": errors[:50],
        "note": "a valid record can still be incomplete; run `status` or the QC report for coverage",
    }, indent=2))
    return 0 if not errors else 1


def cmd_import(args) -> int:
    store = _store(args)
    try:
        result = store.import_file(args.annotator, args.file)
    except (ValueError, FileNotFoundError) as exc:
        print(f"error: {exc}")
        return 1
    print(json.dumps(result, indent=2))
    if result.get("rejected"):
        print("import rejected: no rows were written (see the reject report)")
        return 1
    return 0


def cmd_export(args) -> int:
    store = _store(args)
    if args.merged:
        print(json.dumps(store.export_merged(), indent=2))
        return 0
    if not args.annotator:
        print("error: pass --annotator NAME or --merged")
        return 1
    try:
        print(json.dumps(store.export_annotator(args.annotator), indent=2))
    except ValueError as exc:
        print(f"error: {exc}")
        return 1
    return 0


def cmd_agreement(args) -> int:
    store = _store(args)
    report = agreement_mod.agreement_for(store)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = res = out / "agreement.json"
    res.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "reason": report.get("reason"),
        "annotators": report["annotators"],
        "items_annotated": report["n_items_annotated"],
        "written": str(path),
    }, indent=2))
    return 0


def cmd_adjudicate(args) -> int:
    store = _store(args)
    if args.list:
        qc = quality_mod.qc_report(store)
        print(json.dumps(qc["adjudication"], indent=2, ensure_ascii=False))
        return 0
    missing = [name for name in ("item", "task", "label", "adjudicator") if not getattr(args, name)]
    if missing:
        print(f"error: --{missing[0]} is required (or use --list)")
        return 1
    try:
        record = store.append_adjudication(
            item_id=args.item,
            task=args.task,
            label=args.label,
            adjudicator=args.adjudicator,
            candidate_id=args.candidate,
            note=args.note,
            status=args.status,
        )
    except ValueError as exc:
        print(f"error: {exc}")
        return 1
    print(json.dumps(record, indent=2, ensure_ascii=False))
    return 0


def cmd_report(args) -> int:
    store = _store(args)
    try:
        result = report_mod.write_report(store, Path(args.out_dir), overwrite=args.overwrite)
    except FileExistsError as exc:
        print(f"error: {exc}")
        return 1
    print(json.dumps(result, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", default=str(store_mod.DEFAULT_DATA_DIR))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("build-sample", help="build the frozen 300-query bilingual-ready English sample")
    p.add_argument("--n", type=int, default=sampling.DEFAULT_N)
    p.add_argument("--seed", type=int, default=sampling.DEFAULT_SEED)
    p.add_argument("--overwrite", action="store_true")
    p.set_defaults(func=cmd_build_sample)

    p = sub.add_parser("make-ui", help="write one offline HTML page per annotator")
    p.add_argument("--annotator", nargs="+", default=["A001", "A002"])
    p.add_argument("--overwrite", action="store_true")
    p.set_defaults(func=cmd_make_ui)

    p = sub.add_parser("status", help="coverage summary")
    p.set_defaults(func=cmd_status)

    p = sub.add_parser("validate", help="re-validate every stored record")
    p.set_defaults(func=cmd_validate)

    p = sub.add_parser("import", help="import one annotator's JSON/JSONL/CSV export")
    p.add_argument("--annotator", required=True)
    p.add_argument("--file", required=True)
    p.set_defaults(func=cmd_import)

    p = sub.add_parser("export", help="export one annotator or the merged dataset")
    p.add_argument("--annotator")
    p.add_argument("--merged", action="store_true")
    p.set_defaults(func=cmd_export)

    p = sub.add_parser("agreement", help="compute agreement (PENDING until two annotators exist)")
    p.set_defaults(func=cmd_agreement)

    p = sub.add_parser("adjudicate", help="record a human adjudication or list pending ones")
    p.add_argument("--list", action="store_true")
    p.add_argument("--item")
    p.add_argument("--task")
    p.add_argument("--candidate")
    p.add_argument("--label")
    p.add_argument("--adjudicator")
    p.add_argument("--note", default="")
    p.add_argument("--status", default="resolved", choices=["resolved", "unresolved"])
    p.set_defaults(func=cmd_adjudicate)

    p = sub.add_parser("report", help="write REPORT.md, QC_REPORT.md and JSON/CSV tables")
    p.add_argument("--overwrite", action="store_true")
    p.set_defaults(func=cmd_report)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
