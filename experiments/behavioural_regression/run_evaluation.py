#!/usr/bin/env python
"""Run the authored behavioural regression probes and write deterministic results."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from br_cases import CASES  # noqa: E402
from br_probes import evaluate_all  # noqa: E402
from br_report import render, summarize  # noqa: E402

DEFAULT_OUTPUT = ROOT / "results" / "behavioural_regression"
FIELDS = [
    "case_id",
    "category",
    "probe",
    "input",
    "expected",
    "observed",
    "status",
    "failure_type",
    "synthetic",
    "note",
]


def _json_cell(value) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def write_outputs(output_dir: Path, records, metrics: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with (output_dir / "cases.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        for record in records:
            row = dict(record)
            row["expected"] = _json_cell(row["expected"])
            row["observed"] = _json_cell(row["observed"])
            row["synthetic"] = "true"
            writer.writerow({field: row[field] for field in FIELDS})

    failures = [record for record in records if record["status"] == "fail"]
    payload = {
        "experiment": metrics["experiment"],
        "evidence_class": metrics["evidence_class"],
        "synthetic": True,
        "n_cases": metrics["n_cases"],
        "n_failures": len(failures),
        "case_sha256": metrics["case_sha256"],
        "failures": failures,
    }
    (output_dir / "failures.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "REPORT.md").write_text(render(metrics, records), encoding="utf-8")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)

    records = evaluate_all(CASES)
    metrics = summarize(records, CASES)
    write_outputs(args.output_dir, records, metrics)

    print(f"cases:   {metrics['n_cases']}")
    print(f"passed:  {metrics['passed']}")
    print(f"failed:  {metrics['failed']}")
    print(f"output:  {args.output_dir}")
    for code in metrics["categories"]:
        block = metrics["by_category"][code]
        print(f"  {code}: {block['passed']}/{block['total']} passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
