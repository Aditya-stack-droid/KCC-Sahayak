"""Annotation storage: per-annotator append-only shards, import/export and
adjudication records.

Design rules
------------
* Each annotator writes to their own shard ``data/annotations/<name>.jsonl``;
  a shard is never read by the annotation UI of another annotator, which is
  how blinding is enforced structurally rather than by etiquette.
* Records are append-only.  Re-annotating an item adds a revision; the latest
  record wins and the earlier ones are kept and counted.
* An import is all-or-nothing: if any row fails schema validation nothing is
  written and a rejection report is produced.  Partially valid sheets are
  therefore never silently truncated.
* Adjudication is a separate append-only log with an explicit adjudicator name
  and status.  Nothing here can adjudicate on its own or call a model.
"""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from . import schema

HERE = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = HERE / "data"

CANDIDATE_FIELD = re.compile(r"^(c\d+)_(relevance|quality|safety)$")
TASK_BY_SUFFIX = {
    "relevance": "retrieval_relevance",
    "quality": "answer_quality",
    "safety": "answer_safety",
}
QUERY_FIELDS = (
    "intent",
    "multi_intent",
    "out_of_scope",
    "safety",
    "missing_context",
    "ambiguous",
    "uncertain",
    "certainty",
    "notes",
)
ADJUDICATION_TASKS = (
    "intent",
    "multi_intent",
    "out_of_scope",
    "safety",
    "missing_context",
    "ambiguous",
    "retrieval_relevance",
    "answer_quality",
    "answer_safety",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def read_jsonl(path: Path) -> list[dict]:
    if not Path(path).exists():
        return []
    rows: list[dict] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]], overwrite: bool = False) -> Path:
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite {path} (pass --overwrite to replace)")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path


def write_json(path: Path, obj: Any, overwrite: bool = False) -> Path:
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite {path} (pass --overwrite to replace)")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


@dataclass
class Store:
    """Filesystem layout for one annotation study."""

    data_dir: Path = DEFAULT_DATA_DIR

    # ---- paths ----------------------------------------------------------
    @property
    def items_path(self) -> Path:
        return Path(self.data_dir) / "items.jsonl"

    @property
    def key_path(self) -> Path:
        return Path(self.data_dir) / "key.jsonl"

    @property
    def summary_path(self) -> Path:
        return Path(self.data_dir) / "sample_summary.json"

    @property
    def annotations_dir(self) -> Path:
        return Path(self.data_dir) / "annotations"

    @property
    def adjudication_path(self) -> Path:
        return Path(self.data_dir) / "adjudication.jsonl"

    @property
    def exports_dir(self) -> Path:
        return Path(self.data_dir) / "exports"

    @property
    def ui_dir(self) -> Path:
        return Path(self.data_dir) / "ui"

    # ---- items ----------------------------------------------------------
    def load_items(self) -> list[dict]:
        return read_jsonl(self.items_path)

    def load_key(self) -> dict[str, dict]:
        return {row["item_id"]: row for row in read_jsonl(self.key_path)}

    def candidate_ids(self) -> dict[str, list[str]]:
        return {
            item["item_id"]: [c["candidate_id"] for c in item.get("candidates", [])]
            for item in self.load_items()
        }

    def load_summary(self) -> dict:
        if self.summary_path.exists():
            return json.loads(self.summary_path.read_text(encoding="utf-8"))
        return {}

    # ---- annotations ----------------------------------------------------
    def annotation_files(self) -> list[Path]:
        if not self.annotations_dir.exists():
            return []
        return sorted(self.annotations_dir.glob("*.jsonl"))

    def annotators(self) -> list[str]:
        names = set()
        for path in self.annotation_files():
            names.add(path.stem)
        for record in read_jsonl(self.adjudication_path):
            if record.get("adjudicator"):
                names.add(str(record["adjudicator"]))
        return sorted(names)

    def load_records(self, annotator: str | None = None) -> list[dict]:
        rows: list[dict] = []
        for path in self.annotation_files():
            if annotator is not None and path.stem != annotator:
                continue
            rows.extend(read_jsonl(path))
        return rows

    def latest(self, annotator: str | None = None) -> dict[tuple[str, str], dict]:
        latest: dict[tuple[str, str], dict] = {}
        for record in self.load_records(annotator):
            latest[(str(record["annotator"]), str(record["item_id"]))] = record
        return latest

    def revision_counts(self) -> dict[tuple[str, str], int]:
        counts: dict[tuple[str, str], int] = {}
        for record in self.load_records():
            key = (str(record["annotator"]), str(record["item_id"]))
            counts[key] = counts.get(key, 0) + 1
        return counts

    def append_records(self, annotator: str, records: Sequence[Mapping[str, Any]]) -> int:
        annotator = str(annotator).strip()
        if not annotator:
            raise ValueError("annotator is required")
        self.annotations_dir.mkdir(parents=True, exist_ok=True)
        path = self.annotations_dir / f"{annotator}.jsonl"
        with path.open("a", encoding="utf-8") as handle:
            for record in records:
                payload = dict(record)
                payload["annotator"] = annotator
                payload.setdefault("created_at", utc_now())
                handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        return len(records)

    # ---- import ---------------------------------------------------------
    @staticmethod
    def _records_from_csv(path: Path, annotator: str) -> list[dict]:
        records: list[dict] = []
        with Path(path).open(encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                item_id = str(row.get("item_id") or "").strip()
                if not item_id:
                    continue
                record: dict[str, Any] = {"annotator": str(row.get("annotator") or annotator).strip(),
                                          "item_id": item_id}
                for field in QUERY_FIELDS:
                    if field in row and str(row.get(field) or "").strip() != "":
                        record[field] = row[field]
                for column, value in row.items():
                    match = CANDIDATE_FIELD.match(str(column or "").strip())
                    if not match or str(value or "").strip() == "":
                        continue
                    task = TASK_BY_SUFFIX[match.group(2)]
                    record.setdefault(task, {})[match.group(1)] = str(value).strip()
                if any(field in record for field in QUERY_FIELDS) or any(
                    task in record for task in TASK_BY_SUFFIX.values()
                ):
                    records.append(record)
        return records

    def import_file(self, annotator: str, path: str | Path) -> dict:
        annotator = str(annotator).strip()
        path = Path(path)
        if not annotator:
            raise ValueError("annotator is required")
        if not path.exists():
            raise FileNotFoundError(path)

        if path.suffix.lower() == ".json":
            payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(payload, Mapping):
                records = payload.get("records") or payload.get("annotations") or []
                annotator = str(payload.get("annotator") or annotator).strip()
            elif isinstance(payload, list):
                records = payload
            else:
                raise ValueError("JSON import must be an object with 'records' or a list")
        elif path.suffix.lower() in {".jsonl", ".ndjson"}:
            records = read_jsonl(path)
        elif path.suffix.lower() == ".csv":
            records = self._records_from_csv(path, annotator)
        else:
            raise ValueError(f"unsupported import format: {path.suffix}")

        candidates = self.candidate_ids()
        existing = self.latest()
        errors: list[dict] = []
        cleaned: list[dict] = []

        skipped_empty = 0
        for row_number, record in enumerate(records, start=1):
            item_id = str(record.get("item_id", "")).strip()
            if item_id not in candidates:
                errors.append({"row": row_number, "item_id": item_id, "errors": ["unknown item_id"]})
                continue
            payload = dict(record)
            payload["annotator"] = annotator
            payload["item_id"] = item_id
            if not has_any_label(payload):
                skipped_empty += 1
                continue
            messages = schema.validate_record(payload, candidates[item_id])
            if messages:
                errors.append({"row": row_number, "item_id": item_id, "errors": messages})
                continue
            cleaned.append(schema.normalise_record(payload))

        if errors:
            self.exports_dir.mkdir(parents=True, exist_ok=True)
            reject_path = self.exports_dir / f"rejected_{annotator}_{date_stamp()}.json"
            write_json(reject_path, {
                "annotator": annotator,
                "source_file": str(path),
                "n_records": len(records),
                "n_rejected": len(errors),
                "errors": errors,
            }, overwrite=True)
            return {
                "imported": 0,
                "rejected": len(errors),
                "records_in_file": len(records),
                "reject_report": str(reject_path),
                "errors": errors,
            }

        duplicates = sum(1 for rec in cleaned if (annotator, rec["item_id"]) in existing)
        self.append_records(annotator, cleaned)
        return {
            "imported": len(cleaned),
            "rejected": 0,
            "skipped_empty": skipped_empty,
            "records_in_file": len(records),
            "revisions": duplicates,
            "annotator": annotator,
            "source_file": str(path),
        }

    # ---- export ---------------------------------------------------------
    def _csv_rows(self, records: Sequence[Mapping[str, Any]]) -> tuple[list[str], list[list[str]]]:
        candidate_ids = sorted({c for rec in records for task in schema.CANDIDATE_TASKS
                                for c in (rec.get(task) or {})})
        header = ["item_id", *QUERY_FIELDS]
        for task, suffix in (("retrieval_relevance", "relevance"),
                             ("answer_quality", "quality"),
                             ("answer_safety", "safety")):
            header.extend(f"{candidate}_{suffix}" for candidate in candidate_ids)
        rows: list[list[str]] = []
        for record in records:
            row = [str(record.get("item_id", ""))]
            for field in QUERY_FIELDS:
                value = record.get(field, "")
                if isinstance(value, (list, tuple)):
                    value = ";".join(str(v) for v in value)
                row.append("" if value is None else str(value))
            for task in schema.CANDIDATE_TASKS:
                labels = record.get(task) or {}
                row.extend(str(labels.get(candidate, "")) for candidate in candidate_ids)
            rows.append(row)
        return header, rows

    def export_annotator(self, annotator: str) -> dict:
        records = [rec for (_, _), rec in sorted(self.latest(annotator).items())]
        if not records:
            raise ValueError(f"no annotations for {annotator}")
        self.exports_dir.mkdir(parents=True, exist_ok=True)
        json_path = self.exports_dir / f"annotations_{annotator}.json"
        write_json(json_path, {
            "annotator": annotator,
            "exported_at": utc_now(),
            "records": records,
        }, overwrite=True)

        csv_path = self.exports_dir / f"annotations_{annotator}.csv"
        header, rows = self._csv_rows(records)
        with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(header)
            writer.writerows(rows)
        return {"annotator": annotator, "records": len(records),
                "json": str(json_path), "csv": str(csv_path)}

    def export_merged(self) -> dict:
        """One row per item and annotator, plus resolved adjudicated labels."""
        items = self.load_items()
        latest = self.latest()
        adjudications = self.adjudication_index()
        rows: list[dict] = []
        for item in items:
            item_id = item["item_id"]
            for annotator in sorted({ann for ann, iid in latest if iid == item_id}):
                record = latest[(annotator, item_id)]
                row: dict[str, Any] = {"item_id": item_id, "query": item["query"], "annotator": annotator}
                for field in QUERY_FIELDS:
                    value = record.get(field, "")
                    row[field] = ";".join(value) if isinstance(value, (list, tuple)) else value
                for task in schema.CANDIDATE_TASKS:
                    for candidate in [c["candidate_id"] for c in item["candidates"]]:
                        row[f"{candidate}_{task}"] = (record.get(task) or {}).get(candidate, "")
                rows.append(row)

        resolved: list[dict] = []
        for item in items:
            item_id = item["item_id"]
            record = {"item_id": item_id, "query": item["query"]}
            for task in ADJUDICATION_TASKS:
                targets = [""] if task not in schema.CANDIDATE_TASKS else [
                    c["candidate_id"] for c in item["candidates"]
                ]
                for target in targets:
                    entry = adjudications.get((item_id, task, target))
                    if entry and entry.get("status") == "resolved":
                        column = task if not target else f"{target}_{task}"
                        record[f"{column}_adjudicated"] = entry["label"]
            resolved.append(record)

        self.exports_dir.mkdir(parents=True, exist_ok=True)
        json_path = write_json(self.exports_dir / "merged_annotations.json", {
            "exported_at": utc_now(),
            "n_annotators": len(self.annotators()),
            "provenance": "human annotations (not expert-verified); adjudicated labels only where status=resolved",
            "rows": rows,
            "adjudicated": resolved,
        }, overwrite=True)
        return {"rows": len(rows), "items": len(items), "json": str(json_path)}

    # ---- adjudication ---------------------------------------------------
    def load_adjudications(self) -> list[dict]:
        return read_jsonl(self.adjudication_path)

    def adjudication_index(self) -> dict[tuple[str, str, str], dict]:
        index: dict[tuple[str, str, str], dict] = {}
        for record in self.load_adjudications():
            key = (str(record["item_id"]), str(record["task"]), str(record.get("candidate_id") or ""))
            index[key] = record
        return index

    def append_adjudication(self, item_id: str, task: str, label: Any, adjudicator: str,
                            candidate_id: str | None = None, note: str = "",
                            status: str = "resolved") -> dict:
        if task not in ADJUDICATION_TASKS:
            raise ValueError(f"task must be one of {list(ADJUDICATION_TASKS)}")
        if status not in {"resolved", "unresolved"}:
            raise ValueError("status must be 'resolved' or 'unresolved'")
        items = {item["item_id"]: item for item in self.load_items()}
        if item_id not in items:
            raise ValueError(f"unknown item_id {item_id!r}")
        candidates = [c["candidate_id"] for c in items[item_id]["candidates"]]
        if candidate_id and candidate_id not in candidates:
            raise ValueError(f"unknown candidate_id {candidate_id!r} for {item_id}")

        normalised = validate_adjudicated_label(task, label)
        record = {
            "item_id": item_id,
            "task": task,
            "candidate_id": candidate_id or "",
            "label": normalised,
            "adjudicator": str(adjudicator).strip(),
            "status": status,
            "note": str(note or "").strip(),
            "created_at": utc_now(),
        }
        if not record["adjudicator"]:
            raise ValueError("adjudicator is required")
        self.adjudication_path.parent.mkdir(parents=True, exist_ok=True)
        with self.adjudication_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        return record


def has_any_label(record: Mapping[str, Any]) -> bool:
    """True if a record contains at least one actual judgement.

    The local UI creates a placeholder object as soon as an item is opened;
    those placeholders must be skipped on import rather than rejected.
    """
    if str(record.get("intent") or "").strip():
        return True
    if str(record.get("safety") or "").strip():
        return True
    if record.get("out_of_scope") not in (None, ""):
        return True
    if record.get("certainty") not in (None, ""):
        return True
    if record.get("missing_context"):
        return True
    if record.get("multi_intent"):
        return True
    if record.get("notes"):
        return True
    return any(record.get(task) for task in schema.CANDIDATE_TASKS)


def date_stamp() -> str:
    return datetime.now().strftime("%Y%m%dT%H%M%S")


def validate_adjudicated_label(task: str, label: Any) -> Any:
    if task == "intent":
        code = schema.normalise_intent(label)
        if code is None:
            raise ValueError(f"intent: unknown label {label!r}")
        return code
    if task == "multi_intent":
        codes = [schema.normalise_intent(v) for v in schema._as_list(label)]
        if not codes or any(code is None for code in codes):
            raise ValueError(f"multi_intent: unknown label in {label!r}")
        return list(dict.fromkeys(codes))
    if task in {"out_of_scope", "ambiguous"}:
        return schema.as_bool(label, task)
    if task in {"safety", "answer_safety"}:
        return schema._normalise_choice(label, schema.SAFETY_LABELS, task)
    if task == "retrieval_relevance":
        return schema._normalise_choice(label, schema.RELEVANCE_LABELS, task)
    if task == "answer_quality":
        return schema._normalise_choice(label, schema.QUALITY_LABELS, task)
    if task == "missing_context":
        values = [schema._normalise_choice(v, schema.CONTEXT_LABELS, task)
                  for v in schema._as_list(label)]
        if not values or any(v is None for v in values):
            raise ValueError(f"missing_context: unknown value in {label!r}")
        if "none" in values and len(values) > 1:
            raise ValueError("missing_context: 'none' cannot be combined with other values")
        return values
    raise ValueError(f"unsupported task {task!r}")
