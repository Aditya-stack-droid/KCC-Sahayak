"""Quality control for the annotation study.

Checks implemented
------------------
* coverage: which sampled items each annotator has not labelled
* incompleteness: which tasks are still blank on a labelled item
* stale/duplicate records: more than one record for the same (annotator, item)
* invalid labels: re-validation of stored records against the schema
* consistency: structural contradictions (captured by the schema) and
  copy-paste suspicion (a long identical label run between two annotators)
* adjudication status: disagreements with no resolved adjudication

Nothing here resolves a disagreement; it only reports what a human must decide.
"""

from __future__ import annotations

import itertools
from collections import Counter
from typing import Any, Mapping, Sequence

from . import schema
from .agreement import _collect as collect_labels
from .agreement import disagreements
from . import store as store_mod


def longest_identical_run(pairs: Sequence[tuple[str, str]]) -> int:
    """Longest run of consecutive items where both annotators chose the same label."""
    best = current = 0
    previous: tuple[str, str] | None = None
    for pair in pairs:
        if pair[0] and pair[1] and pair[0] == pair[1]:
            current = current + 1 if previous == pair else 1
            best = max(best, current)
        else:
            current = 0
        previous = pair
    return best


def qc_report(store: store_mod.Store) -> dict:
    items = store.load_items()
    key = store.load_key()
    if not items:
        return {
            "status": "PENDING",
            "reason": "annotation sample has not been built yet",
            "items": 0,
        }

    candidates = store.candidate_ids()
    records = store.load_records()
    latest = store.latest()
    revisions = store.revision_counts()
    annotators = sorted({str(record["annotator"]) for record in records})
    item_ids = [item["item_id"] for item in items]

    per_annotator: dict[str, dict] = {}
    for annotator in annotators:
        own = {item_id: record for (ann, item_id), record in latest.items() if ann == annotator}
        invalid: list[dict] = []
        partial: dict[str, list[str]] = {}
        distribution: Counter = Counter()
        for item_id, record in own.items():
            messages = schema.validate_record(record, candidates.get(item_id, []))
            if messages:
                invalid.append({"item_id": item_id, "errors": messages})
            missing = schema.is_complete(record, candidates.get(item_id, []))
            if missing:
                partial[item_id] = missing
            distribution[str(record.get("intent"))] += 1

        duplicates = sorted(item_id for (ann, item_id), count in revisions.items()
                            if ann == annotator and count > 1)
        warnings: list[str] = []
        if len(own) >= 20 and distribution and max(distribution.values()) == len(own):
            warnings.append("all annotated items received the same intent label; check for rubber-stamping")
        if duplicates:
            warnings.append(f"{len(duplicates)} item(s) were annotated more than once; latest record wins")

        per_annotator[annotator] = {
            "items_annotated": len(own),
            "coverage": round(len(own) / len(item_ids), 4),
            "items_not_annotated": [item_id for item_id in item_ids if item_id not in own],
            "items_with_missing_tasks": len(partial),
            "missing_task_counts": dict(Counter(task for tasks in partial.values() for task in tasks)),
            "invalid_records": invalid,
            "duplicate_items": duplicates,
            "intent_distribution": dict(distribution),
            "warnings": warnings,
        }

    coverage_counter = Counter()
    for _, item_id in latest:
        coverage_counter[item_id] += 1
    uncovered = [item_id for item_id in item_ids if coverage_counter[item_id] == 0]
    multi_annotated = [item_id for item_id in item_ids if coverage_counter[item_id] >= 2]

    consistency: dict[str, Any] = {}
    for left, right in itertools.combinations(annotators, 2):
        pairs = []
        for item_id in item_ids:
            left_label = latest.get((left, item_id), {}).get("intent")
            right_label = latest.get((right, item_id), {}).get("intent")
            pairs.append((str(left_label or ""), str(right_label or "")))
        run = longest_identical_run(pairs)
        consistency[f"{left}|{right}"] = {
            "overlapping_items": sum(1 for a, b in pairs if a and b),
            "longest_identical_intent_run": run,
            "possible_copy_flag": run >= 15,
        }

    # --- adjudication tracking ------------------------------------------
    adjudications = store.load_adjudications()
    index = store.adjudication_index()
    pending: list[dict] = []
    for task in ("intent", "safety", "out_of_scope", "ambiguous"):
        for row in disagreements(collect_labels(records, task)):
            entry = index.get((row["unit"], task, ""))
            if not entry or entry.get("status") != "resolved":
                pending.append({
                    "item_id": row["unit"],
                    "task": task,
                    "candidate_id": "",
                    "labels": row["labels"],
                    "status": "unresolved_record" if entry else "not_adjudicated",
                })
    for task in schema.CANDIDATE_TASKS:
        for row in disagreements(collect_labels(records, task)):
            item_id, candidate_id = row["unit"]
            entry = index.get((item_id, task, candidate_id))
            if not entry or entry.get("status") != "resolved":
                pending.append({
                    "item_id": item_id,
                    "task": task,
                    "candidate_id": candidate_id,
                    "labels": row["labels"],
                    "status": "unresolved_record" if entry else "not_adjudicated",
                })

    resolved = [record for record in adjudications if record.get("status") == "resolved"]
    status = "PENDING" if not records else "IN_PROGRESS"
    reason = None
    if not records:
        reason = "no annotator has submitted a sheet yet"
    if records and not uncovered and not pending and not any(
        info["items_with_missing_tasks"] for info in per_annotator.values()
    ) and len(annotators) >= 2:
        status = "READY_FOR_ANALYSIS"

    return {
        "generated_at": store_mod.utc_now(),
        "status": status,
        "reason": reason,
        "n_items": len(items),
        "n_annotators": len(annotators),
        "annotators": annotators,
        "items_uncovered_by_any_annotator": uncovered,
        "items_with_two_or_more_annotators": len(multi_annotated),
        "items_with_fewer_than_two": len(item_ids) - len(multi_annotated),
        "per_annotator": per_annotator,
        "consistency": consistency,
        "adjudication": {
            "records": len(adjudications),
            "resolved": len(resolved),
            "unresolved_records": len(adjudications) - len(resolved),
            "pending_disagreements": pending,
            "pending_count": len(pending),
        },
        "llm_labels_used_for_qc": False,
        "notes": [
            "QC never resolves a disagreement; it lists work for an adjudicator.",
            "The existing LLM labels are not compared against themselves here.",
            "Duplicate records are revisions, not errors; the latest record is used.",
            "The copied-label check is a warning, not proof of misconduct.",
        ],
    }
