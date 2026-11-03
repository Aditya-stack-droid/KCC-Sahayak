"""Agreement statistics for human annotations.

Only labels typed by people are used here.  The LLM labels from the existing
project are compared separately and always reported as *human vs LLM*, never
as human consensus.  If fewer than two independent annotators exist, every
function returns a PENDING result instead of a number.

Implemented
-----------
* Cohen's kappa (two annotators, complete cases per pair)
* Fleiss' kappa (three or more annotators, items rated by all of them)
* observed/expected agreement, confusion matrices, per-class one-vs-rest kappa
* set-valued agreement for multi-intent and missing-context labels
* disagreement rows for adjudication
"""

from __future__ import annotations

import itertools
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from . import schema, store as store_mod


# --------------------------------------------------------------------------- #
# Core statistics
# --------------------------------------------------------------------------- #
def cohen_kappa(a: Sequence[str], b: Sequence[str], categories: Sequence[str] | None = None) -> dict:
    """Cohen's kappa for two aligned label sequences.

    Missing labels must already be removed by the caller.  Returns ``None``
    for kappa when expected agreement is 1 (both raters constant on the same
    category), because kappa is undefined rather than perfect there.
    """
    if len(a) != len(b):
        raise ValueError(f"label sequences differ in length: {len(a)} != {len(b)}")
    n = len(a)
    cats = list(categories) if categories is not None else sorted(set(a) | set(b))
    unknown = (set(a) | set(b)) - set(cats)
    if unknown:
        raise ValueError(f"labels outside the declared categories: {sorted(unknown)}")
    if n == 0:
        return {"n": 0, "observed_agreement": None, "expected_agreement": None,
                "kappa": None, "categories": cats}

    index = {c: i for i, c in enumerate(cats)}
    counts_a = [0] * len(cats)
    counts_b = [0] * len(cats)
    agreements = 0
    for x, y in zip(a, b):
        counts_a[index[x]] += 1
        counts_b[index[y]] += 1
        agreements += int(x == y)

    observed = agreements / n
    expected = sum((ca / n) * (cb / n) for ca, cb in zip(counts_a, counts_b))
    kappa = None if expected >= 1.0 else (observed - expected) / (1.0 - expected)
    return {
        "n": n,
        "observed_agreement": round(float(observed), 6),
        "expected_agreement": round(float(expected), 6),
        "kappa": None if kappa is None else round(float(kappa), 6),
        "kappa_defined": kappa is not None,
        "categories": cats,
    }


def fleiss_kappa(counts: Sequence[Sequence[int]]) -> dict:
    """Fleiss' kappa from an ``items x categories`` count matrix.

    Every item must have the same number of raters (complete cases only).
    """
    table = np.asarray(counts, dtype=float)
    if table.ndim != 2:
        raise ValueError("counts must be a two-dimensional item x category matrix")
    n_items, n_categories = table.shape
    if n_items == 0:
        return {"n_items": 0, "n_raters": 0, "observed_agreement": None,
                "expected_agreement": None, "kappa": None}
    raters = table.sum(axis=1)
    if not np.allclose(raters, raters[0]):
        raise ValueError("Fleiss' kappa requires the same number of raters per item")
    n = float(raters[0])
    if n < 2:
        raise ValueError("Fleiss' kappa requires at least two raters per item")

    per_item = ((table ** 2).sum(axis=1) - n) / (n * (n - 1))
    observed = float(per_item.mean())
    proportions = table.sum(axis=0) / (n_items * n)
    expected = float((proportions ** 2).sum())
    kappa = None if expected >= 1.0 else (observed - expected) / (1.0 - expected)
    return {
        "n_items": int(n_items),
        "n_raters": int(n),
        "n_categories": int(n_categories),
        "observed_agreement": round(observed, 6),
        "expected_agreement": round(expected, 6),
        "kappa": None if kappa is None else round(float(kappa), 6),
        "kappa_defined": kappa is not None,
    }


def confusion(a: Sequence[str], b: Sequence[str], categories: Sequence[str]) -> dict:
    cats = list(categories)
    table = {row: {col: 0 for col in cats} for row in cats}
    for x, y in zip(a, b):
        table[x][y] += 1
    return {
        "categories": cats,
        "counts": table,
        "row_totals": {c: sum(table[c].values()) for c in cats},
        "col_totals": {c: sum(table[r][c] for r in cats) for c in cats},
    }


def per_class_kappa(a: Sequence[str], b: Sequence[str], categories: Sequence[str]) -> dict:
    """One-vs-rest kappa per class plus how often each class was used."""
    out: dict[str, dict] = {}
    for category in categories:
        aa = ["1" if x == category else "0" for x in a]
        bb = ["1" if y == category else "0" for y in b]
        stats = cohen_kappa(aa, bb, ["0", "1"])
        out[category] = {
            "support_a": sum(1 for x in a if x == category),
            "support_b": sum(1 for y in b if y == category),
            "support_either": sum(1 for x, y in zip(a, b) if x == category or y == category),
            "kappa": stats["kappa"],
            "observed_agreement": stats["observed_agreement"],
        }
    return out


def _pairs(annotators: Sequence[str]) -> Iterable[tuple[str, str]]:
    return itertools.combinations(sorted(annotators), 2)


# --------------------------------------------------------------------------- #
# Task-level reports
# --------------------------------------------------------------------------- #
def task_report(per_item: Mapping[str, Mapping[str, str]], categories: Sequence[str]) -> dict:
    """Pairwise Cohen + Fleiss for one categorical task.

    ``per_item`` maps an item (or item+candidate) key to ``{annotator: label}``.
    Items missing an annotator's label are excluded only for that pair, and the
    count of excluded items is reported so missingness is visible.
    """
    annotators = sorted({ann for labels in per_item.values() for ann in labels})
    multi = {key: labels for key, labels in per_item.items() if len(labels) >= 2}

    result: dict[str, Any] = {
        "annotators": annotators,
        "n_units": len(per_item),
        "n_units_multi_annotated": len(multi),
        "n_units_single_annotated": len(per_item) - len(multi),
        "pairs": {},
        "fleiss": None,
        "fleiss_reason": None,
    }
    if len(annotators) < 2:
        result["status"] = "PENDING"
        result["reason"] = "fewer than two independent annotators"
        return result

    for left, right in _pairs(annotators):
        common = [key for key, labels in per_item.items() if left in labels and right in labels]
        if not common:
            continue
        a = [per_item[key][left] for key in common]
        b = [per_item[key][right] for key in common]
        stats = cohen_kappa(a, b, categories)
        stats.update({
            "n_common": len(common),
            "n_missing_left": sum(1 for key, labels in per_item.items() if left in labels and right not in labels),
            "n_missing_right": sum(1 for key, labels in per_item.items() if right in labels and left not in labels),
            "confusion": confusion(a, b, categories),
            "per_class": per_class_kappa(a, b, categories),
        })
        result["pairs"][f"{left}|{right}"] = stats

    if len(annotators) >= 3:
        complete = [key for key, labels in per_item.items() if all(ann in labels for ann in annotators)]
        if complete:
            table = np.zeros((len(complete), len(categories)), dtype=float)
            index = {c: i for i, c in enumerate(categories)}
            for row, key in enumerate(complete):
                for ann in annotators:
                    table[row, index[per_item[key][ann]]] += 1
            result["fleiss"] = fleiss_kappa(table)
            result["fleiss"]["n_complete_units"] = len(complete)
            result["fleiss"]["annotators"] = annotators
        else:
            result["fleiss_reason"] = "no unit was rated by all annotators"
    else:
        result["fleiss_reason"] = "fewer than three annotators"

    result["status"] = "OK" if result["pairs"] else "PENDING"
    if not result["pairs"]:
        result["reason"] = "no item was annotated by two different annotators"
    return result


def set_report(per_item: Mapping[str, Mapping[str, Sequence[str]]]) -> dict:
    """Agreement for set-valued labels (multi-intent, missing context).

    Kappa is not defined for label sets, so this reports exact-set agreement
    and mean Jaccard similarity instead.  That limitation is deliberate.
    """
    annotators = sorted({ann for labels in per_item.values() for ann in labels})
    pairs: dict[str, dict] = {}
    for left, right in _pairs(annotators):
        common = [key for key, labels in per_item.items() if left in labels and right in labels]
        if not common:
            continue
        exact = 0
        jaccard = []
        for key in common:
            a = set(per_item[key][left])
            b = set(per_item[key][right])
            exact += int(a == b)
            union = a | b
            jaccard.append(len(a & b) / len(union) if union else 1.0)
        pairs[f"{left}|{right}"] = {
            "n_common": len(common),
            "exact_set_agreement": round(exact / len(common), 6),
            "mean_jaccard": round(float(np.mean(jaccard)), 6),
        }
    return {
        "annotators": annotators,
        "n_units": len(per_item),
        "pairs": pairs,
        "metric": "set agreement (kappa is not defined for label sets)",
        "status": "OK" if pairs else "PENDING",
    }


def disagreements(per_item: Mapping[str, Mapping[str, Any]]) -> list[dict]:
    rows: list[dict] = []
    for key, labels in sorted(per_item.items()):
        values = [value for value in labels.values() if value not in (None, "", [])]
        if len({str(value) for value in values}) > 1:
            rows.append({"unit": key, "labels": {ann: value for ann, value in labels.items()}})
    return rows


# --------------------------------------------------------------------------- #
# Collection helpers
# --------------------------------------------------------------------------- #
def _collect(records: Iterable[Mapping[str, Any]], task: str) -> dict:
    per_unit: dict[Any, dict[str, Any]] = {}
    for record in records:
        annotator = str(record["annotator"])
        item_id = str(record["item_id"])
        if task in schema.CANDIDATE_TASKS:
            labels = record.get(task) or {}
            for candidate, value in labels.items():
                if value in (None, ""):
                    continue
                per_unit.setdefault((item_id, str(candidate)), {})[annotator] = str(value)
            continue
        value = record.get(task)
        if value in (None, "", []):
            continue
        if task in {"multi_intent", "missing_context"}:
            per_unit.setdefault(item_id, {})[annotator] = tuple(sorted(str(v) for v in value))
        elif task in {"out_of_scope", "ambiguous"}:
            per_unit.setdefault(item_id, {})[annotator] = str(bool(value))
        else:
            per_unit.setdefault(item_id, {})[annotator] = str(value)
    return per_unit


def human_vs_llm(records: Iterable[Mapping[str, Any]], key: Mapping[str, Mapping[str, Any]]) -> dict:
    """Compare each human annotator's intent label against the existing LLM label.

    This is explicitly **human vs LLM**, not human consensus and not expert
    validation.  The LLM labels are the ones shipped with the project.
    """
    output: dict[str, Any] = {
        "reference": "existing LLM-generated intent labels (Claude), not human or expert labels",
        "annotators": {},
        "status": "PENDING",
    }
    by_item = _collect(records, "intent")
    for annotator in sorted({ann for labels in by_item.values() for ann in labels}):
        pairs = []
        for item_id, labels in by_item.items():
            if annotator not in labels or item_id not in key:
                continue
            reference = schema.normalise_intent(key[item_id].get("llm_label"))
            if reference is None:
                continue
            pairs.append((reference, labels[annotator]))
        if not pairs:
            continue
        reference_labels = [p[0] for p in pairs]
        human_labels = [p[1] for p in pairs]
        stats = cohen_kappa(reference_labels, human_labels, schema.INTENT_ORDER)
        stats["accuracy_vs_llm"] = round(
            sum(int(a == b) for a, b in pairs) / len(pairs), 6
        )
        stats["confusion"] = confusion(reference_labels, human_labels, schema.INTENT_ORDER)
        output["annotators"][annotator] = stats
        output["status"] = "OK"
    if output["status"] == "PENDING":
        output["reason"] = "no human intent labels available"
    return output


def agreement_for(store: store_mod.Store) -> dict:
    """Full agreement report for the study (PENDING until two people annotate)."""
    items = store.load_items()
    key = store.load_key()
    latest = store.latest()
    records = list(latest.values())
    annotators = sorted({record["annotator"] for record in records})

    report: dict[str, Any] = {
        "generated_at": store_mod.utc_now(),
        "n_items_in_sample": len(items),
        "n_annotation_records": len(store.load_records()),
        "annotators": annotators,
        "n_items_annotated": len({record["item_id"] for record in records}),
        "labels_are_human": True,
        "expert_verified": False,
        "notes": [
            "Kappa values here would be computed from human annotation files only.",
            "The existing LLM labels are compared separately under human_vs_llm.",
            "Missing labels are handled by complete-case per pair; counts are reported.",
        ],
        "tasks": {},
        "human_vs_llm": human_vs_llm(records, key),
        "set_tasks": {},
    }

    if len(annotators) < 2:
        report["status"] = "PENDING"
        report["reason"] = (
            "fewer than two independent annotators have returned sheets; "
            "no human-human kappa exists yet"
        )
    else:
        overlap = 0
        by_item = _collect(records, "intent")
        for labels in by_item.values():
            if len(labels) >= 2:
                overlap += 1
        if overlap == 0:
            report["status"] = "PENDING"
            report["reason"] = "no item was annotated by two different annotators"
        else:
            report["status"] = "OK"

    categorical = {
        "intent": schema.INTENT_ORDER,
        "safety": schema.SAFETY_LABELS,
        "out_of_scope": ("False", "True"),
        "ambiguous": ("False", "True"),
    }
    for task, categories in categorical.items():
        per_item = _collect(records, task)
        report["tasks"][task] = task_report(per_item, categories)
        if report["status"] == "OK":
            report.setdefault("disagreements", {})[task] = disagreements(per_item)

    for task in schema.CANDIDATE_TASKS:
        candidates = sorted({unit[1] for unit in _collect(records, task)})
        per_candidate = {}
        for candidate in candidates:
            subset = {
                unit[0]: labels for unit, labels in _collect(records, task).items()
                if unit[1] == candidate
            }
            categories = (schema.RELEVANCE_LABELS if task == "retrieval_relevance"
                          else schema.QUALITY_LABELS if task == "answer_quality"
                          else schema.SAFETY_LABELS)
            per_candidate[candidate] = task_report(subset, categories)
            if report["status"] == "OK":
                report.setdefault("disagreements", {})[f"{task}:{candidate}"] = disagreements(subset)
        report["tasks"][task] = {
            "per_candidate": per_candidate,
            "aggregate": task_report(_collect(records, task), {
                "retrieval_relevance": schema.RELEVANCE_LABELS,
                "answer_quality": schema.QUALITY_LABELS,
                "answer_safety": schema.SAFETY_LABELS,
            }[task]),
            "status": "OK" if any(v.get("status") == "OK" for v in per_candidate.values()) else "PENDING",
        }

    for task in ("multi_intent", "missing_context"):
        per_item = _collect(records, task)
        report["set_tasks"][task] = set_report(per_item)
        if report["status"] == "OK":
            report.setdefault("disagreements", {})[task] = disagreements(per_item)

    return report
