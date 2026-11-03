"""Entity-level evaluation for the synthetic PII test set.

Matching is strict: a prediction is correct only when its character span and
entity type exactly match a gold annotation.  This is deliberately harsh for
the conservative name recogniser; partial name matches are reported as both a
false positive and a false negative rather than silently counted as correct.

All examples are synthetic.  These numbers are not an estimate of real-world
PII prevalence or of production readiness.
"""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path
from typing import Iterable, Sequence

from .recognizers import ENTITY_TYPES, detect

DATA_PATH = Path(__file__).resolve().parent / "data" / "synthetic_pii_eval.csv"


def load_rows(path: Path = DATA_PATH) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["entities"] = json.loads(row["entities"])
    return rows


def _keys(entities: Iterable[dict]) -> set[tuple[int, int, str]]:
    return {(int(e["start"]), int(e["end"]), str(e["type"])) for e in entities}


def _counts(gold: set, pred: set) -> tuple[int, int, int]:
    return len(gold & pred), len(pred - gold), len(gold - pred)


def _prf(tp: int, fp: int, fn: int) -> dict:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def _bootstrap_f1(records: Sequence[tuple[set, set, str, str, str]], n_boot: int = 2000, seed: int = 0) -> dict:
    """Percentile bootstrap over documents for micro F1."""
    if not records:
        return {"low": 0.0, "high": 0.0, "mean": 0.0}
    rng = random.Random(seed)
    values: list[float] = []
    for _ in range(n_boot):
        tp = fp = fn = 0
        for _ in range(len(records)):
            gold, pred, _lang, _doc_id, _notes = records[rng.randrange(len(records))]
            a, b, c = _counts(gold, pred)
            tp += a
            fp += b
            fn += c
        values.append(_prf(tp, fp, fn)["f1"])
    values.sort()
    lo = values[max(0, int(0.025 * n_boot) - 1)]
    hi = values[min(n_boot - 1, int(0.975 * n_boot))]
    return {
        "low": round(lo, 4),
        "high": round(hi, 4),
        "mean": round(sum(values) / len(values), 4),
        "n_boot": n_boot,
        "seed": seed,
    }


def evaluate(rows: Sequence[dict], policy: str = "balanced") -> dict:
    """Evaluate one detection policy and return a JSON-serialisable report."""
    records: list[tuple[set, set, str, str, str]] = []
    errors: list[dict] = []
    total_tp = total_fp = total_fn = 0
    negative_docs = 0
    negative_docs_with_fp = 0
    exact_docs = 0

    by_type = {kind: {"tp": 0, "fp": 0, "fn": 0, "gold": 0, "pred": 0} for kind in ENTITY_TYPES}
    by_language: dict[str, dict] = {}

    for row in rows:
        gold_dicts = row["entities"]
        gold = _keys(gold_dicts)
        preds = detect(row["text"], policy=policy)
        pred = {(d.start, d.end, d.type) for d in preds}

        tp, fp, fn = _counts(gold, pred)
        total_tp += tp
        total_fp += fp
        total_fn += fn
        if gold == pred:
            exact_docs += 1

        lang = row["language"]
        bucket = by_language.setdefault(
            lang, {"tp": 0, "fp": 0, "fn": 0, "documents": 0, "negative_documents": 0, "negative_with_fp": 0}
        )
        bucket["documents"] += 1
        bucket["tp"] += tp
        bucket["fp"] += fp
        bucket["fn"] += fn

        if not gold:
            negative_docs += 1
            bucket["negative_documents"] += 1
            if pred:
                negative_docs_with_fp += 1
                bucket["negative_with_fp"] += 1

        for kind in ENTITY_TYPES:
            g = {k for k in gold if k[2] == kind}
            p = {k for k in pred if k[2] == kind}
            by_type[kind]["gold"] += len(g)
            by_type[kind]["pred"] += len(p)
            a, b, c = _counts(g, p)
            by_type[kind]["tp"] += a
            by_type[kind]["fp"] += b
            by_type[kind]["fn"] += c

        if fp or fn:
            errors.append(
                {
                    "id": row["id"],
                    "language": lang,
                    "gold_types": sorted({k[2] for k in gold}),
                    "pred_types": sorted({d.type for d in preds}),
                    "false_positive_spans": sorted(
                        {f"{d.type}@{d.start}-{d.end}" for d in preds if (d.start, d.end, d.type) not in gold}
                    ),
                    "false_negative_spans": sorted({f"{k[2]}@{k[0]}-{k[1]}" for k in gold - pred}),
                    "gold_entities": [e["type"] for e in gold_dicts],
                    "notes": row["notes"],
                }
            )

        records.append((gold, pred, lang, row["id"], row["notes"]))

    for lang, bucket in by_language.items():
        bucket.update(_prf(bucket["tp"], bucket["fp"], bucket["fn"]))
        bucket.pop("tp")
        bucket.pop("fp")
        bucket.pop("fn")
        bucket["fpr_document_level"] = round(
            bucket["negative_with_fp"] / bucket["negative_documents"], 4
        ) if bucket["negative_documents"] else None

    per_type = {}
    for kind, bucket in by_type.items():
        metrics = _prf(bucket["tp"], bucket["fp"], bucket["fn"])
        metrics["gold"] = bucket["gold"]
        metrics["pred"] = bucket["pred"]
        per_type[kind] = metrics

    overall = _prf(total_tp, total_fp, total_fn)
    overall.update(
        {
            "documents": len(rows),
            "gold_entities": sum(len(r["entities"]) for r in rows),
            "predicted_entities": sum(len(detect(r["text"], policy=policy)) for r in rows),
            "exact_document_matches": exact_docs,
            "exact_document_rate": round(exact_docs / len(rows), 4) if rows else 0.0,
            "false_positive_rate_document_level": round(
                negative_docs_with_fp / negative_docs, 4
            ) if negative_docs else None,
            "negative_documents": negative_docs,
            "negative_documents_with_false_positive": negative_docs_with_fp,
            "f1_95ci": _bootstrap_f1(records),
        }
    )

    return {
        "policy": policy,
        "synthetic": True,
        "matching": "strict exact span + entity type",
        "overall": overall,
        "per_type": per_type,
        "per_language": by_language,
        "errors": errors,
    }


def evaluate_both_policies(rows: Sequence[dict]) -> dict:
    return {
        "dataset": "synthetic_pii_eval.csv",
        "synthetic": True,
        "n_examples": len(rows),
        "policies": {
            "balanced": evaluate(rows, policy="balanced"),
            "strict": evaluate(rows, policy="strict"),
        },
    }
