#!/usr/bin/env python
"""Descriptive conflict-rule scan over multi-answer question groups.

Reuses Batch A's grouped retrieval metadata and informative-answer mask (no
model, no re-encoding).  Reports how often the deterministic rules fire across
pairs of answers to questions in the same 0.95-similarity group.

This is a rule-fire rate, **not** an error rate: no human adjudication exists,
and a flagged group is only a candidate for review.  Outputs a redacted review
queue for that purpose.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.conflict_detection.conflicts import detect_pair  # noqa: E402

ART = ROOT / "experiments" / "grouped_evaluation" / "artifacts"
DEFAULT_OUT = ROOT / "results" / "conflict_detection"


def _redact(text: str) -> str:
    try:
        from experiments.privacy.redact import redact_text

        return redact_text(str(text))
    except Exception:
        return "[redaction unavailable: text withheld]"


def _bucket(n: int) -> str:
    if n <= 2:
        return "2"
    if n <= 5:
        return "3-5"
    if n <= 10:
        return "6-10"
    return "11+"


def load_group_answers(min_length: int = 25) -> tuple[dict[int, list[str]], dict]:
    meta = pd.read_csv(ART / "grouped_retrieval_metadata.csv")
    mask = np.load(ART / "grouped_retrieval_informative_mask.npy")
    if len(meta) != len(mask):
        raise SystemExit(
            f"metadata/mask length mismatch: {len(meta)} vs {len(mask)}; "
            "refusing to guess the alignment"
        )
    groups = pd.read_csv(ART / "groups.csv", usecols=["qid", "group_id"])
    frame = meta.copy()
    frame["informative"] = mask
    frame = frame.merge(groups, on="qid", how="left")
    if frame["group_id"].isna().any():
        raise SystemExit("some corpus rows have no group assignment")
    frame["answers"] = frame["answers"].fillna("").astype(str).str.strip()
    frame = frame[(frame["informative"]) & (frame["answers"].str.len() >= min_length)]

    grouped: dict[int, list[str]] = {}
    for group_id, answers in frame.groupby("group_id")["answers"]:
        grouped[int(group_id)] = list(dict.fromkeys(answers.tolist()))
    stats = {
        "corpus_rows_in_grouped_index": int(len(meta)),
        "informative_rows": int(mask.sum()),
        "rows_used_for_groups": int(len(frame)),
        "groups_in_index": int(frame["group_id"].nunique()),
    }
    return grouped, stats


def scan(grouped: dict[int, list[str]], n_groups: int = 1000, seed: int = 0,
         max_answers: int = 5) -> tuple[dict, list[dict]]:
    eligible = {group: answers for group, answers in grouped.items() if len(answers) >= 2}
    rng = random.Random(seed)
    ids = sorted(eligible)
    sampled = rng.sample(ids, min(n_groups, len(ids)))

    pair_count = 0
    conflicted_pairs = 0
    conflicted_groups = 0
    kind_pairs: Counter = Counter()
    severity_conflicts: Counter = Counter()
    by_bucket: dict[str, dict] = {}
    review: list[dict] = []

    for group_id in sampled:
        answers = eligible[group_id][:]
        if len(answers) > max_answers:
            answers = rng.sample(answers, max_answers)
        bucket = _bucket(len(answers))
        entry = by_bucket.setdefault(bucket, {"groups": 0, "groups_with_conflict": 0})
        entry["groups"] += 1
        group_has_conflict = False
        for index_a, index_b in itertools.combinations(range(len(answers)), 2):
            pair_count += 1
            conflicts = detect_pair(answers[index_a], answers[index_b], index_a, index_b)
            if not conflicts:
                continue
            conflicted_pairs += 1
            group_has_conflict = True
            for kind in {conflict.kind for conflict in conflicts}:
                kind_pairs[kind] += 1
            for conflict in conflicts:
                severity_conflicts[conflict.severity] += 1
            if len(review) < 200:
                review.append({
                    "group_id": group_id,
                    "answer_a": _redact(answers[index_a]),
                    "answer_b": _redact(answers[index_b]),
                    "kinds": ";".join(sorted({conflict.kind for conflict in conflicts})),
                    "severity": conflicts[0].severity,
                    "explanation": conflicts[0].explanation,
                })
        if group_has_conflict:
            conflicted_groups += 1
            entry["groups_with_conflict"] += 1

    for entry in by_bucket.values():
        entry["rule_fire_rate"] = round(
            entry["groups_with_conflict"] / entry["groups"], 4
        ) if entry["groups"] else None

    summary = {
        "protocol": {
            "source": "Batch A grouped retrieval metadata + informative mask",
            "grouping": "0.95 cosine question groups (Batch A)",
            "sampled_groups": len(sampled),
            "seed": seed,
            "max_answers_per_group": max_answers,
            "answer_dedup": "exact duplicate text removed within a group",
            "informative_filter": "batch A informative mask + answer length >= 25",
        },
        "eligible_groups": len(eligible),
        "sampled_groups": len(sampled),
        "pairs_evaluated": pair_count,
        "pairs_with_conflict": conflicted_pairs,
        "pair_rule_fire_rate": round(conflicted_pairs / pair_count, 4) if pair_count else None,
        "groups_with_conflict": conflicted_groups,
        "group_rule_fire_rate": round(conflicted_groups / len(sampled), 4) if sampled else None,
        "pairs_by_kind": dict(kind_pairs),
        "conflicts_by_severity": dict(severity_conflicts),
        "by_group_size": by_bucket,
        "review_rows": len(review),
        "limitations": [
            "Rule-fire rate, not an error rate: no human adjudicated these groups.",
            "Groups can legitimately contain answers for different crops/stages, so some flags are context differences.",
            "Only the train+val side of the grouped split is scanned; test rows are not included.",
            "At most five answers per group and at most 200 review rows are kept.",
        ],
    }
    return summary, review


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--groups", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-answers", type=int, default=5)
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    summary_path = out / "corpus_scan.json"
    review_path = out / "group_review.csv"
    for path in (summary_path, review_path):
        if path.exists() and not args.overwrite:
            print(f"refusing to overwrite {path} (pass --overwrite)")
            return 1

    grouped, stats = load_group_answers()
    summary, review = scan(grouped, n_groups=args.groups, seed=args.seed,
                           max_answers=args.max_answers)
    summary["dataset"] = stats
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with review_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "group_id", "answer_a", "answer_b", "kinds", "severity", "explanation",
        ])
        writer.writeheader()
        writer.writerows(review)
    print(json.dumps({
        "eligible_groups": summary["eligible_groups"],
        "sampled_groups": summary["sampled_groups"],
        "pairs_evaluated": summary["pairs_evaluated"],
        "pair_rule_fire_rate": summary["pair_rule_fire_rate"],
        "group_rule_fire_rate": summary["group_rule_fire_rate"],
        "pairs_by_kind": summary["pairs_by_kind"],
        "review_rows": summary["review_rows"],
        "written": [str(summary_path), str(review_path)],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
