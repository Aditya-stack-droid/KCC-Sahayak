"""Reference pools for multi-reference retrieval evaluation (Improvement 3).

A *reference pool* is the set of distinct, informative answers that KCC advisors
actually logged for rows in the same question group as an evaluation query.

Construction rules
------------------
* Grouping is Batch A's question-identity grouping
  (``experiments/grouped_evaluation/artifacts/groups.csv``): identical
  normalised questions plus transitive MiniLM cosine >= 0.95.
* Only rows on the **test** side of the grouped split contribute references,
  so a pool can never contain an answer that is also in the retrieval index.
* Only *informative* answers are references (``src/answer_quality``): call-log
  notes such as "explained in details" are not answers.
* References are deduplicated by normalised text, so one wording counts once
  even when several advisors recorded it.

Nothing here is a gold standard. A pool is a range of historical advisor
answers; agreement with it is not evidence of agronomic correctness.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
GROUPED_ART = ROOT / "experiments" / "grouped_evaluation" / "artifacts"
DATA = ROOT / "data" / "processed"
RESULTS = ROOT / "results" / "multi_reference"
MR_ART = HERE / "artifacts"

SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from answer_quality import informative_mask  # noqa: E402


def _load_module(name: str, path: Path):
    """Load a single file as a module without adding its directory to sys.path."""
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# Batch A's metric primitives, reused unchanged (not copied).
grouped_metrics = _load_module(
    "kcc_grouped_metrics", ROOT / "experiments" / "grouped_evaluation" / "metrics.py")

token_f1 = grouped_metrics.token_f1
bootstrap_ci = grouped_metrics.bootstrap_ci
novelty_bucket = grouped_metrics.novelty_bucket

TOKEN_RE = re.compile(r"\w+")
REQUIRED_COLUMNS = ("qid", "questions", "answers", "query_type", "group_id", "split")


def normalise_answer(text) -> str:
    """Lowercase word tokens joined by single spaces (same tokens token_f1 uses)."""
    return " ".join(TOKEN_RE.findall(str(text).lower()))


def sha1_short(text: str, n: int = 16) -> str:
    return hashlib.sha1(str(text).encode("utf-8")).hexdigest()[:n]


def prepare_corpus_frame(raw: pd.DataFrame) -> pd.DataFrame:
    """Add row ids, normalised answers and the informativeness screen."""
    missing = [c for c in REQUIRED_COLUMNS if c not in raw.columns]
    if missing:
        raise ValueError(f"corpus frame is missing columns: {missing}")
    df = raw.copy().reset_index(drop=True)
    df["row_id"] = np.arange(len(df), dtype=np.int64)
    df["answer_norm"] = df["answers"].map(normalise_answer)
    df["informative"] = informative_mask(df["answers"])
    return df


def load_grouped_corpus() -> pd.DataFrame:
    """Original corpus joined to Batch A's group ids and grouped split."""
    df = pd.read_csv(DATA / "full_labeled.csv")
    split = pd.read_csv(GROUPED_ART / "corpus_split.csv")
    groups = pd.read_csv(GROUPED_ART / "groups.csv")
    if not (len(df) == len(split) == len(groups)):
        raise SystemExit("corpus, split and group files disagree in length")
    if not np.array_equal(df["qid"].to_numpy(), split["qid"].to_numpy()):
        raise SystemExit("corpus_split.csv qids do not align with the corpus")
    if not np.array_equal(df["qid"].to_numpy(), groups["qid"].to_numpy()):
        raise SystemExit("groups.csv qids do not align with the corpus")
    df = df.copy()
    df["group_id"] = groups["group_id"].to_numpy()
    df["split"] = split["split"].to_numpy()
    return prepare_corpus_frame(df)


def assert_no_split_leakage(df: pd.DataFrame) -> None:
    """A question group must live entirely inside one split."""
    spans = df.groupby("group_id")["split"].nunique()
    bad = spans[spans > 1]
    if len(bad):
        raise ValueError(
            f"{len(bad)} question groups span more than one split "
            f"(e.g. group {int(bad.index[0])}); reference pools would leak")
    unknown = set(df["split"].unique()) - {"train", "val", "test"}
    if unknown:
        raise ValueError(f"unexpected split labels: {sorted(unknown)}")


@dataclass
class ReferencePools:
    """Distinct informative test-side answers, indexed by question group."""

    rows: pd.DataFrame            # test rows, aligned to the grouped test order
    pools: dict                   # group_id -> [answer_id, ...] (row order)
    refs: pd.DataFrame            # answer_id, group_id, row_id, n_words, sha1
    raw: dict                     # answer_id -> original answer text
    norm: dict                    # answer_id -> normalised answer text
    own_id: dict                  # row_id -> answer_id or None (informative own answer)

    def pool_norms(self, group_id: int) -> list:
        return [self.norm[a] for a in self.pools.get(int(group_id), [])]

    def pool_answer_ids(self, group_id: int) -> list:
        return list(self.pools.get(int(group_id), []))


def build_reference_pools(df: pd.DataFrame) -> ReferencePools:
    """Build the per-group reference pools from the test side only."""
    assert_no_split_leakage(df)
    test = df[df["split"] == "test"].reset_index(drop=True)

    pools = defaultdict(list)
    raw, norm = {}, {}
    rep = {}
    records = []
    for row in test[test["informative"]].itertuples(index=False):
        key = (int(row.group_id), row.answer_norm)
        if key in rep:
            continue
        aid = f"a{int(row.row_id)}"
        rep[key] = aid
        pools[int(row.group_id)].append(aid)
        raw[aid] = str(row.answers)
        norm[aid] = row.answer_norm
        records.append({
            "answer_id": aid,
            "group_id": int(row.group_id),
            "row_id": int(row.row_id),
            "n_words": len(str(row.answers).split()),
            "text_sha1": sha1_short(row.answers),
        })

    refs = pd.DataFrame.from_records(
        records, columns=["answer_id", "group_id", "row_id", "n_words", "text_sha1"])
    if len(refs):
        refs = refs.sort_values("row_id").reset_index(drop=True)

    own_id = {}
    for row in test.itertuples(index=False):
        if not row.informative:
            own_id[int(row.row_id)] = None
            continue
        own_id[int(row.row_id)] = rep.get((int(row.group_id), row.answer_norm))

    return ReferencePools(
        rows=test, pools=dict(pools), refs=refs, raw=raw, norm=norm, own_id=own_id)


def pool_stats(pools: ReferencePools, seed: int = 42, sample_groups: int = 500,
               cap: int = 12) -> dict:
    """Availability and diversity diagnostics for the reference pools.

    ``supports_multi_reference`` is a pre-registered criterion, not a result:
    at least half of test rows must have >=2 distinct references, and at least
    half of sampled multi-reference groups must contain genuinely different
    answers (max pairwise token-F1 below 0.80).
    """
    rows = pools.rows
    n_rows = int(len(rows))
    sizes = rows["group_id"].map(lambda g: len(pools.pools.get(int(g), []))).to_numpy()
    sizes = sizes.astype(int)

    multi = [g for g, aids in pools.pools.items() if len(aids) > 1]
    rng = np.random.default_rng(seed)
    if len(multi) > sample_groups:
        picked = rng.choice(len(multi), size=sample_groups, replace=False)
        sample = [sorted(multi)[int(i)] for i in sorted(picked.tolist())]
    else:
        sample = sorted(multi)

    diversity = []
    for g in sample:
        aids = pools.pools[g][:cap]
        texts = [pools.norm[a] for a in aids]
        best = 0.0
        for i, a in enumerate(texts):
            for b in texts[i + 1:]:
                best = max(best, token_f1(a, b))
        diversity.append(best)
    diversity_arr = np.asarray(diversity, dtype=float) if diversity else np.zeros(0)

    share_ge2 = float((sizes >= 2).mean()) if n_rows else 0.0
    share_ge5 = float((sizes >= 5).mean()) if n_rows else 0.0
    share_zero = float((sizes == 0).mean()) if n_rows else 0.0
    group_sizes = np.asarray([len(a) for a in pools.pools.values()], dtype=int)
    share_in_giant = float((sizes >= 100).mean()) if n_rows else 0.0
    share_diverse = float((diversity_arr < 0.80).mean()) if diversity_arr.size else 0.0

    out = {
        "n_test_rows": n_rows,
        "n_test_groups": int(rows["group_id"].nunique()),
        "n_groups_with_references": int(len(pools.pools)),
        "n_unique_references": int(len(pools.refs)),
        "n_rows_with_reference": int((sizes >= 1).sum()),
        "n_rows_with_2plus_references": int((sizes >= 2).sum()),
        "n_rows_with_5plus_references": int((sizes >= 5).sum()),
        "n_rows_without_reference": int((sizes == 0).sum()),
        "share_rows_with_reference": round(float((sizes >= 1).mean()), 6) if n_rows else None,
        "share_rows_with_2plus_references": round(share_ge2, 6),
        "share_rows_with_5plus_references": round(share_ge5, 6),
        "share_rows_without_reference": round(share_zero, 6),
        "references_per_test_row": {
            "mean": round(float(sizes.mean()), 4) if n_rows else None,
            "median": float(np.median(sizes)) if n_rows else None,
            "p90": float(np.percentile(sizes, 90)) if n_rows else None,
            "max": int(sizes.max()) if n_rows else None,
            "note": "Row-weighted: rows inside very large near-duplicate groups pull the mean up.",
        },
        "references_per_group": {
            "n_groups": int(len(group_sizes)),
            "mean": round(float(group_sizes.mean()), 4) if group_sizes.size else None,
            "median": float(np.median(group_sizes)) if group_sizes.size else None,
            "p90": float(np.percentile(group_sizes, 90)) if group_sizes.size else None,
            "max": int(group_sizes.max()) if group_sizes.size else None,
        },
        "skew": {
            "n_rows_in_groups_with_ge_100_references": int((sizes >= 100).sum()),
            "share_rows_in_groups_with_ge_100_references": round(share_in_giant, 4),
            "note": ("The reference distribution is highly skewed: a few very large "
                     "transitive near-duplicate clusters hold hundreds of answers. "
                     "Per-group and per-row statistics are reported separately."),
        },
        "reference_diversity": {
            "n_groups_sampled": int(len(sample)),
            "seed": seed,
            "cap_per_group": cap,
            "mean_max_pairwise_token_f1": round(float(diversity_arr.mean()), 4)
            if diversity_arr.size else None,
            "median_max_pairwise_token_f1": round(float(np.median(diversity_arr)), 4)
            if diversity_arr.size else None,
            "share_groups_below_0_80": round(share_diverse, 4),
            "note": ("Pairwise token-F1 between reference answers in the same group. "
                     "Low values mean the group holds genuinely different advice, not "
                     "restatements of one answer."),
        },
        "criteria": {
            "min_share_rows_with_2plus_references": 0.50,
            "min_share_diverse_multi_ref_groups": 0.50,
        },
        "supports_multi_reference": bool(share_ge2 >= 0.50 and share_diverse >= 0.50),
    }
    out["summary_line"] = (
        f"{out['n_unique_references']:,} distinct references in "
        f"{out['n_groups_with_references']:,} test groups; "
        f"{out['share_rows_with_2plus_references']:.1%} of test rows have >=2; "
        f"mean max pairwise reference F1 {out['reference_diversity']['mean_max_pairwise_token_f1']}; "
        f"supports_multi_reference={out['supports_multi_reference']}")
    return out


def write_reference_artifacts(pools: ReferencePools, stats: dict) -> None:
    """Write pool structure and diagnostics (metadata only, never answer text)."""
    MR_ART.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    (MR_ART / "reference_pools.json").write_text(
        json.dumps({str(g): aids for g, aids in sorted(pools.pools.items())}, indent=1),
        encoding="utf-8")
    pools.refs.to_csv(MR_ART / "reference_answers.csv", index=False)
    (MR_ART / "pool_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")


def split_checks(df: pd.DataFrame) -> dict:
    """Compact evidence that reference pools are split-safe."""
    spans = df.groupby("group_id")["split"].nunique()
    return {
        "groups_spanning_splits": int((spans > 1).sum()),
        "n_groups": int(len(spans)),
        "split_counts": {k: int(v) for k, v in df["split"].value_counts().items()},
        "reference_source_split": "test only",
    }
