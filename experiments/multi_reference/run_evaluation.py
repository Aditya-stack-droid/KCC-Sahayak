"""Run the Improvement 3 multi-reference retrieval evaluation.

Protocol
--------
P1 (primary, comparable with Batch A): unscreened grouped index, dense retrieval
on all 13,831 grouped test rows, hybrid (alpha=0.4) on the identical 3,000-query
sample, with dense recomputed on that sample for the paired comparison.

P2 (secondary, deployed behaviour): the same 3,000 queries, candidates screened
to informative answers only.

A hard reproduction gate compares the rebuilt rankings against the frozen Batch A
numbers before any new metric is reported; a mismatch stops the run.

Usage:
    OMP_NUM_THREADS=1 .venv312/bin/python experiments/multi_reference/run_evaluation.py
"""
from __future__ import annotations

import importlib.util
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import mref_embeddings as embeddings  # noqa: E402
import mref_metrics as metrics  # noqa: E402
import mref_ranking as ranking  # noqa: E402
import mref_refdata as refdata  # noqa: E402

SEED_BOOTSTRAP = 0
SEED_RANDOM_REFERENCE = 42
DRAWS = 5
TOLERANCE = 1e-3


def _env() -> dict:
    path = refdata.ROOT / "experiments" / "batch_b" / "common.py"
    spec = importlib.util.spec_from_file_location("batch_b_common", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    env = mod.environment()
    env["multi_reference_encoder"] = embeddings.MODEL_NAME
    return env


def _topic_match(ids: np.ndarray, cats: np.ndarray, y: np.ndarray) -> np.ndarray:
    labels = cats[np.clip(ids, 0, len(cats) - 1)]
    return labels == y[:, None]


def reproduction_gate(R: dict, pools: refdata.ReferencePools) -> dict:
    """Recompute Batch A's own-reference numbers from the rebuilt rankings."""
    saved = json.loads(
        (refdata.ROOT / "results" / "improved_evaluation" / "grouped_retrieval.json")
        .read_text(encoding="utf-8"))
    answers = R["answers_test"]
    meta_answers = R["meta"]["answers"].astype(str).to_numpy()
    gold_inf = pools.rows["informative"].to_numpy()

    def own_f1(ids: np.ndarray, gold: np.ndarray) -> np.ndarray:
        top = meta_answers[np.clip(ids[:, 0], 0, len(meta_answers) - 1)]
        if len(top) != len(gold):
            raise ValueError(f"ranking has {len(top)} rows but {len(gold)} gold answers")
        return np.array([refdata.token_f1(a, b) for a, b in zip(top, gold)])

    sample = R["sample"]
    dense_all = own_f1(R["dense_ids"][:, :10], answers)
    dense_sample = own_f1(R["dense_ids"][sample, :10], answers[sample])
    hybrid_sample = own_f1(R["hybrid_ids"][:, :10], answers[sample])

    cats = R["meta"]["query_type"].to_numpy()
    dense_match = _topic_match(R["dense_ids"][:, :10], cats, R["y_test"])
    hybrid_match = _topic_match(R["hybrid_ids"][:, :10], cats, R["y_test"][sample])
    y_sample = R["y_test"][sample]

    candidates = [
        ("dense all test | own-reference token-F1",
         saved["dense_all_test"]["answer_token_f1_top1_all"], float(dense_all.mean())),
        ("dense all test | own-reference token-F1 (informative gold)",
         saved["dense_all_test"]["answer_token_f1_top1_informative_gold"],
         float(dense_all[gold_inf].mean())),
        ("dense all test | topic match @1",
         saved["dense_all_test"]["category_match_top1"], float(dense_match[:, 0].mean())),
        ("dense 3k sample | own-reference token-F1",
         saved["dense_sample"]["answer_token_f1_top1_all"], float(dense_sample.mean())),
        ("dense 3k sample | topic match @1",
         saved["dense_sample"]["category_match_top1"], float(dense_match[sample, 0].mean())),
        ("hybrid 3k sample | own-reference token-F1",
         saved["hybrid_sample"]["answer_token_f1_top1_all"], float(hybrid_sample.mean())),
        ("hybrid 3k sample | topic match @1",
         saved["hybrid_sample"]["category_match_top1"], float(hybrid_match[:, 0].mean())),
    ]
    checks = []
    for name, expected, actual in candidates:
        ok = expected is not None and abs(float(expected) - float(actual)) <= TOLERANCE
        checks.append({
            "check": name,
            "batch_a_value": None if expected is None else round(float(expected), 6),
            "recomputed_value": round(float(actual), 6),
            "abs_diff": None if expected is None else round(abs(float(expected) - float(actual)), 8),
            "tolerance": TOLERANCE,
            "ok": bool(ok),
        })
    return {
        "source": "results/improved_evaluation/grouped_retrieval.json (frozen Batch A output)",
        "checks": checks,
        "ok": all(c["ok"] for c in checks),
        "n_sample_queries": int(len(sample)),
        "sample_seed": ranking.SEED,
        "note": ("Rankings are rebuilt from Batch A's own artefacts. If these do not "
                 "match, the comparison is invalid and the run stops."),
    }


def score(ids: np.ndarray, positions: np.ndarray, pools: refdata.ReferencePools,
          lookup: dict, group_vecs: dict, meta, dense_sims: np.ndarray) -> pd.DataFrame:
    """Per-query reference metrics for one ranking matrix."""
    rng = np.random.default_rng(SEED_RANDOM_REFERENCE)
    meta_answers = meta["answers"].astype(str).to_numpy()
    rows = pools.rows
    records = []
    for j, pos in enumerate(positions):
        pos = int(pos)
        row = rows.iloc[pos]
        gid = int(row["group_id"])
        aids = pools.pools.get(gid, [])
        pool_norms = [pools.norm[a] for a in aids]
        top = int(ids[j, 0])
        retrieved = refdata.normalise_answer(meta_answers[top]) if top >= 0 else ""
        own_norm = row["answer_norm"] if isinstance(row["answer_norm"], str) else ""
        own_inf = bool(row["informative"])

        own_f1 = refdata.token_f1(retrieved, own_norm) if own_norm else np.nan
        max_f1 = metrics.max_reference_f1(retrieved, pool_norms, own_norm, own_inf)
        max_other = metrics.pool_max_f1(retrieved, pool_norms, exclude_norm=own_norm or None)
        rand_other = metrics.random_other_reference_f1(
            retrieved, pool_norms, own_norm or None, rng, draws=DRAWS)

        semantic = np.nan
        if aids:
            vecs = group_vecs[gid]
            q = lookup.get(meta_answers[top])
            if q is not None:
                semantic = float(np.max(vecs @ q))

        records.append({
            "position": pos,
            "qid": int(row["qid"]),
            "group_id": gid,
            "pool_size": len(aids),
            "gold_informative": own_inf,
            "own_f1": own_f1,
            "max_f1": np.nan if max_f1 is None else float(max_f1),
            "max_other_f1": np.nan if max_other is None else float(max_other),
            "random_other_f1": np.nan if rand_other is None else float(rand_other),
            "max_semantic": semantic,
            "bucket": refdata.novelty_bucket(float(dense_sims[pos])),
        })
    frame = pd.DataFrame.from_records(records)
    frame["top1_index"] = ids[:, 0]
    return frame


def aggregates(frame: pd.DataFrame) -> dict:
    """Subset summaries for one system."""
    bucket = frame["bucket"]
    masks = {
        "all": np.ones(len(frame), dtype=bool),
        "gold_informative": frame["gold_informative"].to_numpy(dtype=bool),
        "pool_ge_2": (frame["pool_size"] >= 2).to_numpy(),
        "pool_eq_1": (frame["pool_size"] == 1).to_numpy(),
        "pool_ge_20": (frame["pool_size"] >= 20).to_numpy(),
        "pool_eq_0": (frame["pool_size"] == 0).to_numpy(),
        "gold_note": ~frame["gold_informative"].to_numpy(dtype=bool),
        "pool_eq_1_gold_informative": (frame["pool_size"] == 1).to_numpy()
        & frame["gold_informative"].to_numpy(dtype=bool),
        "pool_ge_2_gold_informative": (frame["pool_size"] >= 2).to_numpy()
        & frame["gold_informative"].to_numpy(dtype=bool),
        "novel": (bucket == "novel").to_numpy(),
        "similar": (bucket == "similar").to_numpy(),
        "near_duplicate": (bucket == "near_duplicate").to_numpy(),
    }
    out = {}
    for name, mask in masks.items():
        summary = metrics.subset_summary(frame, mask)
        summary["recovered_misses"] = metrics.recovered_misses(frame[mask])
        out[name] = summary
    out["pool_ge_2"]["pool_stats"] = {
        "mean_pool_size": round(float(frame.loc[masks["pool_ge_2"], "pool_size"].mean()), 4),
        "median_pool_size": float(frame.loc[masks["pool_ge_2"], "pool_size"].median()),
        "max_pool_size": int(frame.loc[masks["pool_ge_2"], "pool_size"].max()),
    }
    return out


def main() -> None:
    t0 = time.time()
    refdata.MR_ART.mkdir(parents=True, exist_ok=True)
    refdata.RESULTS.mkdir(parents=True, exist_ok=True)

    df = refdata.load_grouped_corpus()
    pools = refdata.build_reference_pools(df)
    stats = refdata.pool_stats(pools)
    refdata.write_reference_artifacts(pools, stats)
    print(stats["summary_line"])
    print(f"split checks: {refdata.split_checks(df)}")

    R = ranking.build_rankings(df)
    if not np.array_equal(pools.rows["row_id"].to_numpy(), R["test_idx"]):
        raise SystemExit("reference pools are not aligned with the grouped test order")

    gate = reproduction_gate(R, pools)
    for c in gate["checks"]:
        print(f"  gate {'OK ' if c['ok'] else 'FAIL'} {c['check']}: "
              f"batch A {c['batch_a_value']} vs rebuilt {c['recomputed_value']}")
    if not gate["ok"]:
        raise SystemExit("reproduction gate failed: refusing to report new metrics")

    # ---------------- answer embeddings (references + retrieved candidates)
    ref_ids = [a for a in pools.refs["answer_id"].tolist()]
    ref_vecs = embeddings.encode_texts([pools.raw[a] for a in ref_ids],
                                       "reference_answer_embeddings")
    ref_pos = {a: i for i, a in enumerate(ref_ids)}
    group_vecs = {g: ref_vecs[[ref_pos[a] for a in aids]]
                  for g, aids in pools.pools.items()}

    # ---------------- protocols
    n_test = R["n_test"]
    sample = R["sample"]
    dense_all_ids = R["dense_ids"][:, :10]
    dense_sample_ids = R["dense_ids"][sample, :10]
    hybrid_ids = R["hybrid_ids"][:, :10]
    dense_all_screened = ranking.screened_top_k(R["dense_ids"], R["valid"])
    dense_sample_screened = dense_all_screened[sample]
    hybrid_screened = ranking.screened_top_k(R["hybrid_ids"], R["valid"])

    protocols = {
        "P1_dense_all": ("dense", dense_all_ids, np.arange(n_test), "all 13,831 test rows",
                         "unscreened"),
        "P1_dense_sample": ("dense", dense_sample_ids, sample, "fixed 3,000-query sample",
                            "unscreened"),
        "P1_hybrid_sample": ("hybrid", hybrid_ids, sample, "fixed 3,000-query sample",
                             "unscreened"),
        "P2_dense_sample": ("dense", dense_sample_screened, sample,
                            "fixed 3,000-query sample", "informative-only candidates"),
        "P2_hybrid_sample": ("hybrid", hybrid_screened, sample,
                             "fixed 3,000-query sample", "informative-only candidates"),
    }

    meta_answers = R["meta"]["answers"].astype(str).to_numpy()
    top1_texts = []
    for _, (_, ids, _, _, _) in protocols.items():
        top1_texts.extend(meta_answers[np.clip(ids[:, 0], 0, len(meta_answers) - 1)].tolist())
    lookup = embeddings.lookup(top1_texts, "retrieved_answer_embeddings")

    frames = {}
    for name, (system, ids, positions, desc, condition) in protocols.items():
        t = time.time()
        frames[name] = score(ids, positions, pools, lookup, group_vecs,
                             R["meta"], R["dense_sims"][:, 0])
        print(f"scored {name}: {len(frames[name])} rows in {time.time()-t:.1f}s")

    systems = {}
    for name, (system, ids, positions, desc, condition) in protocols.items():
        systems[name] = {
            "system": system,
            "rows": desc,
            "candidate_condition": condition,
            "n_rows": int(len(frames[name])),
            "subsets": aggregates(frames[name]),
        }

    # ---------------- paired comparisons (identical queries, paired by row)
    p1_dense = frames["P1_dense_all"]
    p1_dsample = frames["P1_dense_sample"]
    p1_hybrid = frames["P1_hybrid_sample"]
    p2_dsample = frames["P2_dense_sample"]
    p2_hybrid = frames["P2_hybrid_sample"]

    paired = {
        "P1_dense_all | max-other-reference vs own-reference": metrics.paired_bootstrap(
            p1_dense["max_other_f1"].to_numpy(), p1_dense["own_f1"].to_numpy(),
            n_boot=2000, seed=SEED_BOOTSTRAP),
        "P1_dense_all | max-reference vs own-reference": metrics.paired_bootstrap(
            p1_dense["max_f1"].to_numpy(), p1_dense["own_f1"].to_numpy(),
            n_boot=2000, seed=SEED_BOOTSTRAP),
        "P1_dense_all | random-other-reference vs own-reference": metrics.paired_bootstrap(
            p1_dense["random_other_f1"].to_numpy(), p1_dense["own_f1"].to_numpy(),
            n_boot=2000, seed=SEED_BOOTSTRAP),
        "P1_dense_all | max-other vs own | informative own answer only":
            metrics.paired_bootstrap(
                p1_dense.loc[p1_dense["gold_informative"], "max_other_f1"].to_numpy(),
                p1_dense.loc[p1_dense["gold_informative"], "own_f1"].to_numpy(),
                n_boot=2000, seed=SEED_BOOTSTRAP),
        "P1_sample | hybrid vs dense | own-reference": metrics.paired_bootstrap(
            p1_hybrid["own_f1"].to_numpy(), p1_dsample["own_f1"].to_numpy(),
            n_boot=2000, seed=SEED_BOOTSTRAP),
        "P1_sample | hybrid vs dense | max-other-reference": metrics.paired_bootstrap(
            p1_hybrid["max_other_f1"].to_numpy(), p1_dsample["max_other_f1"].to_numpy(),
            n_boot=2000, seed=SEED_BOOTSTRAP),
        "P1_sample | hybrid vs dense | max-reference": metrics.paired_bootstrap(
            p1_hybrid["max_f1"].to_numpy(), p1_dsample["max_f1"].to_numpy(),
            n_boot=2000, seed=SEED_BOOTSTRAP),
        "P1_sample | hybrid vs dense | max-other | informative own answer only":
            metrics.paired_bootstrap(
                p1_hybrid.loc[p1_hybrid["gold_informative"], "max_other_f1"].to_numpy(),
                p1_dsample.loc[p1_dsample["gold_informative"], "max_other_f1"].to_numpy(),
                n_boot=2000, seed=SEED_BOOTSTRAP),
        "P2_sample | hybrid vs dense | own-reference": metrics.paired_bootstrap(
            p2_hybrid["own_f1"].to_numpy(), p2_dsample["own_f1"].to_numpy(),
            n_boot=2000, seed=SEED_BOOTSTRAP),
        "P2_sample | hybrid vs dense | max-other-reference": metrics.paired_bootstrap(
            p2_hybrid["max_other_f1"].to_numpy(), p2_dsample["max_other_f1"].to_numpy(),
            n_boot=2000, seed=SEED_BOOTSTRAP),
    }
    for label, result in paired.items():
        result["comparison"] = label

    note_rows = p1_dense[~p1_dense["gold_informative"]]
    diagnostics = {
        "non_informative_own_rows": {
            "n": int(len(note_rows)),
            "share_of_test_rows": round(float(len(note_rows) / len(p1_dense)), 4),
            "mean_own_f1_against_note": round(float(note_rows["own_f1"].mean()), 6),
            "mean_max_other_f1": round(float(note_rows["max_other_f1"].mean()), 6),
            "note": (
                "On these rows the own logged 'answer' is a call-log note, not advice. The "
                "original metric still scores the retrieved answer against it, which is why "
                "single-reference F1 can exceed the informative-reference maximum on rows "
                "with one reference. This baseline is not an answer."),
        },
        "semantic_scale": {
            "note": (
                "Max semantic similarity grows with pool size because it is a maximum over "
                "more candidates; it is a wording-closeness statistic, not correctness and "
                "not a calibrated probability."),
        },
    }

    out = {
        "improvement": "Improvement 3 - multi-reference retrieval evaluation",
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "protocol": {
            "split": "Batch A grouped corpus split (StratifiedGroupKFold, seed 0)",
            "index": "Batch A grouped dense index (train+val only)",
            "fusion": f"alpha={ranking.ALPHA}, same as deployed and Batch A",
            "reference_definition": (
                "Distinct informative advisor answers logged on the test side of the "
                "query's question group. No reference is invented and no pool contains "
                "an index-side row."),
            "reference_source": "test side of the grouped split only",
            "answer_text_policy": (
                "Answer text is not written to any result file; reports and artefacts "
                "carry ids, hashes and metrics only."),
            "P1": "unscreened grouped index (comparable with Batch A)",
            "P2": "informative-only candidates (deployed screening)",
            "hybrid_sample": int(len(sample)),
        },
        "seeds": {
            "hybrid_sample": ranking.SEED,
            "bootstrap": SEED_BOOTSTRAP,
            "random_reference_draws": SEED_RANDOM_REFERENCE,
            "random_reference_draw_count": DRAWS,
        },
        "split_checks": refdata.split_checks(df),
        "reference_pools": stats,
        "reproduction_gate": gate,
        "systems": systems,
        "paired_comparisons": paired,
        "diagnostics": diagnostics,
        "limitations": [
            "References are historical advisor answers, not verified-correct advice; "
            "agreement with them is not agricultural correctness.",
            "Near-duplicate grouping is transitive at cosine >= 0.95, so some large "
            "groups may merge related but not identical questions.",
            "Contradictory answers can coexist in one pool (Improvement 1 shows such "
            "conflicts exist); max-reference F1 rewards matching any of them.",
            "Semantic similarity measures wording closeness only; it is not correctness "
            "and is reported separately from F1.",
            "No human relevance or expert agronomic judgement exists for these pools "
            "(Batch B Phase 2 annotation remains PENDING).",
            "Max-reference F1 and max semantic similarity both rise mechanically with pool "
            "size, because both are maxima over more candidates; the random-reference "
            "control is reported for that reason.",
            "On rows whose own logged answer is a call-log note, the original single-"
            "reference metric scores against that note; those rows are reported separately "
            "and excluded from the informative-own paired comparison.",
            "Only one grouped split seed was used, because the index rebuild is "
            "expensive; results are conditional on that split.",
        ],
    }

    out_path = refdata.RESULTS / "metrics.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    p1_dense.to_csv(refdata.RESULTS / "per_query_dense_all.csv", index=False)
    paired_frame = pd.DataFrame({
        "position": sample,
        "qid": p1_dsample["qid"].to_numpy(),
        "group_id": p1_dsample["group_id"].to_numpy(),
        "pool_size": p1_dsample["pool_size"].to_numpy(),
        "bucket": p1_dsample["bucket"].to_numpy(),
        "dense_own_f1": p1_dsample["own_f1"].to_numpy(),
        "dense_max_f1": p1_dsample["max_f1"].to_numpy(),
        "dense_max_other_f1": p1_dsample["max_other_f1"].to_numpy(),
        "dense_random_other_f1": p1_dsample["random_other_f1"].to_numpy(),
        "dense_max_semantic": p1_dsample["max_semantic"].to_numpy(),
        "hybrid_own_f1": p1_hybrid["own_f1"].to_numpy(),
        "hybrid_max_f1": p1_hybrid["max_f1"].to_numpy(),
        "hybrid_max_other_f1": p1_hybrid["max_other_f1"].to_numpy(),
        "hybrid_random_other_f1": p1_hybrid["random_other_f1"].to_numpy(),
        "hybrid_max_semantic": p1_hybrid["max_semantic"].to_numpy(),
        "screened_dense_own_f1": p2_dsample["own_f1"].to_numpy(),
        "screened_dense_max_other_f1": p2_dsample["max_other_f1"].to_numpy(),
        "screened_hybrid_own_f1": p2_hybrid["own_f1"].to_numpy(),
        "screened_hybrid_max_other_f1": p2_hybrid["max_other_f1"].to_numpy(),
    })
    paired_frame.to_csv(refdata.RESULTS / "per_query_paired_sample.csv", index=False)

    env = _env()
    out["environment"] = env
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print(f"\nwrote {out_path} in {time.time()-t0:.1f}s")
    d = systems["P1_dense_all"]["subsets"]
    print(f"P1 dense all: own F1 {d['all']['own_reference_f1']['mean']}, "
          f"max-other F1 {d['pool_ge_2']['max_other_reference_f1']['mean']}")


if __name__ == "__main__":
    main()
