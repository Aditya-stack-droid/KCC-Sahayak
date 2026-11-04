"""Run the out-of-scope detection evaluation (Improvement 4).

Approach A: mean cosine to the k nearest agricultural questions (Batch A index,
train+val only).
Approach B: logistic regression on MiniLM embeddings, trained on agricultural
corpus queries plus authored synthetic out-of-scope queries.

Thresholds are selected on validation data only, under a fixed policy: maximise
out-of-scope recall subject to rejecting at most 2% of real agricultural
validation queries. The test split is scored once.

Usage:
    OMP_NUM_THREADS=1 .venv312/bin/python experiments/out_of_scope/run_oos_evaluation.py
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
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import oos_data  # noqa: E402
import oos_detectors as det  # noqa: E402
import oos_metrics as om  # noqa: E402

TARGET_FRR = 0.02
MARGIN_KNN = 0.03
MARGIN_LR = 0.05
N_TRAIN = 20000
N_VAL = 5000
BENCH_IN_SCOPE = 144
DATASET_SEED = 42
BOOTSTRAP_SEED = 0
REGISTERS = ("en", "hi", "hinglish")


def _env() -> dict:
    path = oos_data.ROOT / "experiments" / "batch_b" / "common.py"
    spec = importlib.util.spec_from_file_location("batch_b_common_oos", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    env = mod.environment()
    env["encoder"] = oos_data.MODEL_NAME
    return env


def main() -> None:
    t0 = time.time()
    oos_data.ART.mkdir(parents=True, exist_ok=True)
    oos_data.RESULTS.mkdir(parents=True, exist_ok=True)
    (HERE / "data").mkdir(parents=True, exist_ok=True)

    corpus = oos_data.load_corpus_splits()
    synth = oos_data.split_synthetic(oos_data.build_synthetic_frame())
    ambiguous = oos_data.build_ambiguous_frame()
    mixed = oos_data.build_mixed_frame()

    synth.to_csv(HERE / "data" / "synthetic_out_of_scope.csv", index=False)
    ambiguous.to_csv(HERE / "data" / "ambiguous_queries.csv", index=False)
    mixed.to_csv(HERE / "data" / "mixed_intent_queries.csv", index=False)

    observed = pd.concat([synth, ambiguous, mixed], ignore_index=True)
    emb_syn = oos_data.encode_texts(observed["text"].tolist(),
                                    "synthetic_query_embeddings")
    pos = {qid: i for i, qid in enumerate(observed["query_id"])}

    emb = np.load(oos_data.GROUPED_ART / "corpus_embeddings.npy")
    qids = np.load(oos_data.GROUPED_ART / "corpus_qids.npy")
    if len(emb) != len(corpus) or not np.array_equal(qids, corpus["qid"].to_numpy()):
        raise SystemExit("corpus embeddings do not align with the corpus")

    rng = np.random.default_rng(DATASET_SEED)
    train_pool = np.flatnonzero(corpus["split"].to_numpy() == "train")
    val_pool = np.flatnonzero(corpus["split"].to_numpy() == "val")
    test_pos = np.flatnonzero(corpus["split"].to_numpy() == "test")
    train_pos = rng.permutation(train_pool)[:N_TRAIN]
    val_pos = rng.permutation(val_pool)[:N_VAL]

    knn = det.KNNRejector()
    knn_val = knn.scores(emb[val_pos])
    knn_test = knn.scores(emb[test_pos])
    knn_syn = knn.scores(emb_syn)

    lr = det.EmbeddingLRDetector(C=1.0, seed=0)
    train_ids = [pos[q] for q in synth.loc[synth["split"] == "train", "query_id"]]
    lr.fit(emb[train_pos], emb_syn[train_ids])
    lr.save(oos_data.ART / "oos_lr_detector.joblib")
    lr_val = lr.scores(emb[val_pos])
    lr_test = lr.scores(emb[test_pos])
    lr_syn = lr.scores(emb_syn)

    val_oos_ids = [pos[q] for q in synth.loc[synth["split"] == "val", "query_id"]]
    test_oos_ids = [pos[q] for q in synth.loc[synth["split"] == "test", "query_id"]]
    thr_knn = om.select_threshold(knn_val, knn_syn[val_oos_ids], TARGET_FRR)
    thr_lr = om.select_threshold(lr_val, lr_syn[val_oos_ids], TARGET_FRR)

    bench_rng = np.random.default_rng(DATASET_SEED)
    bench_idx = bench_rng.choice(len(test_pos), size=min(BENCH_IN_SCOPE, len(test_pos)),
                                 replace=False)
    bench_idx.sort()

    y_true = np.r_[np.zeros(len(bench_idx), dtype=int),
                   np.ones(len(test_oos_ids), dtype=int)]
    benchmark = {}
    for name, scores_in, scores_out, thr, margin in (
            ("knn_distance", knn_test[bench_idx], knn_syn[test_oos_ids],
             thr_knn["threshold"], MARGIN_KNN),
            ("embedding_lr", lr_test[bench_idx], lr_syn[test_oos_ids],
             thr_lr["threshold"], MARGIN_LR)):
        scores = np.r_[scores_in, scores_out]
        rejected = scores < thr
        metrics = om.binary_metrics(y_true, rejected)
        metrics["auroc"] = round(float(roc_auc_score(y_true, -scores)), 4)
        metrics["f1_ci95"] = om.f1_ci(y_true, rejected, seed=BOOTSTRAP_SEED)
        metrics["mean_score_in_scope"] = round(float(np.mean(scores_in)), 4)
        metrics["mean_score_out_of_scope"] = round(float(np.mean(scores_out)), 4)
        benchmark[name] = metrics

    full_in_scope = {}
    for name, scores, thr in (("knn_distance", knn_test, thr_knn["threshold"]),
                              ("embedding_lr", lr_test, thr_lr["threshold"])):
        rejected = scores < thr
        full_in_scope[name] = {
            "n": int(len(scores)),
            "n_rejected": int(rejected.sum()),
            "false_rejection_rate": round(float(rejected.mean()), 6),
            "note": "All grouped-test agricultural queries, not a sample.",
        }

    test_synth = synth[synth["split"] == "test"]
    by_register = {}
    for register in REGISTERS:
        ids = [pos[q] for q in test_synth.loc[test_synth["register"] == register, "query_id"]]
        entry = {"n": len(ids)}
        for name, scores, thr in (("knn_distance", knn_syn, thr_knn["threshold"]),
                                  ("embedding_lr", lr_syn, thr_lr["threshold"])):
            entry[name] = {
                "n_rejected": int((scores[ids] < thr).sum()),
                "rejection_rate": round(float((scores[ids] < thr).mean()), 4),
            }
        by_register[register] = entry

    def describe(frame: pd.DataFrame, scores: np.ndarray, thr: float, margin: float) -> dict:
        rows = []
        for _, row in frame.iterrows():
            d = det.decide(scores[pos[row["query_id"]]], thr, margin, row["text"])
            rows.append({
                "query_id": row["query_id"],
                "register": row["register"],
                "score": round(float(d.score), 4),
                "decision": d.decision,
                "accepted": bool(d.accepted),
                "reason": d.reason,
            })
        frame_out = pd.DataFrame(rows)
        return {
            "n": int(len(frame_out)),
            "decision_counts": {k: int(v) for k, v in
                                frame_out["decision"].value_counts().items()},
            "n_rejected": int((~frame_out["accepted"]).sum()),
            "rejection_rate": round(float((~frame_out["accepted"]).mean()), 4),
            "rows": frame_out,
        }

    ambiguous_out = {
        "knn_distance": describe(ambiguous, knn_syn, thr_knn["threshold"], MARGIN_KNN),
        "embedding_lr": describe(ambiguous, lr_syn, thr_lr["threshold"], MARGIN_LR),
    }
    mixed_out = {
        "knn_distance": describe(mixed, knn_syn, thr_knn["threshold"], MARGIN_KNN),
        "embedding_lr": describe(mixed, lr_syn, thr_lr["threshold"], MARGIN_LR),
    }

    sample_texts = corpus.loc[val_pos, "questions"].astype(str).tolist()
    evidence = {
        "corpus_val_sample": {
            "n": len(sample_texts),
            "with_evidence": int(sum(1 for t in sample_texts if det.agricultural_evidence(t))),
        },
        "synthetic_test_out_of_scope": {
            "n": len(test_synth),
            "with_evidence": int(sum(1 for t in test_synth["text"] if det.agricultural_evidence(t))),
        },
        "ambiguous": {
            "n": len(ambiguous),
            "with_evidence": int(sum(1 for t in ambiguous["text"] if det.agricultural_evidence(t))),
        },
        "mixed_intent": {
            "n": len(mixed),
            "with_evidence": int(sum(1 for t in mixed["text"] if det.agricultural_evidence(t))),
        },
        "note": "Conservative English glossary rule; used to flag, never to accept silently.",
    }

    sensitivity = {}
    for name, scores in (("knn_distance", knn_test[bench_idx]), ("embedding_lr", lr_test[bench_idx])):
        scores_all = np.r_[scores, (knn_syn if name == "knn_distance" else lr_syn)[test_oos_ids]]
        grid = (np.round(np.linspace(0.30, 0.80, 11), 2) if name == "knn_distance"
                else np.round(np.linspace(0.05, 0.95, 10), 2))
        sensitivity[name] = om.threshold_sweep(y_true, scores_all, grid)

    register_mix = corpus.loc[corpus["split"] == "test", "questions"].sample(
        n=min(5000, int((corpus["split"] == "test").sum())), random_state=DATASET_SEED)
    register_counts = register_mix.map(oos_data.detect_register).value_counts().to_dict()

    per_query = pd.DataFrame({
        "query_id": ([corpus.iloc[p]["qid"] for p in test_pos[bench_idx]]
                     + list(test_synth["query_id"])),
        "source": ["corpus"] * len(bench_idx) + ["synthetic"] * len(test_oos_ids),
        "register": (["en"] * len(bench_idx)
                     + list(test_synth["register"])),
        "label": ["in_scope"] * len(bench_idx) + ["out_of_scope"] * len(test_oos_ids),
        "knn_score": np.round(np.r_[knn_test[bench_idx], knn_syn[test_oos_ids]], 4),
        "lr_score": np.round(np.r_[lr_test[bench_idx], lr_syn[test_oos_ids]], 4),
    })
    per_query["knn_rejected"] = per_query["knn_score"] < thr_knn["threshold"]
    per_query["lr_rejected"] = per_query["lr_score"] < thr_lr["threshold"]
    per_query["knn_decision"] = [
        det.decide(s, thr_knn["threshold"], MARGIN_KNN, "").decision
        for s in per_query["knn_score"]]
    per_query["lr_decision"] = [
        det.decide(s, thr_lr["threshold"], MARGIN_LR, "").decision
        for s in per_query["lr_score"]]
    per_query.to_csv(oos_data.RESULTS / "per_query_benchmark.csv", index=False)

    in_scope_rows = pd.DataFrame({
        "qid": corpus.iloc[test_pos]["qid"].to_numpy(),
        "group_id": corpus.iloc[test_pos]["group_id"].to_numpy(),
        "knn_score": np.round(knn_test, 4),
        "lr_score": np.round(lr_test, 4),
    })
    in_scope_rows["knn_rejected"] = in_scope_rows["knn_score"] < thr_knn["threshold"]
    in_scope_rows["lr_rejected"] = in_scope_rows["lr_score"] < thr_lr["threshold"]
    in_scope_rows.to_csv(oos_data.RESULTS / "per_query_in_scope.csv", index=False)

    def public(block: dict) -> dict:
        return {k: v for k, v in block.items() if k != "rows"}

    metrics = {
        "improvement": "Improvement 4 - out-of-scope detection",
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "protocol": {
            "task": "binary in-scope vs out-of-scope, positive class = out-of-scope",
            "approaches": {
                "knn_distance": "mean cosine to the k=5 nearest grouped-index questions (train+val)",
                "embedding_lr": "logistic regression (C=1, balanced) over MiniLM embeddings",
            },
            "threshold_policy": (f"highest threshold with validation false-rejection rate <= "
                                 f"{TARGET_FRR}; selection on validation only"),
            "ambiguity_band": {"knn_distance": MARGIN_KNN, "embedding_lr": MARGIN_LR},
            "benchmark": (f"{len(bench_idx)} agricultural grouped-test queries (seeded sample) "
                          f"+ {len(test_oos_ids)} synthetic out-of-scope queries"),
            "in_scope_source": "Batch A grouped split, test side",
            "out_of_scope_source": "authored synthetic queries, grouped by family",
            "decisions": "in_scope / out_of_scope / ambiguous; binary metrics use score vs threshold",
        },
        "seeds": {"dataset": DATASET_SEED, "bootstrap": BOOTSTRAP_SEED},
        "data": {
            "corpus_rows": int(len(corpus)),
            "corpus_split_counts": {k: int(v) for k, v in
                                    corpus["split"].value_counts().items()},
            "corpus_registers_test_sample": {str(k): int(v) for k, v in register_counts.items()},
            "synthetic": {
                "n_queries": int(len(synth)),
                "n_families": int(synth["family_id"].nunique()),
                "n_categories": int(synth["category"].nunique()),
                "per_split_queries": {k: int(v) for k, v in
                                      synth["split"].value_counts().items()},
                "per_split_families": {k: int(v) for k, v in
                                       synth.groupby("split")["family_id"].nunique().items()},
                "per_register": {k: int(v) for k, v in
                                 synth["register"].value_counts().items()},
                "synthetic": True,
            },
            "ambiguous_queries": int(len(ambiguous)),
            "mixed_intent_queries": int(len(mixed)),
            "split_checks": {
                "synthetic_families_spanning_splits": int(
                    (synth.groupby("family_id")["split"].nunique() > 1).sum()),
                "corpus_groups_spanning_splits": int(
                    (corpus.groupby("group_id")["split"].nunique() > 1).sum()),
            },
        },
        "thresholds": {"knn_distance": thr_knn, "embedding_lr": thr_lr},
        "benchmark": benchmark,
        "full_in_scope_test": full_in_scope,
        "by_register_out_of_scope": by_register,
        "ambiguous": {k: public(v) for k, v in ambiguous_out.items()},
        "mixed_intent": {k: public(v) for k, v in mixed_out.items()},
        "evidence": evidence,
        "threshold_sensitivity": sensitivity,
        "limitations": [
            "Out-of-scope queries are authored synthetic text, not real KCC traffic; "
            "these are synthetic-negative results, not deployment estimates.",
            "The synthetic set is small (180 queries / 60 families) and same-author, "
            "so confidence intervals are wide and vocabulary overlap with training "
            "templates cannot be excluded.",
            "In-scope corpus queries are English by the project register detector; "
            "no Hindi or Hinglish in-scope evaluation data exists, so no multilingual "
            "in-scope claim is made. Hindi/Hinglish appear only as synthetic negatives.",
            "Thresholds were selected on validation data; the test split was scored once, "
            "but the ambiguity band widths are design choices, not tuned quantities.",
            "The agricultural-evidence override uses an English glossary only; it cannot "
            "flag Devanagari or Roman-Hindi farm terms.",
            "No human annotation of scope exists; agreement with the synthetic labels is "
            "not a validated measure of real-world out-of-scope detection.",
            "The kNN reference set and the corpus are the same logged call text, so "
            "in-scope queries are lexically and semantically close to the index by "
            "construction; that is what the detector exploits.",
        ],
        "environment": _env(),
    }
    (oos_data.RESULTS / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8")
    for name in ("knn_distance", "embedding_lr"):
        b = benchmark[name]
        print(f"{name}: P={b['precision']} R={b['recall']} F1={b['f1']} "
              f"FAR={b['false_acceptance_rate']} FRR={b['false_rejection_rate']} "
              f"threshold={metrics['thresholds'][name]['threshold']}")
    print(f"wrote {oos_data.RESULTS / 'metrics.json'} in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
