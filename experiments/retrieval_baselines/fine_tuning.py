#!/usr/bin/env python
"""Fine-tuned embedding retrieval — feasibility analysis + bounded experiment.

Supervision available in this project is weak and noisy: the grouped **train**
queries paired with their own logged advisor answers. There are no human
relevance judgements, ~8.75% of answers are call-log notes, and the evaluation
relevance proxy is a 10-class topic match. This module documents that honestly,
then runs a **small, reproducible** experiment:

* train ``paraphrase-multilingual-MiniLM-L12-v2`` with
  ``MultipleNegativesRankingLoss`` on grouped-train (query, informative answer)
  pairs only (test rows never seen);
* evaluate base vs fine-tuned encoder on a **reduced** 20k-document corpus drawn
  from grouped train, with grouped-test queries — a paired comparison on the
  same corpus. Absolutes are not comparable to the full-corpus numbers.

If training cannot run in this environment, the error is recorded and no result
is fabricated.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (HERE, ROOT, ROOT / "src", ROOT / "experiments"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from common import (  # noqa: E402
    SEED, environment, evaluate_rankings, load_corpus, load_split, sample_positions,
    test_frame, write_json,
)

BASE_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_OUT = ROOT / "results" / "retrieval_baselines" / "fine_tuning"
ARTIFACT_DIR = HERE / "artifacts"
MAX_PAIRS = 4000
CORPUS_SIZE = 20000
QUERIES = 500
EPOCHS = 1
BATCH = 32


def _feasibility() -> dict:
    from answer_quality import informative_mask

    corpus = load_corpus()
    split = load_split().to_numpy()
    train = corpus[split == "train"].reset_index(drop=True)
    informative = informative_mask(train["answers"])
    pairs = train[informative]
    digest = hashlib.sha1("".join(pairs["questions"].astype(str).head(MAX_PAIRS)).encode()).hexdigest()
    return {
        "grouped_train_rows": int(len(train)),
        "informative_train_rows": int(informative.sum()),
        "call_log_note_share": round(float(1 - informative.mean()), 4),
        "proposed_pairs": int(min(MAX_PAIRS, len(pairs))),
        "pair_digest_sha1": digest,
        "supervision": "weak: query paired with its own logged advisor answer; no human relevance labels",
        "suitability": (
            "limited: weak, noisy positives and a 10-class topic proxy mean a fine-tune can "
            "overfit label noise and may not improve topical retrieval; run as a bounded check only"
        ),
    }


def _train(pairs, out_dir: Path) -> dict:
    import torch
    from torch.utils.data import DataLoader
    from sentence_transformers import InputExample, SentenceTransformer, losses

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    model = SentenceTransformer(BASE_MODEL)
    examples = [InputExample(texts=[q, a]) for q, a in pairs]
    loader = DataLoader(examples, shuffle=True, batch_size=BATCH)
    loss = losses.MultipleNegativesRankingLoss(model)
    t0 = time.perf_counter()
    model.fit(train_objectives=[(loader, loss)], epochs=EPOCHS, warmup_steps=10,
              show_progress_bar=False)
    secs = time.perf_counter() - t0
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save(str(out_dir / "model"))
    return {"train_seconds": round(secs, 1), "steps": len(loader) * EPOCHS,
            "checkpoint": str((out_dir / "model").relative_to(ROOT))}


def _encode(model, texts) -> np.ndarray:
    return model.encode(list(texts), batch_size=64, normalize_embeddings=True,
                        show_progress_bar=False).astype("float32")


def _eval_reduced(corpus_texts, corpus_types, q_texts, q_types, encoder) -> np.ndarray:
    import faiss

    C = _encode(encoder, corpus_texts)
    index = faiss.IndexFlatIP(C.shape[1])
    index.add(C)
    Q = _encode(encoder, q_texts)
    _, ids = index.search(Q, 10)
    # relevance = corpus topic == query topic (same proxy as the full evaluation)
    labels = corpus_types[ids] == np.asarray(q_types)[:, None]
    n = len(q_texts)
    rec = lambda r, k: float(np.mean([any(row[:k]) for row in r])) if n else None
    rr = float(np.mean([next((1.0 / (i + 1) for i, h in enumerate(row[:10]) if h), 0.0) for row in r])) if n else None
    return {"n": n, "recall_at_1": round(rec(labels, 1), 4), "recall_at_5": round(rec(labels, 5), 4),
            "recall_at_10": round(rec(labels, 10), 4), "mrr_at_10": round(rr, 4),
            "topic_match_at_1": round(float(labels[:, 0].mean()), 4),
            "_per_query": labels[:, 0].astype(float).tolist()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--feasibility-only", action="store_true")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    target = out / "metrics.json"
    if target.exists() and not args.overwrite:
        print(f"refusing to overwrite {target} (pass --overwrite)")
        return 1

    feasibility = _feasibility()
    result = {"system": "fine_tuned_encoder", "base_model": BASE_MODEL,
              "protocol": {"seed": SEED, "max_pairs": MAX_PAIRS, "corpus_size": CORPUS_SIZE,
                           "queries": QUERIES, "epochs": EPOCHS, "batch": BATCH,
                           "loss": "MultipleNegativesRankingLoss",
                           "split": "grouped train for training; grouped test for evaluation"},
              "feasibility": feasibility, "status": "feasibility_only",
              "environment": environment()}
    if args.feasibility_only:
        write_json(target, result)
        print(json.dumps(feasibility, indent=2))
        return 0

    try:
        from answer_quality import informative_mask
        from sentence_transformers import SentenceTransformer

        corpus = load_corpus()
        split = load_split().to_numpy()
        train = corpus[split == "train"].reset_index(drop=True)
        informative = informative_mask(train["answers"])
        pairs_df = train[informative].head(MAX_PAIRS)
        pairs = list(zip(pairs_df["questions"].astype(str), pairs_df["answers"].astype(str)))
        train_info = _train(pairs, ARTIFACT_DIR)

        # reduced corpus (grouped train) and grouped-test queries
        rng = np.random.default_rng(SEED)
        train_positions = np.flatnonzero(split == "train")
        corpus_sample = rng.choice(train_positions, size=min(CORPUS_SIZE, len(train_positions)), replace=False)
        corpus_sample.sort()
        corpus_texts = corpus["questions"].astype(str).iloc[corpus_sample].tolist()
        corpus_types = corpus["query_type"].to_numpy()[corpus_sample]

        test_df, _ = test_frame()
        qpos = sample_positions(QUERIES)
        q_texts = test_df["questions"].astype(str).iloc[qpos].tolist()
        q_types = test_df["query_type"].to_numpy()[qpos]

        base = SentenceTransformer(BASE_MODEL)
        ft = SentenceTransformer(str(ARTIFACT_DIR / "model"))
        base_m = _eval_reduced(corpus_texts, corpus_types, q_texts, q_types, base)
        ft_m = _eval_reduced(corpus_texts, corpus_types, q_texts, q_types, ft)
        base_per = base_m.pop("_per_query")
        ft_per = ft_m.pop("_per_query")
        diff = np.asarray(ft_per) - np.asarray(base_per)
        from common import paired_bootstrap

        result.update({
            "status": "completed",
            "training": train_info,
            "reduced_corpus": {"documents": len(corpus_texts), "queries": len(q_texts),
                               "note": "reduced tractability experiment; not comparable to full-corpus numbers"},
            "systems": {"base": base_m, "fine_tuned": ft_m},
            "paired_vs_base_topic_at_1": paired_bootstrap(diff),
            "limitations": [
                "Weak supervision: query paired with its own logged answer; no human relevance labels.",
                "Reduced 20k-document corpus and 500 queries; absolutes not comparable to full-corpus results.",
                "Relevance is a 10-class topical proxy; a fine-tune can overfit label noise.",
                "One epoch, fixed seed; not a hyperparameter search.",
            ],
        })
        print(json.dumps({"base": base_m, "fine_tuned": ft_m,
                          "paired": result["paired_vs_base_topic_at_1"]}, indent=2))
    except Exception as exc:  # pragma: no cover - environment-dependent
        result.update({
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
            "blocker": (
                "sentence-transformers `fit` requires the `datasets` package, which is not part of "
                "the project environment. It was NOT installed, to avoid adding a dependency the "
                "original application does not need. The feasibility analysis and full training / "
                "evaluation configuration above are complete and reproducible once `datasets` is "
                "available in an isolated environment."
            ),
            "next_step": "pip install datasets in an isolated env, then re-run fine_tuning.py --overwrite",
        })
        print("fine-tuning could not complete:", result["error"])

    write_json(target, result)
    print("written:", target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
