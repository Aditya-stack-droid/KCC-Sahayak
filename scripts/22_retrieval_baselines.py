"""Retrieval baselines, including a KisanQRS-style one (Rehman et al., 2023).

KisanQRS maps a farmer query to the most similar logged query by combining lexical and
semantic similarity, then ranks the answers. We cannot reproduce its LSTM mapping layer
(no code or weights are public), so the "KisanQRS-style" row here is the same idea reduced to
its retrieval core: an equal-weight lexical + semantic score, no answer screening.

Systems, scored on the 3,000-query held-out test subset against the answer the human advisor
really gave (token F1), on the rows whose gold answer is itself informative:

  lexical only        TF-IDF cosine (alpha = 0)
  dense only          multilingual MiniLM cosine (alpha = 1)
  KisanQRS-style      alpha = 0.5, every logged answer eligible
  ours (hybrid)       tuned alpha, every logged answer eligible
  ours + screening    tuned alpha, call-log notes excluded (the deployed retriever)

Also reported: how often the returned answer is a call-log note / carries dosage or chemical
text, i.e. how often the system would hand the farmer something unusable or unsafe.

Outputs: results/retrieval_baselines.json
Usage:   .venv/Scripts/python.exe scripts/22_retrieval_baselines.py
"""
import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from answer_quality import informative_mask, is_informative  # noqa: E402
from confidence_gate import answer_risk  # noqa: E402
from retrieval import HybridRetriever  # noqa: E402

RES = ROOT / "results"
NOVEL = 0.80


def token_f1(a, b):
    ta, tb = re.findall(r"\w+", str(a).lower()), re.findall(r"\w+", str(b).lower())
    if not ta or not tb:
        return 0.0
    n = sum((Counter(ta) & Counter(tb)).values())
    if n == 0:
        return 0.0
    p, r = n / len(ta), n / len(tb)
    return 2 * p * r / (p + r)


def boot_ci(x, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, float)
    m = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(n)]
    return [round(float(np.percentile(m, 2.5)), 4), round(float(np.percentile(m, 97.5)), 4)]


def main():
    ret = HybridRetriever()
    assert ret.is_hybrid, "run scripts/13_hybrid_retrieval.py first"
    tuned = ret.alpha
    preds = pd.read_csv(RES / "test_predictions.csv")
    test = pd.read_csv(ROOT / "data/processed/test.csv").dropna(subset=["questions", "answers"])
    test = test[test["qid"].isin(preds["qid"])].drop_duplicates("qid").reset_index(drop=True)
    test = test[informative_mask(test["answers"])].reset_index(drop=True)
    embed = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    Q = embed.encode(test["questions"].astype(str).tolist(), batch_size=128,
                     normalize_embeddings=True, show_progress_bar=False).astype("float32")

    systems = {
        "lexical only (TF-IDF)": dict(alpha=0.0, informative_only=False),
        "dense only (MiniLM)": dict(alpha=1.0, informative_only=False),
        "KisanQRS-style (alpha=0.5, no screening)": dict(alpha=0.5, informative_only=False),
        f"ours: hybrid (alpha={tuned}), no screening": dict(alpha=tuned, informative_only=False),
        f"ours: hybrid (alpha={tuned}) + note screening (deployed)": dict(alpha=tuned, informative_only=True),
    }
    dense_top = np.array([ret.search(qe, q, k=1, alpha=1.0, informative_only=False)[0]["similarity"]
                          for q, qe in zip(test["questions"], Q)])
    novel = dense_top < NOVEL
    out = {"n": int(len(test)), "n_novel": int(novel.sum()), "tuned_alpha": tuned, "systems": {}}
    print(f"n={out['n']} (novel {out['n_novel']})  tuned alpha={tuned}")
    print(f"{'system':62s} {'F1':>6s} {'novelF1':>8s} {'note%':>6s} {'risky%':>7s}")
    for name, kw in systems.items():
        f1, note, risky = [], [], []
        for q, qe, gold in zip(test["questions"], Q, test["answers"]):
            hits = ret.search(qe, q, k=1, **kw)
            ans = ret.meta.iloc[hits[0]["id"]]["answers"] if hits else ""
            f1.append(token_f1(ans, gold))
            note.append(not is_informative(ans))
            risky.append(answer_risk(ans) is not None)
        f1, note, risky = np.array(f1), np.array(note), np.array(risky)
        out["systems"][name] = {"f1": round(float(f1.mean()), 4), "f1_ci": boot_ci(f1),
                                "f1_novel": round(float(f1[novel].mean()), 4),
                                "call_log_note_rate": round(float(note.mean()), 4),
                                "dosage_or_chemical_rate": round(float(risky.mean()), 4)}
        print(f"{name:62s} {f1.mean():6.3f} {f1[novel].mean():8.3f} {100 * note.mean():6.1f} {100 * risky.mean():7.1f}")
    out["caveat"] = ("KisanQRS-style row is a simplified re-implementation of its retrieval idea, not the paper's "
                     "LSTM system. Scored on rows whose advisor answer is itself informative.")
    (RES / "retrieval_baselines.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("saved results/retrieval_baselines.json")


if __name__ == "__main__":
    main()
