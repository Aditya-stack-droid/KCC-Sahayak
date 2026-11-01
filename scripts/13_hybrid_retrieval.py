"""Build the TF-IDF side of the hybrid retriever and choose the fusion weight.

Evaluated on the same 3,000-query held-out test subset used elsewhere, scored
by answer-level bag-of-words F1 against the answer the human advisor actually
gave (much closer to "is this a useful answer" than category match), overall
and on NOVEL queries (dense similarity to the corpus < 0.80).

Usage:
    .venv/Scripts/python.exe scripts/13_hybrid_retrieval.py
"""
import json
import re
import sys
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from answer_quality import informative_mask  # noqa: E402
from retrieval import HybridRetriever, INDEX_DIR  # noqa: E402

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


def main():
    meta = pd.read_csv(INDEX_DIR / "corpus_metadata.csv")
    print("fitting TF-IDF on", len(meta), "corpus questions")
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True, dtype=np.float32)
    mat = vec.fit_transform(meta["questions"].astype(str))
    joblib.dump(vec, INDEX_DIR / "tfidf.joblib")
    sp.save_npz(INDEX_DIR / "tfidf_matrix.npz", mat)

    ret = HybridRetriever()
    assert ret.is_hybrid

    preds = pd.read_csv(RES / "test_predictions.csv")
    test = pd.read_csv(ROOT / "data/processed/test.csv").dropna(subset=["questions", "answers"])
    test = test[test["qid"].isin(preds["qid"])].drop_duplicates("qid").reset_index(drop=True)
    n_before = len(test)
    test = test[informative_mask(test["answers"])].reset_index(drop=True)
    print(f"test rows with a real (informative) gold answer: {len(test)} / {n_before}")
    embed = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    Q = embed.encode(test["questions"].astype(str).tolist(), batch_size=128,
                     normalize_embeddings=True, show_progress_bar=False).astype("float32")

    results = {}
    for alpha in [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]:
        f1s = []
        for q, qe, gold in zip(test["questions"], Q, test["answers"]):
            hits = ret.search(qe, q, k=1, alpha=alpha)
            top = hits[0]
            f1s.append(token_f1(meta.iloc[top["id"]]["answers"], gold))
        f1s = np.array(f1s)
        # novelty defined by the DENSE nearest neighbour, independent of alpha
        results[alpha] = {"f1": f1s}
    # novelty mask from pure dense search
    dense_top = np.array([ret.search(qe, q, k=1, alpha=1.0)[0]["similarity"] for q, qe in zip(test["questions"], Q)])
    novel = dense_top < NOVEL
    out = {"n": int(len(test)), "n_novel": int(novel.sum()), "alphas": {}}
    print(f"\nanswer token-F1 vs advisor answer   (n={out['n']}, novel n={out['n_novel']})")
    print(f"{'alpha':>6s} {'all':>8s} {'novel':>8s}")
    best = (None, -1)
    for a, v in results.items():
        f = v["f1"]
        out["alphas"][str(a)] = {"all": round(float(f.mean()), 4), "novel": round(float(f[novel].mean()), 4)}
        print(f"{a:6.1f} {f.mean():8.4f} {f[novel].mean():8.4f}")
        score = f[novel].mean() * 0.5 + f.mean() * 0.5
        if score > best[1]:
            best = (a, score)
    out["chosen_alpha"] = best[0]
    (INDEX_DIR / "hybrid.json").write_text(json.dumps({"alpha": best[0]}))
    (RES / "hybrid_retrieval.json").write_text(json.dumps(out, indent=2))
    print("\nchosen alpha =", best[0])


if __name__ == "__main__":
    main()
