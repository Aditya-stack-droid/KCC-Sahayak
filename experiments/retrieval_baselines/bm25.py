"""Standalone BM25 (Okapi) baseline — pure Python, no new dependency.

Implements the standard Okapi BM25 scoring with the ``rank_bm25`` negative-IDF
epsilon handling, so it matches the established behaviour without adding a
dependency (``rank_bm25`` is not installed in the project environment).

Tokenisation: Unicode word runs (``[\\w\\u0900-\\u097F]+``, lower-cased). This keeps
Devanagari words with their vowel signs intact, but it does **not** stem, does not
handle Hindi morphology and treats Devanagari digits as separate tokens — so a
good score on this corpus (English/Romanised) says nothing about Hindi recall.

All results are deterministic: scores are sorted by (-score, doc_id).
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (HERE, ROOT, ROOT / "experiments"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from common import (  # noqa: E402
    SEED, environment, evaluate_rankings, load_index_metadata, sample_positions,
    test_frame, write_json,
)

_TOKEN = re.compile(r"[\w\u0900-\u097F]+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(str(text).lower()) if t.strip("_")]


class BM25:
    """Okapi BM25 over an in-memory document list."""

    def __init__(self, docs: list[str], k1: float = 1.5, b: float = 0.75, epsilon: float = 0.25):
        self.k1, self.b = k1, b
        tokenized = [tokenize(d) for d in docs]
        self.n = len(tokenized)
        self.doc_len = np.array([len(t) for t in tokenized], dtype=np.float64)
        self.avgdl = float(self.doc_len.mean()) if self.n else 0.0
        postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        df: Counter = Counter()
        for i, toks in enumerate(tokenized):
            tf = Counter(toks)
            for term, f in tf.items():
                postings[term].append((i, f))
                df[term] += 1
        self.postings = dict(postings)
        idf = {t: float(np.log((self.n - d + 0.5) / (d + 0.5) + 1.0)) for t, d in df.items()}
        if idf:
            avg_idf = sum(idf.values()) / len(idf)
            self.idf = {t: (v if v >= 0 else epsilon * avg_idf) for t, v in idf.items()}
        else:
            self.idf = {}

    def search(self, query: str, k: int = 10) -> list[tuple[int, float]]:
        terms = tokenize(query)
        if not terms:
            return []
        scores: dict[int, float] = defaultdict(float)
        for term in terms:
            if term not in self.postings:
                continue
            idf = self.idf[term]
            for doc, f in self.postings[term]:
                dl = self.doc_len[doc]
                denom = f + self.k1 * (1.0 - self.b + self.b * dl / (self.avgdl or 1.0))
                scores[doc] += idf * (f * (self.k1 + 1.0)) / denom
        ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[:k]
        return ranked


def build_from_metadata() -> tuple[BM25, object, dict]:
    meta = load_index_metadata()
    t0 = time.perf_counter()
    bm25 = BM25(meta["questions"].astype(str).tolist())
    build_s = time.perf_counter() - t0
    return bm25, meta, {"build_seconds": round(build_s, 1), "n_docs": len(meta),
                        "vocab": len(bm25.idf)}


def run(sample: int, k: int, out: Path) -> dict:
    bm25, meta, build_info = build_from_metadata()
    test_df, _ = test_frame()
    pos = sample_positions(sample)
    q_texts = test_df["questions"].astype(str).iloc[pos].tolist()

    ranked = np.full((len(pos), k), -1, dtype=np.int64)
    lat = []
    for i, q in enumerate(q_texts):
        t0 = time.perf_counter()
        hits = bm25.search(q, k)
        lat.append((time.perf_counter() - t0) * 1000.0)
        for j, (doc, _score) in enumerate(hits[:k]):
            ranked[i, j] = doc

    metrics = evaluate_rankings(ranked, test_df.iloc[pos].reset_index(drop=True), meta, len(pos))
    per_query = metrics.pop("_per_query")
    lat_arr = np.asarray(lat)
    result = {
        "system": "bm25",
        "protocol": {"sample": int(len(pos)), "seed": SEED, "relevance": "weak-label topic match"},
        "build": build_info,
        "metrics": metrics,
        "latency_ms": {"p50": round(float(np.percentile(lat_arr, 50)), 3),
                       "p95": round(float(np.percentile(lat_arr, 95)), 3),
                       "mean": round(float(lat_arr.mean()), 3)},
        "per_query": per_query,
        "tokenisation": "Unicode word runs (lowercase); no stemming; Devanagari marks preserved",
        "environment": environment(),
    }
    write_json(out / "metrics.json", result)
    print({k: result[k] for k in ("metrics", "latency_ms", "build")})
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sample", type=int, default=2000)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--out", default=str(ROOT / "results" / "retrieval_baselines" / "bm25"))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    out = Path(args.out)
    if (out / "metrics.json").exists() and not args.overwrite:
        print(f"refusing to overwrite {out / 'metrics.json'} (pass --overwrite)")
        return 1
    run(args.sample, args.k, out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
