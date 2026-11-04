"""Hybrid retriever: dense multilingual embeddings (FAISS) fused with TF-IDF.

Dense search is cross-lingual (a Hindi query can find an English log entry)
but fuzzy on in-domain wording; TF-IDF is the reverse. Candidates from both are
pooled, scored by

    fused = alpha * dense_cosine + (1 - alpha) * tfidf_cosine

and ranked by the fused score. The reported `similarity` stays the dense
cosine so the confidence gate's threshold keeps one meaning across languages.
"""
import json
from pathlib import Path

import faiss
import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp

from answer_quality import informative_mask

ROOT = Path(__file__).resolve().parents[1]
INDEX_DIR = ROOT / "models" / "retrieval_index"

# Files the dense/TF-IDF index needs. `models/` is gitignored, so a fresh clone must
# build them (scripts/03_build_retrieval_index.py) before the app can start.
REQUIRED_INDEX_FILES = ("kcc_index.faiss", "corpus_metadata.csv", "embeddings.npy")


def missing_index_files(index_dir: Path = INDEX_DIR) -> list[str]:
    """Names of required retrieval-index files absent from ``index_dir``."""
    index_dir = Path(index_dir)
    return [name for name in REQUIRED_INDEX_FILES if not (index_dir / name).exists()]


class HybridRetriever:
    def __init__(self, index_dir: Path = INDEX_DIR):
        index_dir = Path(index_dir)
        missing = missing_index_files(index_dir)
        if missing:
            raise FileNotFoundError(
                f"Retrieval index is missing at '{index_dir}' "
                f"(absent: {', '.join(missing)}).\n"
                "Model artefacts are not committed; build them once with:\n"
                "    python scripts/03_build_retrieval_index.py"
            )
        self.index = faiss.read_index(str(index_dir / "kcc_index.faiss"))
        self.meta = pd.read_csv(index_dir / "corpus_metadata.csv")
        self.emb = np.load(index_dir / "embeddings.npy", mmap_mode="r")
        mask_path = index_dir / "informative_mask.npy"
        if mask_path.exists():
            self.valid = np.load(mask_path)
        else:
            self.valid = informative_mask(self.meta["answers"])
            np.save(mask_path, self.valid)
        self.alpha = 1.0
        self.tfidf = None
        self.tfidf_matrix = None
        tf_path = index_dir / "tfidf.joblib"
        if tf_path.exists() and (index_dir / "tfidf_matrix.npz").exists():
            self.tfidf = joblib.load(tf_path)
            self.tfidf_matrix = sp.load_npz(index_dir / "tfidf_matrix.npz").tocsr()
            cfg = index_dir / "hybrid.json"
            if cfg.exists():
                self.alpha = float(json.loads(cfg.read_text()).get("alpha", 0.5))
            else:
                self.alpha = 0.5
        self.n = len(self.meta)

    @property
    def is_hybrid(self) -> bool:
        return self.tfidf is not None

    def search(self, q_emb: np.ndarray, q_text: str, k: int = 3, alpha: float | None = None,
               pool: int = 200, informative_only: bool = True):
        """q_emb: (dim,) L2-normalised dense query embedding. Returns a list of
        dicts with id, similarity (dense), tfidf, fused, sorted by fused desc."""
        a = self.alpha if alpha is None else alpha
        dsim, didx = self.index.search(q_emb.reshape(1, -1).astype("float32"), pool)
        cand = set(int(i) for i in didx[0] if i >= 0)
        qv = None
        if self.is_hybrid and a < 1.0:
            qv = self.tfidf.transform([q_text])
            if qv.nnz:
                row = (self.tfidf_matrix @ qv.T).toarray().ravel()
                top = np.argpartition(-row, min(pool, len(row) - 1))[:pool]
                cand |= set(int(i) for i in top if row[i] > 0)
        if informative_only:
            cand = {c for c in cand if self.valid[c]}
        if not cand:
            return []
        cand = np.fromiter(cand, dtype=np.int64)
        dense = np.asarray(self.emb[cand]) @ q_emb.astype("float32")
        if qv is not None and qv.nnz:
            tf = np.asarray((self.tfidf_matrix[cand] @ qv.T).todense()).ravel()
        else:
            tf = np.zeros(len(cand), dtype="float32")
        fused = a * dense + (1.0 - a) * tf
        order = np.argsort(-fused)[:k]
        return [
            {"id": int(cand[i]), "similarity": float(dense[i]), "tfidf": float(tf[i]),
             "fused": float(fused[i])}
            for i in order
        ]
