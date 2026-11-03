"""Shared paths and helpers for the group-aware evaluation pipeline.

This package is additive: it reads the original corpus, annotations, model
artefacts and result files, and writes only under
``experiments/grouped_evaluation/artifacts/`` and
``results/improved_evaluation/``. No original file is written.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ART = HERE / "artifacts"
OUT = ROOT / "results" / "improved_evaluation"
DATA = ROOT / "data" / "processed"
ANNOTATION = ROOT / "data" / "annotation"
GOLD = ROOT / "data" / "gold" / "gold_queries.csv"
INDEX_DIR = ROOT / "models" / "retrieval_index"
INTENT_MODEL = ROOT / "models" / "intent_embed_lr" / "model.joblib"

EMBED_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
EMBED_DIM = 384
SEEDS = [0, 1, 2, 3, 4]
NEAR_DUP = 0.95
SIMILAR = 0.80
LABELS = [
    "Animal Husbandry", "Cultural Practices", "Fisheries", "Government Scheme",
    "Market Price", "Nutrient Management", "Other", "Plant Protection",
    "Varietal Information", "Weather",
]


def ensure_dirs() -> None:
    ART.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)


def text_hash(texts) -> str:
    h = hashlib.sha1()
    for t in texts:
        h.update(str(t).encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


def encode_cached(texts, key: str, force: bool = False) -> np.ndarray:
    """Encode with the existing multilingual encoder, caching by content hash."""
    ensure_dirs()
    emb_path = ART / f"emb_{key}.npy"
    meta_path = ART / f"emb_{key}.json"
    digest = text_hash(texts)
    if emb_path.exists() and meta_path.exists() and not force:
        meta = json.loads(meta_path.read_text())
        arr = np.load(emb_path)
        if meta.get("sha1") == digest and len(arr) == len(texts):
            return arr.astype("float32")
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(EMBED_MODEL)
    arr = model.encode(
        [str(t) for t in texts], batch_size=64, normalize_embeddings=True,
        show_progress_bar=False,
    ).astype("float32")
    np.save(emb_path, arr)
    meta_path.write_text(json.dumps({
        "key": key, "n": int(len(texts)), "sha1": digest, "model": EMBED_MODEL,
        "dim": int(arr.shape[1]),
    }, indent=2))
    return arr


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def load_corpus():
    import pandas as pd

    df = pd.read_csv(DATA / "full_labeled.csv")
    for col in ("qid", "questions", "answers", "query_type"):
        if col not in df.columns:
            raise SystemExit(f"full_labeled.csv is missing column {col!r}")
    return df


def load_annotations():
    import pandas as pd

    ann = pd.read_csv(ANNOTATION / "annotated.csv")
    aug = pd.read_csv(ANNOTATION / "augmented.csv")
    return ann, aug


def load_gold():
    import pandas as pd

    return pd.read_csv(GOLD)
