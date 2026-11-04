"""Cached sentence embeddings for reference/retrieved answer similarity.

Uses the project's existing encoder (``paraphrase-multilingual-MiniLM-L12-v2``),
the same model Batch A grouped and indexed with. Vectors are L2-normalised, so a
dot product is a cosine similarity. Cache keys are content hashes: if the input
texts change, the cache is rebuilt instead of silently reused.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np

from mref_refdata import MR_ART

MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"


def _digest(texts) -> str:
    h = hashlib.sha1()
    for t in texts:
        h.update(str(t).encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


def encode_texts(texts, name: str, force: bool = False) -> np.ndarray:
    """Encode ``texts`` in order, reusing a content-addressed cache when valid."""
    texts = [str(t) for t in texts]
    digest = _digest(texts)
    vec_path = MR_ART / f"{name}.npy"
    meta_path = MR_ART / f"{name}.json"
    if vec_path.exists() and meta_path.exists() and not force:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        arr = np.load(vec_path)
        if meta.get("sha1") == digest and len(arr) == len(texts):
            return arr.astype("float32")

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(MODEL_NAME)
    arr = model.encode(texts, batch_size=64, normalize_embeddings=True,
                       show_progress_bar=False).astype("float32")
    MR_ART.mkdir(parents=True, exist_ok=True)
    np.save(vec_path, arr)
    meta_path.write_text(json.dumps({
        "name": name,
        "model": MODEL_NAME,
        "n": int(len(texts)),
        "dim": int(arr.shape[1]),
        "sha1": digest,
        "normalised": True,
        "note": "Texts are not stored; only their hash and vectors.",
    }, indent=2), encoding="utf-8")
    return arr


def lookup(texts, name: str) -> dict:
    """Map each distinct text to its embedding vector."""
    uniq = sorted(set(str(t) for t in texts))
    arr = encode_texts(uniq, name)
    return {t: arr[i] for i, t in enumerate(uniq)}
