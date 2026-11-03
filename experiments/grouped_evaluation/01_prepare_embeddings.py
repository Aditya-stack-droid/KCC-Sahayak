"""Build a question embedding cache for the full cleaned corpus.

The shipped retrieval index covers train+val (124,987 rows). To group and
re-split the *whole* cleaned corpus we need embeddings for the remaining
13,888 rows. Re-encoding all 138,875 questions is unnecessary; this script
reuses the shipped embeddings by qid and encodes only the missing rows.

Outputs (new files only):
  artifacts/corpus_embeddings.npy   float32 [n, 384], L2-normalised
  artifacts/corpus_qids.npy         int64 [n]
  artifacts/corpus_embeddings.json  provenance and counts
"""
from __future__ import annotations

import json
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from common import ART, EMBED_DIM, EMBED_MODEL, INDEX_DIR, ensure_dirs, load_corpus  # noqa: E402


def main(force: bool = False) -> None:
    ensure_dirs()
    out_emb = ART / "corpus_embeddings.npy"
    if out_emb.exists() and not force:
        print(f"{out_emb} already exists; use --force to rebuild")
        return

    df = load_corpus()
    qids = df["qid"].to_numpy(dtype=np.int64)
    n = len(df)
    print(f"corpus rows: {n}")

    meta = pd.read_csv(INDEX_DIR / "corpus_metadata.csv", usecols=["qid"])
    shipped = np.load(INDEX_DIR / "embeddings.npy", mmap_mode="r")
    if len(meta) != len(shipped):
        raise SystemExit("metadata and embedding rows disagree in the shipped index")
    qid_to_row = {int(q): i for i, q in enumerate(meta["qid"].tolist())}

    emb = np.zeros((n, EMBED_DIM), dtype="float32")
    have = np.zeros(n, dtype=bool)
    for i, q in enumerate(qids.tolist()):
        row = qid_to_row.get(int(q))
        if row is not None:
            emb[i] = np.asarray(shipped[row], dtype="float32")
            have[i] = True
    missing_idx = np.flatnonzero(~have)
    print(f"reused {int(have.sum())} shipped embeddings; encoding {len(missing_idx)} missing rows")

    t0 = time.time()
    if len(missing_idx):
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(EMBED_MODEL)
        encoded = model.encode(
            df["questions"].iloc[missing_idx].astype(str).tolist(),
            batch_size=64, normalize_embeddings=True, show_progress_bar=True,
        ).astype("float32")
        if encoded.shape[1] != EMBED_DIM:
            raise SystemExit(f"unexpected embedding dimension {encoded.shape[1]}")
        emb[missing_idx] = encoded
    elapsed = time.time() - t0

    norms = np.linalg.norm(emb, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-3):
        raise SystemExit("some embeddings are not L2-normalised; refusing to write")
    if np.isnan(emb).any():
        raise SystemExit("NaN embeddings produced")

    np.save(out_emb, emb)
    np.save(ART / "corpus_qids.npy", qids)
    (ART / "corpus_embeddings.json").write_text(json.dumps({
        "model": EMBED_MODEL,
        "n": int(n),
        "dim": int(EMBED_DIM),
        "reused_from_shipped_index": int(have.sum()),
        "encoded_rows": int(len(missing_idx)),
        "encode_seconds": round(elapsed, 1),
        "normalised": True,
        "source": "data/processed/full_labeled.csv + models/retrieval_index/embeddings.npy",
    }, indent=2), encoding="utf-8")
    print(f"wrote {out_emb} ({emb.nbytes / 1e6:.1f} MB) in {elapsed:.1f}s")


if __name__ == "__main__":
    main(force="--force" in sys.argv)
