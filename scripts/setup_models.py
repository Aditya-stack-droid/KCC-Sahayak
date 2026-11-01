"""First-run setup for a fresh clone of the repository.

The trained intent head and the small retrieval files are committed. The two large retrieval files
(embeddings.npy and kcc_index.faiss, ~190 MB each) are not, because GitHub rejects files over 100 MB.
This script rebuilds them deterministically from the committed corpus_metadata.csv, in the same row
order, so retrieval behaves exactly as on the original machine.

Usage:  python scripts/setup_models.py        (about 10-20 minutes on a laptop CPU, once)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
INDEX_DIR = ROOT / "models" / "retrieval_index"
INTENT = ROOT / "models" / "intent_embed_lr" / "model.joblib"
EMBED = "paraphrase-multilingual-MiniLM-L12-v2"


def main():
    problems = [str(p.relative_to(ROOT)) for p in
                (INTENT, INDEX_DIR / "corpus_metadata.csv", INDEX_DIR / "tfidf.joblib",
                 INDEX_DIR / "tfidf_matrix.npz") if not p.exists()]
    if problems:
        sys.exit("Missing files that should come with the repository: " + ", ".join(problems)
                 + "\nRun `git pull` and make sure the clone is complete.")

    emb_path, faiss_path = INDEX_DIR / "embeddings.npy", INDEX_DIR / "kcc_index.faiss"
    if emb_path.exists() and faiss_path.exists():
        print("Retrieval index already present; nothing to do.")
        return

    import faiss
    from sentence_transformers import SentenceTransformer

    meta = pd.read_csv(INDEX_DIR / "corpus_metadata.csv")
    print(f"Encoding {len(meta):,} logged questions with {EMBED} (one-off, ~10-20 min) ...")
    emb = SentenceTransformer(EMBED).encode(meta["questions"].astype(str).tolist(), batch_size=128,
                                            show_progress_bar=True, normalize_embeddings=True).astype("float32")
    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb)
    np.save(emb_path, emb)
    faiss.write_index(index, str(faiss_path))
    print("Done. Start the app with run_app.bat (Windows) or:\n"
          "  python -m uvicorn app:app --app-dir backend --port 8420")


if __name__ == "__main__":
    main()
