"""Build a FAISS approximate-nearest-neighbour index over multilingual
sentence embeddings of the historical KCC questions, so a new (possibly
Hindi/Haryanvi) farmer query can retrieve the closest previously-answered
query and reuse its verified answer.

Usage:
    .venv/Scripts/python.exe scripts/03_build_retrieval_index.py
"""
import sys
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"
INDEX_DIR = ROOT / "models" / "retrieval_index"
INDEX_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"


def main():
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    val_df = pd.read_csv(DATA_DIR / "val.csv")
    corpus_df = pd.concat([train_df, val_df], ignore_index=True)
    corpus_df = corpus_df.dropna(subset=["questions", "answers"]).reset_index(drop=True)

    print(f"Retrieval corpus size: {len(corpus_df)}")
    print(f"Loading embedding model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    print("Encoding corpus questions ...")
    embeddings = model.encode(
        corpus_df["questions"].tolist(),
        batch_size=128,
        show_progress_bar=True,
        normalize_embeddings=True,
    ).astype("float32")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # cosine sim via inner product on normalized vecs
    index.add(embeddings)

    faiss.write_index(index, str(INDEX_DIR / "kcc_index.faiss"))
    corpus_df[["qid", "questions", "answers", "query_type", "crops", "symptoms"]].to_csv(
        INDEX_DIR / "corpus_metadata.csv", index=False
    )
    np.save(INDEX_DIR / "embeddings.npy", embeddings)

    print(f"Saved FAISS index + metadata to {INDEX_DIR}")


if __name__ == "__main__":
    main()
