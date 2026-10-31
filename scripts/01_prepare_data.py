"""Clean the raw KCC question/answer corpus, apply weak QueryType labels and
glossary-based entity extraction, and write train/val/test splits.

Usage:
    .venv/Scripts/python.exe scripts/01_prepare_data.py
"""
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from weak_labels import label_query  # noqa: E402
from ner import extract_entities  # noqa: E402

RAW_PATH = ROOT / "data" / "raw" / "kcc_mapping_full.csv"
OUT_DIR = ROOT / "data" / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MIN_CHARS = 8


def clean_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = s.strip().lower()
    s = " ".join(s.split())
    return s


def main():
    print(f"Loading {RAW_PATH} ...")
    df = pd.read_csv(RAW_PATH)
    print(f"Raw rows: {len(df)}")

    df["questions"] = df["questions"].map(clean_text)
    df["answers"] = df["answers"].map(clean_text)

    df = df[(df["questions"].str.len() >= MIN_CHARS) & (df["answers"].str.len() >= MIN_CHARS)]
    df = df.drop_duplicates(subset=["questions", "answers"])
    df = df.dropna(subset=["questions", "answers"])
    print(f"After cleaning/dedup: {len(df)}")

    print("Applying weak QueryType labels ...")
    df["query_type"] = df["questions"].map(label_query)

    print("Extracting crop/symptom entities ...")
    entities = df["questions"].map(extract_entities)
    df["crops"] = entities.map(lambda e: ";".join(e["crop"]))
    df["symptoms"] = entities.map(lambda e: ";".join(e["symptom"]))

    print("\nQueryType distribution:")
    print(df["query_type"].value_counts())

    df = df.reset_index(drop=True)
    df["qid"] = df.index

    train_df, temp_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["query_type"]
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, random_state=42, stratify=temp_df["query_type"]
    )

    train_df.to_csv(OUT_DIR / "train.csv", index=False)
    val_df.to_csv(OUT_DIR / "val.csv", index=False)
    test_df.to_csv(OUT_DIR / "test.csv", index=False)
    df.to_csv(OUT_DIR / "full_labeled.csv", index=False)

    print(f"\nSaved: train={len(train_df)} val={len(val_df)} test={len(test_df)}")
    print(f"Output dir: {OUT_DIR}")


if __name__ == "__main__":
    main()
