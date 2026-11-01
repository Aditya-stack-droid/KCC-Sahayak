"""Merge the chunked LLM annotations into data/annotation/annotated.csv and
report (a) how often the old keyword labels disagree with meaning-based
labels, and (b) which regions the corpus actually covers.

Usage:
    .venv/Scripts/python.exe scripts/09_merge_annotations.py
"""
import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ANN = ROOT / "data" / "annotation"
CODES = {
    "PP": "Plant Protection", "NM": "Nutrient Management", "WE": "Weather",
    "GS": "Government Scheme", "MP": "Market Price", "VI": "Varietal Information",
    "AH": "Animal Husbandry", "FI": "Fisheries", "CP": "Cultural Practices", "OT": "Other",
}

ASSAM = r"assam|jorhat|kamrup|dibrugarh|sivasagar|sibsagar|sivsagar|sibasagar|dhemaji|lakhimpur|nagaon|golaghat|sonitpur|tinsukia|barpeta|nalbari|darrang|dhubri|goalpara|udalguri|baksa|cachar|bongaigaon|chirang|karbi|guwahati|marigaon|hojai|kokrajhar|sali rice|boro rice|ahu rice|katha"
HARYANA = r"haryana|hisar|rohtak|karnal|panipat|sonipat|sonepat|ambala|kurukshetra|sirsa|bhiwani|jind|kaithal|rewari|gurgaon|gurugram|faridabad|yamunanagar|fatehabad|mahendragarh|narnaul|panchkula|jhajjar|palwal|mewat|nuh"


def main():
    rows = []
    for p in sorted((ANN / "labels").glob("labels_*.txt")):
        for line in p.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            i, code = line.split()
            rows.append((i, CODES[code]))
    lab = pd.DataFrame(rows, columns=["id", "label"])
    pool = pd.read_csv(ANN / "pool.csv")
    df = pool.merge(lab, on="id", how="left")
    missing = int(df["label"].isna().sum())
    dup = int(lab["id"].duplicated().sum())
    assert missing == 0, f"{missing} unlabeled ids"
    assert dup == 0, f"{dup} duplicate ids"
    df = df.rename(columns={"questions": "text", "query_type": "weak_label"})
    df.to_csv(ANN / "annotated.csv", index=False, encoding="utf-8")

    print(f"annotated: {len(df)}  (train pool {int((df.pool == 'train').sum())}, test pool {int((df.pool == 'test').sum())})")
    print("\nlabel distribution:")
    print(df["label"].value_counts().to_string())

    agree = (df["label"] == df["weak_label"]).mean()
    print(f"\nkeyword-rule label == meaning-based label: {agree:.3f}")
    ct = pd.crosstab(df["weak_label"], df["label"])
    print("\n(rows = old keyword label, cols = meaning-based label)")
    print(ct.to_string())

    # geography of the whole corpus (query text only; the corpus has no state column)
    full = pd.read_csv(ROOT / "data/processed/full_labeled.csv", usecols=["questions", "answers"])
    q = (full["questions"].fillna("") + " " + full["answers"].fillna("")).str.lower()
    n = len(full)
    a = q.str.contains(ASSAM, regex=True).sum()
    h = q.str.contains(HARYANA, regex=True).sum()
    geo = {"queries": int(n), "mention_assam_places_or_terms": int(a), "mention_haryana_places": int(h)}
    print("\ngeography:", json.dumps(geo))
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "corpus_geography.json").write_text(json.dumps(geo, indent=2))
    ct.to_csv(ROOT / "results" / "weak_vs_semantic_confusion.csv")
    (ROOT / "results" / "annotation_summary.json").write_text(json.dumps({
        "n": int(len(df)), "keyword_vs_semantic_agreement": round(float(agree), 4),
        "distribution": df["label"].value_counts().to_dict()}, indent=2))


if __name__ == "__main__":
    main()
