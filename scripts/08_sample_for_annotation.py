"""Draw a diverse sample of corpus queries to be annotated by meaning.

The weak keyword labels are only used here to *stratify* the sample so rare
categories are represented; they are hidden from the annotator so they cannot
anchor the labels. Output: data/annotation/chunks/chunk_XX.txt, 250 lines each
("id | text"), plus data/annotation/pool.csv mapping id -> qid/split.

Two pools:
  train_pool  3,000 queries from train.csv  (used to train the new classifier)
  test_pool     500 queries from test.csv   (held-out, in-distribution silver test)
Lexical-diversity filter: at most one query per 6-word prefix, so the sample is
not 20 variations of "asking about control of aphids in mustard".
"""
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "annotation"
CHUNKS = OUT / "chunks"
CHUNKS.mkdir(parents=True, exist_ok=True)

PREFIX = re.compile(r"^(asking|asked|querry|query|enquiry|enquired|inquired|requested|farmer asked)\s+(about|regarding|for|on|that)?\s*", re.I)
RNG = 7

TRAIN_QUOTA = {
    "Plant Protection": 550, "Other": 600, "Nutrient Management": 350, "Varietal Information": 250,
    "Government Scheme": 250, "Animal Husbandry": 250, "Fisheries": 200, "Cultural Practices": 250,
    "Market Price": 150, "Weather": 150,
}


def key(text: str) -> str:
    t = PREFIX.sub("", text.lower().strip())
    t = re.sub(r"[^a-z0-9 ]+", " ", t)
    return " ".join(t.split()[:6])


def pick(df, quota, seen):
    out = []
    for label, n in quota.items():
        pool = df[df["query_type"] == label].sample(frac=1.0, random_state=RNG)
        got = 0
        for _, r in pool.iterrows():
            q = str(r["questions"])
            k = key(q)
            if len(q) < 12 or len(q) > 220 or k in seen:
                continue
            seen.add(k)
            out.append(r)
            got += 1
            if got >= n:
                break
        print(f"  {label:22s} wanted {n:4d} got {got:4d}")
    return pd.DataFrame(out)


def main():
    train = pd.read_csv(ROOT / "data/processed/train.csv").dropna(subset=["questions", "query_type"])
    test = pd.read_csv(ROOT / "data/processed/test.csv").dropna(subset=["questions", "query_type"])
    seen = set()
    print("train pool:")
    tr = pick(train, TRAIN_QUOTA, seen)
    tr["pool"] = "train"
    print("test pool:")
    te = pick(test, {k: max(25, v // 6) for k, v in TRAIN_QUOTA.items()}, seen)
    te["pool"] = "test"

    pool = pd.concat([tr, te], ignore_index=True).sample(frac=1.0, random_state=RNG).reset_index(drop=True)
    pool["id"] = [f"a{i:04d}" for i in range(len(pool))]
    pool[["id", "qid", "pool", "questions", "answers", "query_type"]].to_csv(OUT / "pool.csv", index=False)

    for c in range(0, len(pool), 250):
        part = pool.iloc[c:c + 250]
        lines = [f"{r.id} | {str(r.questions).strip()}" for r in part.itertuples()]
        (CHUNKS / f"chunk_{c // 250:02d}.txt").write_text("\n".join(lines), encoding="utf-8")
    print(f"pool={len(pool)} (train {len(tr)}, test {len(te)}); chunks={-(-len(pool) // 250)}")


if __name__ == "__main__":
    main()
