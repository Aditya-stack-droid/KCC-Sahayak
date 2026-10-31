"""Select source queries to be re-expressed in spoken style (English, Hinglish,
Devanagari Hindi) for training-data augmentation.

The corpus is terse call-centre log text ("asking about spacing of cauliflower").
A voice-first system receives how farmers actually talk. We take 40 annotated
TRAIN-pool queries per class (never test-pool, never gold) and write natural
spoken paraphrases of each, keeping the label.

Output: data/annotation/aug_sources.txt  ("id | label | text"), 100 per block.
"""
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ANN = ROOT / "data" / "annotation"
PER_CLASS = 40


def usable(t: str) -> bool:
    t = str(t)
    if not (22 <= len(t) <= 170):
        return False
    if re.search(r"rainfall\(mm\)|temp\(max", t):
        return False
    return len(t.split()) >= 4


def main():
    df = pd.read_csv(ANN / "annotated.csv")
    df = df[(df["pool"] == "train") & df["text"].map(usable)]
    picks = (
        df.groupby("label", group_keys=False)
        .apply(lambda g: g.sample(min(PER_CLASS, len(g)), random_state=11), include_groups=False)
    )
    # groupby(include_groups=False) drops the label column -> re-attach
    picks = df.loc[picks.index]
    picks = picks.sort_values("id").reset_index(drop=True)
    lines = [f"{r.id} | {r.label} | {r.text.strip()}" for r in picks.itertuples()]
    blocks = [lines[i:i + 100] for i in range(0, len(lines), 100)]
    (ANN / "aug").mkdir(exist_ok=True)
    for i, b in enumerate(blocks):
        (ANN / "aug" / f"src_{i:02d}.txt").write_text("\n".join(b), encoding="utf-8")
    print(f"{len(lines)} sources in {len(blocks)} blocks")
    print(picks["label"].value_counts().to_string())


if __name__ == "__main__":
    main()
