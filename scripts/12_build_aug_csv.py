"""Assemble data/annotation/augmented.csv from the hand-written paraphrase files
(data/annotation/aug/out_*.txt, one line per source query:
 "id ### english ### hinglish ### hindi"), carrying over each source's label.

Guards against leakage: every source id must belong to the TRAIN pool.
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ANN = ROOT / "data" / "annotation"


def main():
    ann = pd.read_csv(ANN / "annotated.csv").set_index("id")
    rows = []
    for p in sorted((ANN / "aug").glob("out_*.txt")):
        for ln in p.read_text(encoding="utf-8").splitlines():
            if not ln.strip():
                continue
            parts = [x.strip() for x in ln.split("###")]
            assert len(parts) == 4, f"bad line in {p.name}: {ln[:60]}"
            sid, en, hing, hi = parts
            assert sid in ann.index, f"unknown id {sid}"
            assert ann.loc[sid, "pool"] == "train", f"{sid} is not from the train pool"
            lab = ann.loc[sid, "label"]
            for style, text in (("en", en), ("hinglish", hing), ("hi", hi)):
                assert text, f"empty {style} text for {sid}"
                rows.append({"text": text, "label": lab, "style": style, "source_id": sid})
    df = pd.DataFrame(rows)
    df.to_csv(ANN / "augmented.csv", index=False, encoding="utf-8")
    print(f"{len(df)} rows from {df['source_id'].nunique()} sources")
    print(df.groupby(["label", "style"]).size().unstack().to_string())


if __name__ == "__main__":
    main()
