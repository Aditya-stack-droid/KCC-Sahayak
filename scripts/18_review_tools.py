"""Human validation of the LLM-assisted labels.

  make   write data/review/review_sheet.csv : 250 shuffled queries (all 150 gold +
         100 random silver-test queries) with an EMPTY `human_label` column and the
         model's labels withheld, so a human annotator is not anchored.
  score  read one or more completed sheets and report
           * agreement between each human and the LLM labels (accuracy, Cohen's kappa)
           * agreement between the humans (if 2+ sheets)
           * the disagreements, for adjudication
         and write results/human_agreement.json.

Labelling codes (see docs/annotation_guidelines.md):
  PP Plant Protection   NM Nutrient Management   WE Weather   GS Government Scheme
  MP Market Price       VI Varietal Information  AH Animal Husbandry
  FI Fisheries          CP Cultural Practices    OT Other

Usage:
    python scripts/18_review_tools.py make
    python scripts/18_review_tools.py score data/review/review_alice.csv [data/review/review_bob.csv]
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, cohen_kappa_score

ROOT = Path(__file__).resolve().parents[1]
REV = ROOT / "data" / "review"
CODES = {"PP": "Plant Protection", "NM": "Nutrient Management", "WE": "Weather", "GS": "Government Scheme",
         "MP": "Market Price", "VI": "Varietal Information", "AH": "Animal Husbandry", "FI": "Fisheries",
         "CP": "Cultural Practices", "OT": "Other"}


def make():
    REV.mkdir(parents=True, exist_ok=True)
    gold = pd.read_csv(ROOT / "data/gold/gold_queries.csv")[["id", "text", "label"]].assign(set="gold")
    ann = pd.read_csv(ROOT / "data/annotation/annotated.csv")
    sil = ann[ann["pool"] == "test"].sample(100, random_state=5)[["id", "text", "label"]].assign(set="silver")
    allq = pd.concat([gold, sil], ignore_index=True).sample(frac=1.0, random_state=13).reset_index(drop=True)
    allq[["id", "text", "label", "set"]].to_csv(REV / "answer_key_DO_NOT_SHARE.csv", index=False, encoding="utf-8")
    sheet = allq[["id", "text"]].assign(human_label="")
    sheet.to_csv(REV / "review_sheet.csv", index=False, encoding="utf-8-sig")  # utf-8-sig: opens correctly in Excel
    (REV / "README.txt").write_text(
        "Fill the human_label column with ONE code per row, using your own judgement of what the farmer is asking.\n"
        "Do not look at the answer key. Codes:\n" + "\n".join(f"  {k}  {v}" for k, v in CODES.items()) +
        "\n\nRules of thumb are in docs/annotation_guidelines.md. Save as CSV (keep UTF-8) and send it back.\n"
        "Make a separate copy of review_sheet.csv for each annotator.\n", encoding="utf-8")
    print(f"wrote {REV/'review_sheet.csv'} ({len(sheet)} rows) and the answer key")


def load_sheet(path):
    df = pd.read_csv(path, encoding="utf-8-sig")
    df["human_label"] = df["human_label"].astype(str).str.strip().str.upper().map(lambda x: CODES.get(x, x))
    bad = ~df["human_label"].isin(CODES.values())
    if bad.any():
        print(f"  ! {int(bad.sum())} rows in {Path(path).name} have missing/invalid codes and are skipped")
    return df[~bad]


def score(paths):
    key = pd.read_csv(REV / "answer_key_DO_NOT_SHARE.csv")
    out = {"annotators": {}}
    sheets = []
    for p in paths:
        s = load_sheet(p).merge(key[["id", "label", "set"]], on="id")
        sheets.append(s.set_index("id")["human_label"])
        res = {"n": int(len(s)), "accuracy_vs_llm": round(float(accuracy_score(s["label"], s["human_label"])), 4),
               "cohen_kappa_vs_llm": round(float(cohen_kappa_score(s["label"], s["human_label"])), 4)}
        for st in ["gold", "silver"]:
            m = s["set"] == st
            if m.any():
                res[f"kappa_{st}"] = round(float(cohen_kappa_score(s.loc[m, "label"], s.loc[m, "human_label"])), 4)
        out["annotators"][Path(p).name] = res
        dis = s[s["label"] != s["human_label"]][["id", "text", "label", "human_label"]]
        dis.to_csv(REV / f"disagreements_{Path(p).stem}.csv", index=False, encoding="utf-8-sig")
        print(Path(p).name, res, f"-> {len(dis)} disagreements saved")
    if len(sheets) >= 2:
        both = pd.concat(sheets[:2], axis=1, join="inner").dropna()
        out["human_vs_human"] = {"n": int(len(both)),
                                 "cohen_kappa": round(float(cohen_kappa_score(both.iloc[:, 0], both.iloc[:, 1])), 4),
                                 "accuracy": round(float((both.iloc[:, 0] == both.iloc[:, 1]).mean()), 4)}
        print("human vs human:", out["human_vs_human"])
    (ROOT / "results" / "human_agreement.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "make":
        make()
    elif len(sys.argv) >= 3 and sys.argv[1] == "score":
        score(sys.argv[2:])
    else:
        print(__doc__)
