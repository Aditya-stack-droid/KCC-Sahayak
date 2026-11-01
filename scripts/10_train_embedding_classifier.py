"""Train and compare intent classifiers on the meaning-based annotations.

Train:      annotated.csv rows with pool == "train" (+ optional translations)
Silver test annotated.csv rows with pool == "test"  (in-distribution, held out)
Gold test   data/gold/gold_queries.csv (independent phrasing; en / hinglish / hi)

Models compared (all trained on the SAME labelled data):
  tfidf_svc            TF-IDF + LinearSVC (lexical baseline)
  minilm_lr            multilingual MiniLM sentence embeddings + logistic regression
  old_distilbert       the previously shipped model (trained on keyword labels)
  keyword_rules        the original rule labeller

Usage:
    .venv/Scripts/python.exe scripts/10_train_embedding_classifier.py [--save]
"""
import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.svm import LinearSVC
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from weak_labels import label_query  # noqa: E402

ANN = ROOT / "data" / "annotation"
GOLD = ROOT / "data" / "gold" / "gold_queries.csv"
OUT = ROOT / "models" / "intent_embed_lr"
RES = ROOT / "results"
EMBED = "paraphrase-multilingual-MiniLM-L12-v2"


def sc(y, p):
    return {"accuracy": round(float(accuracy_score(y, p)), 4),
            "macro_f1": round(float(f1_score(y, p, average="macro", zero_division=0)), 4)}


def old_distilbert(texts):
    d = ROOT / "models" / "intent_classifier"
    tok = AutoTokenizer.from_pretrained(d)
    m = AutoModelForSequenceClassification.from_pretrained(d).eval()
    labels = json.loads((d / "label_encoder.json").read_text())
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), 64):
            enc = tok(texts[i:i + 64], truncation=True, padding=True, max_length=64, return_tensors="pt")
            out += [labels[j] for j in m(**enc).logits.argmax(-1).tolist()]
    return np.array(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--save", action="store_true")
    ap.add_argument("--extra", default=None, help="optional CSV (text,label) of extra training rows, e.g. translations")
    args = ap.parse_args()

    ann = pd.read_csv(ANN / "annotated.csv")
    train = ann[ann["pool"] == "train"][["text", "label"]]
    if args.extra:
        extra = pd.read_csv(args.extra)[["text", "label"]]
        train = pd.concat([train, extra], ignore_index=True)
        print(f"+ {len(extra)} extra rows from {args.extra}")
    silver = ann[ann["pool"] == "test"].reset_index(drop=True)
    gold = pd.read_csv(GOLD)
    print(f"train={len(train)} silver_test={len(silver)} gold={len(gold)}")

    embed = SentenceTransformer(EMBED)
    enc = lambda xs: embed.encode(list(xs), batch_size=128, normalize_embeddings=True, show_progress_bar=False)
    Xtr, Xsi, Xgo = enc(train["text"]), enc(silver["text"]), enc(gold["text"])
    ytr = train["label"].values

    # choose C by 5-fold CV on the training data
    skf = StratifiedKFold(5, shuffle=True, random_state=0)
    best = (None, -1)
    for C in [1, 3, 10, 30, 100]:
        pred = cross_val_predict(LogisticRegression(C=C, max_iter=2000), Xtr, ytr, cv=skf)
        f1 = f1_score(ytr, pred, average="macro")
        print(f"  C={C:<4} cv macro-F1={f1:.4f}")
        if f1 > best[1]:
            best = (C, f1)
    lr = LogisticRegression(C=best[0], max_iter=2000).fit(Xtr, ytr)

    tf = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)
    svc = LinearSVC().fit(tf.fit_transform(train["text"]), ytr)

    systems = {
        "keyword_rules": lambda t, X: np.array([label_query(s) for s in t]),
        "tfidf_svc": lambda t, X: svc.predict(tf.transform(t)),
        "old_distilbert": lambda t, X: old_distilbert(list(t)),
        "minilm_lr": lambda t, X: lr.predict(X),
    }
    results = {"chosen_C": best[0], "n_train": int(len(train))}
    gold_preds = gold.copy()
    for name, fn in systems.items():
        ps = fn(silver["text"], Xsi)
        pg = fn(gold["text"], Xgo)
        gold_preds[f"pred_{name}"] = pg
        entry = {"silver": sc(silver["label"], ps), "gold": sc(gold["label"], pg)}
        for st in ["en", "hinglish", "hi"]:
            m = (gold["style"] == st).values
            entry[f"gold_{st}"] = sc(gold["label"][m], pg[m])
        results[name] = entry

    print(f"\n{'system':16s} {'silver':>8s} {'gold':>7s} {'en':>6s} {'hinglish':>9s} {'hi':>6s}")
    for name in systems:
        e = results[name]
        print(f"{name:16s} {e['silver']['accuracy']:8.3f} {e['gold']['accuracy']:7.3f} "
              f"{e['gold_en']['accuracy']:6.3f} {e['gold_hinglish']['accuracy']:9.3f} {e['gold_hi']['accuracy']:6.3f}")

    per_class = (gold_preds.assign(ok=gold_preds["label"] == gold_preds["pred_minilm_lr"])
                 .groupby("label")["ok"].mean().round(3).to_dict())
    results["minilm_lr_gold_per_class"] = per_class
    print("\nminilm_lr gold per-class:", per_class)

    RES.mkdir(exist_ok=True)
    tag = "_aug" if args.extra else ""
    (RES / f"intent_v2_eval{tag}.json").write_text(json.dumps(results, indent=2))
    gold_preds.to_csv(RES / f"gold_predictions_v2{tag}.csv", index=False, encoding="utf-8")
    if args.save:
        # --- confidence calibration: temperature scaling fitted on out-of-fold logits
        oof = cross_val_predict(LogisticRegression(C=best[0], max_iter=2000), Xtr, ytr, cv=skf,
                                method="decision_function")
        classes = list(lr.classes_)
        yi = np.array([classes.index(c) for c in ytr])

        def nll(T):
            z = oof / T
            z = z - z.max(1, keepdims=True)
            lp = z - np.log(np.exp(z).sum(1, keepdims=True))
            return -lp[np.arange(len(yi)), yi].mean()

        grid = np.linspace(0.2, 3.0, 57)
        T = float(grid[int(np.argmin([nll(t) for t in grid]))])

        def softmax(z):
            z = z - z.max(1, keepdims=True)
            e = np.exp(z)
            return e / e.sum(1, keepdims=True)

        def ece(P, y, bins=10):
            conf, pred = P.max(1), P.argmax(1)
            ok = (pred == y).astype(float)
            edges = np.linspace(0, 1, bins + 1)
            e = 0.0
            for lo, hi in zip(edges[:-1], edges[1:]):
                m = (conf > lo) & (conf <= hi)
                if m.any():
                    e += m.mean() * abs(ok[m].mean() - conf[m].mean())
            return float(e)

        ysi = np.array([classes.index(c) for c in silver["label"]])
        dsi = lr.decision_function(Xsi)
        cal = {
            "temperature": T,
            "ece_silver_uncalibrated": round(ece(softmax(dsi), ysi), 4),
            "ece_silver_calibrated": round(ece(softmax(dsi / T), ysi), 4),
            "ece_oof_uncalibrated": round(ece(softmax(oof), yi), 4),
            "ece_oof_calibrated": round(ece(softmax(oof / T), yi), 4),
        }
        print("calibration:", cal)
        (RES / "calibration.json").write_text(json.dumps(cal, indent=2))
        OUT.mkdir(parents=True, exist_ok=True)
        joblib.dump({"lr": lr, "classes": classes, "embed_model": EMBED, "temperature": T},
                    OUT / "model.joblib")
        print("saved", OUT / "model.joblib")


if __name__ == "__main__":
    main()
