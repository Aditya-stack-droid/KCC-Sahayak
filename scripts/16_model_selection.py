"""Choose the production intent model and calibrate it.

Candidates (all trained on annotated-train + spoken-style augmentation):
  emb_lr        MiniLM sentence embedding + logistic regression
  word_svc      word TF-IDF + LinearSVC
  char_svc      char-n-gram TF-IDF + LinearSVC
  combo_lr      [embedding || char-n-gram TF-IDF] + logistic regression (grid over C, weight)

Scored on: silver test, gold (en / hinglish / hi), and a SPELLING-PERTURBED copy of
the Hinglish gold queries (stress test: the training paraphrases and the gold set
share one author's romanisation, which lexical models can memorise).

Selection score = mean(silver, gold_en, gold_hi, hinglish_perturbed).
--save writes the winner to models/intent_embed_lr/model.joblib with temperature
scaling fitted on out-of-fold logits, and results/calibration.json.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from intent_model import IntentHead, fit_head, make_tfidf  # noqa: E402

ANN = ROOT / "data" / "annotation"
RES = ROOT / "results"
EMBED = "paraphrase-multilingual-MiniLM-L12-v2"

SUBS = [("aa", "a"), ("ee", "i"), ("oo", "u"), ("sh", "s"), ("ch", "c"), ("w", "v"), ("kh", "k"),
        ("th", "t"), ("dh", "d")]


def perturb(text, rng, p=0.5):
    out = []
    for w in text.split():
        if len(w) > 3 and rng.random() < p:
            for a, b in SUBS:
                if rng.random() < 0.5:
                    w = w.replace(a, b)
        out.append(w)
    return " ".join(out)


def acc(y, p):
    return float(accuracy_score(y, p))


def ece(P, y_idx, bins=10):
    conf, pred = P.max(1), P.argmax(1)
    ok = (pred == y_idx).astype(float)
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs(ok[m].mean() - conf[m].mean())
    return float(e)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--save", action="store_true")
    args = ap.parse_args()

    ann = pd.read_csv(ANN / "annotated.csv")
    train = pd.concat([ann[ann["pool"] == "train"][["text", "label"]],
                       pd.read_csv(ANN / "augmented.csv")[["text", "label"]]], ignore_index=True)
    silver = ann[ann["pool"] == "test"].reset_index(drop=True)
    gold = pd.read_csv(ROOT / "data/gold/gold_queries.csv")
    hing = gold[gold["style"] == "hinglish"].reset_index(drop=True)
    rngs = [np.random.default_rng(s) for s in range(5)]
    pert = [[perturb(t, r) for t in hing["text"]] for r in rngs]
    print("example perturbation:", hing["text"][0], "->", pert[0][0])

    embed = SentenceTransformer(EMBED)
    enc = lambda xs: embed.encode(list(xs), batch_size=128, normalize_embeddings=True, show_progress_bar=False)
    E = {"train": enc(train["text"]), "silver": enc(silver["text"]), "gold": enc(gold["text"]),
         "pert": [enc(p) for p in pert]}
    y = train["label"].values

    def evaluate(predict):
        """predict(emb, texts) -> labels"""
        res = {"silver": acc(silver["label"], predict(E["silver"], silver["text"]))}
        pg = predict(E["gold"], gold["text"])
        res["gold"] = acc(gold["label"], pg)
        for st in ["en", "hinglish", "hi"]:
            m = (gold["style"] == st).values
            res[f"gold_{st}"] = acc(gold["label"][m], pg[m])
        res["hinglish_perturbed"] = float(np.mean([acc(hing["label"], predict(E["pert"][i], pert[i]))
                                                   for i in range(len(pert))]))
        res["selection_score"] = float(np.mean([res["silver"], res["gold_en"], res["gold_hi"],
                                                res["hinglish_perturbed"]]))
        return {k: round(v, 4) for k, v in res.items()}

    results, heads = {}, {}

    h = fit_head(E["train"], train["text"], y, C=30, use_tfidf=False)
    results["emb_lr"] = evaluate(lambda e, t: np.array(h.classes)[h.decision(e, t).argmax(1)])
    heads["emb_lr"] = h

    wv = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)
    wsvc = LinearSVC().fit(wv.fit_transform(train["text"]), y)
    results["word_svc"] = evaluate(lambda e, t: wsvc.predict(wv.transform(list(t))))

    cv = make_tfidf()
    csvc = LinearSVC().fit(cv.fit_transform(train["text"]), y)
    results["char_svc"] = evaluate(lambda e, t: csvc.predict(cv.transform(list(t))))

    for C in [10, 30]:
        for w in [1.0, 2.0, 3.0]:
            name = f"combo_lr_C{C}_w{w:g}"
            hh = fit_head(E["train"], train["text"], y, C=C, use_tfidf=True, emb_weight=w)
            results[name] = evaluate(lambda e, t, hh=hh: np.array(hh.classes)[hh.decision(e, t).argmax(1)])
            heads[name] = hh

    print(f"\n{'model':22s} {'silver':>7s} {'gold':>6s} {'en':>6s} {'hing':>6s} {'hi':>6s} {'hing*':>7s} {'score':>7s}")
    for k, v in results.items():
        print(f"{k:22s} {v['silver']:7.3f} {v['gold']:6.3f} {v['gold_en']:6.3f} {v['gold_hinglish']:6.3f} "
              f"{v['gold_hi']:6.3f} {v['hinglish_perturbed']:7.3f} {v['selection_score']:7.3f}")
    print("(hing* = Hinglish gold with randomly perturbed spellings, mean of 5 seeds)")

    cands = {k: v for k, v in results.items() if k in heads}
    best = max(cands, key=lambda k: cands[k]["selection_score"])
    print("\nselected:", best)
    (RES / "model_selection.json").write_text(json.dumps({"results": results, "selected": best}, indent=2))

    if args.save:
        bh = heads[best]
        # --- temperature scaling from out-of-fold decision values
        skf = StratifiedKFold(5, shuffle=True, random_state=0)
        classes = bh.classes
        yi = np.array([classes.index(c) for c in y])
        oof = np.zeros((len(y), len(classes)))
        C = bh.lr.C
        for tr, te in skf.split(E["train"], y):
            fold = fit_head(E["train"][tr], train["text"].iloc[tr], y[tr], C=C,
                            use_tfidf=bh.tfidf is not None, emb_weight=bh.emb_weight)
            d = fold.decision(E["train"][te], train["text"].iloc[te])
            for j, c in enumerate(fold.classes):
                oof[te, classes.index(c)] = d[:, j]

        def probs(z, T):
            z = z / T
            z = z - z.max(1, keepdims=True)
            e = np.exp(z)
            return e / e.sum(1, keepdims=True)

        grid = np.linspace(0.3, 4.0, 75)
        nll = [-np.log(probs(oof, T)[np.arange(len(yi)), yi] + 1e-12).mean() for T in grid]
        T = float(grid[int(np.argmin(nll))])
        ysi = np.array([classes.index(c) for c in silver["label"]])
        dsi = bh.decision(E["silver"], silver["text"])
        cal = {"temperature": T,
               "ece_silver_uncalibrated": round(ece(probs(dsi, 1.0), ysi), 4),
               "ece_silver_calibrated": round(ece(probs(dsi, T), ysi), 4),
               "ece_oof_uncalibrated": round(ece(probs(oof, 1.0), yi), 4),
               "ece_oof_calibrated": round(ece(probs(oof, T), yi), 4),
               "model": best}
        print("calibration:", cal)
        (RES / "calibration.json").write_text(json.dumps(cal, indent=2))
        bh.temperature = T
        bh.embed_model = EMBED
        bh.save(ROOT / "models" / "intent_embed_lr" / "model.joblib")
        print("saved production intent head")


if __name__ == "__main__":
    main()
