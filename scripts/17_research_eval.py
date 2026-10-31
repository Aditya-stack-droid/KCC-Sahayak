"""Research-grade evaluation of the intent classifier.

1. Main comparison with 95% bootstrap CIs and McNemar exact tests vs the production model.
2. Ablations: embedding / char-n-gram / both, with and without spoken-style augmentation.
3. Learning curve: accuracy vs number of annotated training queries (3 seeds).
4. Novelty: accuracy on test queries with / without a near-duplicate in training.
5. "Agreement is not accuracy": the old keyword labels vs the new model, on the old test set.

All randomness is seeded. Embeddings are cached in results/cache/.
Outputs: results/research_eval.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import binomtest
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from intent_model import fit_head, make_tfidf  # noqa: E402
from answer_quality import informative_mask  # noqa: E402
from weak_labels import label_query  # noqa: E402

sys.path.insert(0, str(ROOT / "scripts"))
from importlib import import_module  # noqa: E402

ANN = ROOT / "data" / "annotation"
RES = ROOT / "results"
CACHE = RES / "cache"
CACHE.mkdir(parents=True, exist_ok=True)
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


_embedder = None


def embed(texts, key):
    """Cached sentence embeddings (key names the cache file)."""
    global _embedder
    f = CACHE / f"{key}.npy"
    if f.exists():
        e = np.load(f)
        if len(e) == len(texts):
            return e
    if _embedder is None:
        _embedder = SentenceTransformer(EMBED)
    e = _embedder.encode(list(texts), batch_size=128, normalize_embeddings=True,
                         show_progress_bar=False).astype("float32")
    np.save(f, e)
    return e


def boot_ci(correct, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    c = np.asarray(correct, dtype=float)
    idx = rng.integers(0, len(c), (n_boot, len(c)))
    means = c[idx].mean(1)
    return [round(float(np.percentile(means, 2.5)), 4), round(float(np.percentile(means, 97.5)), 4)]


def mcnemar(a_correct, b_correct):
    a, b = np.asarray(a_correct, bool), np.asarray(b_correct, bool)
    n01 = int((a & ~b).sum())  # a right, b wrong
    n10 = int((~a & b).sum())
    n = n01 + n10
    p = 1.0 if n == 0 else float(binomtest(min(n01, n10), n, 0.5).pvalue)
    return {"a_only_right": n01, "b_only_right": n10, "p_value": round(p, 4)}


def legacy_distilbert(texts):
    d = ROOT / "models" / "intent_classifier"
    tok = AutoTokenizer.from_pretrained(d)
    m = AutoModelForSequenceClassification.from_pretrained(d).eval()
    labels = json.loads((d / "label_encoder.json").read_text())
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), 64):
            enc = tok(list(texts[i:i + 64]), truncation=True, padding=True, max_length=64, return_tensors="pt")
            out += [labels[j] for j in m(**enc).logits.argmax(-1).tolist()]
    return np.array(out)


def main():
    ann = pd.read_csv(ANN / "annotated.csv")
    tr0 = ann[ann["pool"] == "train"][["text", "label"]].reset_index(drop=True)
    aug = pd.read_csv(ANN / "augmented.csv")[["text", "label"]]
    silver = ann[ann["pool"] == "test"].reset_index(drop=True)
    gold = pd.read_csv(ROOT / "data/gold/gold_queries.csv")
    hing = gold[gold["style"] == "hinglish"].reset_index(drop=True)
    pert = [[perturb(t, np.random.default_rng(s)) for t in hing["text"]] for s in range(5)]

    E_tr0 = embed(tr0["text"], "train_ann")
    E_aug = embed(aug["text"], "train_aug")
    E_si = embed(silver["text"], "silver")
    E_go = embed(gold["text"], "gold")
    E_pe = [embed(p, f"pert{i}") for i, p in enumerate(pert)]

    tr_all = pd.concat([tr0, aug], ignore_index=True)
    E_all = np.vstack([E_tr0, E_aug])

    def train(use_tfidf, use_aug, C=10, w=1.0, emb=True):
        T = tr_all if use_aug else tr0
        E = E_all if use_aug else E_tr0
        if not emb:  # char-only: zero the embedding block
            E = np.zeros_like(E)
        return fit_head(E, T["text"], T["label"].values, C=C, use_tfidf=use_tfidf, emb_weight=w)

    def predict(head, E, texts, emb=True):
        E = E if emb else np.zeros_like(E)
        return np.array(head.classes)[head.decision(E, list(texts)).argmax(1)]

    # ------------------------------------------------ 1. main comparison + CIs + McNemar
    print("== 1. main comparison")
    prod = train(True, True)
    systems = {}
    systems["production (emb+char, aug)"] = lambda E, t, emb: predict(prod, E, t)
    wv = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True).fit(tr_all["text"])
    wsvc = LinearSVC().fit(wv.transform(tr_all["text"]), tr_all["label"])
    systems["word TF-IDF + SVM"] = lambda E, t, emb: wsvc.predict(wv.transform(list(t)))
    cv = make_tfidf().fit(tr_all["text"])
    csvc = LinearSVC().fit(cv.transform(tr_all["text"]), tr_all["label"])
    systems["char TF-IDF + SVM"] = lambda E, t, emb: csvc.predict(cv.transform(list(t)))
    emb_only = train(False, True)
    systems["embedding + LR"] = lambda E, t, emb: predict(emb_only, E, t)
    systems["keyword rules (original)"] = lambda E, t, emb: np.array([label_query(s) for s in t])
    old_si, old_go = legacy_distilbert(list(silver["text"])), legacy_distilbert(list(gold["text"]))
    systems["DistilBERT on keyword labels (original)"] = None  # handled below

    sets = {
        "silver": (E_si, silver["text"], silver["label"].values, np.ones(len(silver), bool)),
        "gold_all": (E_go, gold["text"], gold["label"].values, np.ones(len(gold), bool)),
        "gold_en": (E_go, gold["text"], gold["label"].values, (gold["style"] == "en").values),
        "gold_hinglish": (E_go, gold["text"], gold["label"].values, (gold["style"] == "hinglish").values),
        "gold_hi": (E_go, gold["text"], gold["label"].values, (gold["style"] == "hi").values),
    }
    main_tbl, correct = {}, {}
    for name, fn in systems.items():
        main_tbl[name] = {}
        for sname, (E, T, y, mask) in sets.items():
            if fn is None:
                pred = old_si if sname == "silver" else old_go
            else:
                pred = fn(E, list(T), True)
            c = (np.asarray(pred) == y)[mask]
            correct[(name, sname)] = c
            main_tbl[name][sname] = {"acc": round(float(c.mean()), 4), "ci95": boot_ci(c), "n": int(mask.sum())}
        # spelling-perturbed Hinglish
        if fn is None:
            main_tbl[name]["hinglish_perturbed"] = None
        else:
            accs = [float((fn(E_pe[i], pert[i], True) == hing["label"].values).mean()) for i in range(5)]
            main_tbl[name]["hinglish_perturbed"] = {"acc": round(float(np.mean(accs)), 4), "sd": round(float(np.std(accs)), 4)}
    base = "production (emb+char, aug)"
    sig = {}
    for name in systems:
        if name == base:
            continue
        sig[name] = {s: mcnemar(correct[(base, s)], correct[(name, s)]) for s in ["silver", "gold_all"]}

    # ------------------------------------------------ 2. ablations
    print("== 2. ablations")
    abl = {}
    for label, (tf, aug_on, emb_on) in {
        "embedding only, no augmentation": (False, False, True),
        "char n-grams only, no augmentation": (True, False, False),
        "embedding only, + augmentation": (False, True, True),
        "char n-grams only, + augmentation": (True, True, False),
        "embedding + char, no augmentation": (True, False, True),
        "embedding + char, + augmentation (production)": (True, True, True),
    }.items():
        h = train(tf, aug_on, emb=emb_on)
        row = {}
        for sname, (E, T, y, mask) in sets.items():
            pred = predict(h, E, list(T), emb_on)
            row[sname] = round(float((pred == y)[mask].mean()), 4)
        row["hinglish_perturbed"] = round(float(np.mean([
            (predict(h, E_pe[i], pert[i], emb_on) == hing["label"].values).mean() for i in range(5)])), 4)
        abl[label] = row

    # ------------------------------------------------ 3. learning curve
    print("== 3. learning curve")
    lc = {}
    for n in [250, 500, 1000, 2000, len(tr0)]:
        sil, goen = [], []
        for seed in range(3):
            idx = np.random.default_rng(seed).permutation(len(tr0))[:n]
            h = fit_head(E_tr0[idx], tr0["text"].iloc[idx], tr0["label"].values[idx], C=10, use_tfidf=True)
            sil.append(float((predict(h, E_si, silver["text"]) == silver["label"].values).mean()))
            m = (gold["style"] == "en").values
            goen.append(float((predict(h, E_go, gold["text"])[m] == gold["label"].values[m]).mean()))
        lc[str(n)] = {"silver_mean": round(float(np.mean(sil)), 4), "silver_sd": round(float(np.std(sil)), 4),
                      "gold_en_mean": round(float(np.mean(goen)), 4), "gold_en_sd": round(float(np.std(goen)), 4)}

    # ------------------------------------------------ 4. novelty
    print("== 4. novelty")
    sim_to_train = (E_si @ E_tr0.T).max(1)
    pred_si = predict(prod, E_si, silver["text"])
    ok = pred_si == silver["label"].values
    nov = {}
    for lo, hi, name in [(0, 0.8, "novel (<0.80)"), (0.8, 0.9, "similar (0.80-0.90)"), (0.9, 1.01, "near-duplicate (>=0.90)")]:
        m = (sim_to_train >= lo) & (sim_to_train < hi)
        nov[name] = {"n": int(m.sum()), "accuracy": round(float(ok[m].mean()), 4) if m.any() else None}

    # ------------------------------------------------ 5. agreement is not accuracy
    print("== 5. agreement vs accuracy")
    wp = pd.read_csv(RES / "test_predictions.csv")
    E_wp = embed(wp["questions"], "weak_test")
    prod_on_weak = predict(prod, E_wp, wp["questions"])
    agree = {
        "old_distilbert_agreement_with_keyword_labels": round(float((wp["pred_query_type"] == wp["query_type"]).mean()), 4),
        "production_agreement_with_keyword_labels": round(float((prod_on_weak == wp["query_type"].values).mean()), 4),
        "old_distilbert_accuracy_vs_meaning_labels_silver": round(float((old_si == silver["label"].values).mean()), 4),
        "production_accuracy_vs_meaning_labels_silver": round(float(ok.mean()), 4),
        "n_weak_test": int(len(wp)),
    }

    # ------------------------------------------------ 6. corpus audit
    print("== 6. corpus audit")
    meta = pd.read_csv(ROOT / "models/retrieval_index/corpus_metadata.csv")
    inf = informative_mask(meta["answers"])
    sim = wp["retrieval_similarity"].to_numpy()
    audit = {
        "corpus_rows": int(len(meta)),
        "answers_that_are_call_log_notes": int((~inf).sum()),
        "share_answers_that_are_call_log_notes": round(float((~inf).mean()), 4),
        "weak_test_near_duplicate_share_ge_0.95": round(float((sim >= 0.95).mean()), 4),
        "weak_test_near_duplicate_share_ge_0.90": round(float((sim >= 0.90).mean()), 4),
        "weak_test_novel_share_lt_0.80": round(float((sim < 0.80).mean()), 4),
        "retrieval_category_match_near_duplicate": round(float(wp.loc[sim >= 0.95, "retrieval_type_match"].mean()), 4),
        "retrieval_category_match_novel": round(float(wp.loc[sim < 0.80, "retrieval_type_match"].mean()), 4),
        "geography": json.loads((RES / "corpus_geography.json").read_text()),
    }

    out = {"corpus_audit": audit, "main": main_tbl, "mcnemar_vs_production": sig, "ablation": abl, "learning_curve": lc,
           "novelty_silver": nov, "agreement_vs_accuracy": agree}
    (RES / "research_eval.json").write_text(json.dumps(out, indent=2))

    print("\nMain table (accuracy [95% CI])")
    for name, r in main_tbl.items():
        print(f"{name:42s} " + "  ".join(f"{s}={r[s]['acc']:.3f}[{r[s]['ci95'][0]:.2f},{r[s]['ci95'][1]:.2f}]" for s in ["silver", "gold_all"]))
    print("\nAblation:"); [print(f"  {k:48s} {v}") for k, v in abl.items()]
    print("\nLearning curve:"); [print(f"  n={k:>5s} {v}") for k, v in lc.items()]
    print("\nNovelty:", nov)
    print("\nAgreement vs accuracy:", agree)


if __name__ == "__main__":
    main()
