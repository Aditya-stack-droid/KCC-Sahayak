"""Rigorous evaluation of KCC-Sahayak against baselines.

Part A  Intent classification
  - weak-label test subset (the same 3,000 queries scored in 04_evaluate.py)
  - the independent gold set (data/gold/gold_queries.csv): English / Hinglish / Hindi
  Models: majority class, keyword rules, TF-IDF + LinearSVC, fine-tuned DistilBERT.

Part B  Retrieval (3,000 test queries against the train+val corpus)
  Methods: TF-IDF cosine vs multilingual MiniLM + FAISS.
  Metrics: category match AND answer-level bag-of-words F1 against the gold
  answer, reported overall and on "novel" queries (no near-duplicate in the
  corpus), because ~70% of test queries have a near-duplicate in training.

Outputs go to results/: baselines.json, gold_eval.json, gold_predictions.csv.

Usage:
    .venv/Scripts/python.exe scripts/07_evaluate_gold_and_baselines.py
"""
import json
import re
import sys
from collections import Counter
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.svm import LinearSVC
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from weak_labels import label_query  # noqa: E402

DATA = ROOT / "data" / "processed"
GOLD = ROOT / "data" / "gold" / "gold_queries.csv"
RES = ROOT / "results"
INTENT_DIR = ROOT / "models" / "intent_classifier"
INDEX_DIR = ROOT / "models" / "retrieval_index"
EMBED_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
NOVEL_SIM = 0.80


def bert_predict(texts, tok, model, labels, bs=64):
    preds, confs = [], []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(texts), bs):
            enc = tok(texts[i:i + bs], truncation=True, padding=True, max_length=64, return_tensors="pt")
            p = torch.softmax(model(**enc).logits, dim=-1)
            c, idx = p.max(dim=-1)
            preds += [labels[j] for j in idx.tolist()]
            confs += c.tolist()
    return np.array(preds), np.array(confs)


def scores(y, p):
    return {
        "accuracy": round(float(accuracy_score(y, p)), 4),
        "macro_f1": round(float(f1_score(y, p, average="macro", zero_division=0)), 4),
    }


def token_f1(a, b):
    ta, tb = re.findall(r"\w+", str(a).lower()), re.findall(r"\w+", str(b).lower())
    if not ta or not tb:
        return 0.0
    common = Counter(ta) & Counter(tb)
    n = sum(common.values())
    if n == 0:
        return 0.0
    p, r = n / len(ta), n / len(tb)
    return 2 * p * r / (p + r)


def main():
    train = pd.read_csv(DATA / "train.csv").dropna(subset=["questions", "query_type"])
    val = pd.read_csv(DATA / "val.csv").dropna(subset=["questions", "answers"])
    corpus = pd.concat([train, val], ignore_index=True).dropna(subset=["questions", "answers"])
    preds_df = pd.read_csv(RES / "test_predictions.csv")  # DistilBERT preds on the 3,000 subset
    gold = pd.read_csv(GOLD)
    labels = json.loads((INTENT_DIR / "label_encoder.json").read_text())

    # ---------------- Part A: intent classification ----------------
    print("Fitting TF-IDF + LinearSVC on the full train split ...")
    tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    Xtr = tfidf.fit_transform(train["questions"])
    svc = LinearSVC(C=1.0).fit(Xtr, train["query_type"])
    majority = train["query_type"].value_counts().idxmax()

    tok = AutoTokenizer.from_pretrained(INTENT_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(INTENT_DIR)

    # A1: weak-label test subset (same 3,000 queries as 04_evaluate.py)
    sub = preds_df.copy()
    y = sub["query_type"].values
    novel = sub["retrieval_similarity"].values < NOVEL_SIM
    a1 = {}
    systems = {
        "majority_class": np.array([majority] * len(sub)),
        "keyword_rules": np.array([label_query(q) for q in sub["questions"]]),
        "tfidf_linearsvc": svc.predict(tfidf.transform(sub["questions"])),
        "finetuned_distilbert": sub["pred_query_type"].values,
    }
    for name, p in systems.items():
        a1[name] = {"all": scores(y, p), "novel_only": scores(y[novel], p[novel])}
    a1["n"] = int(len(sub))
    a1["n_novel"] = int(novel.sum())

    # A2: independent gold set
    gy = gold["label"].values
    gp_bert, gc_bert = bert_predict(gold["text"].tolist(), tok, model, labels)
    gsystems = {
        "majority_class": np.array([majority] * len(gold)),
        "keyword_rules": np.array([label_query(q) for q in gold["text"]]),
        "tfidf_linearsvc": svc.predict(tfidf.transform(gold["text"])),
        "finetuned_distilbert": gp_bert,
    }
    a2 = {"n": int(len(gold))}
    for name, p in gsystems.items():
        entry = {"all": scores(gy, p)}
        for st in ["en", "hinglish", "hi"]:
            m = (gold["style"] == st).values
            entry[st] = scores(gy[m], p[m])
        a2[name] = entry

    gold_out = gold.copy()
    gold_out["pred_distilbert"] = gp_bert
    gold_out["conf_distilbert"] = np.round(gc_bert, 4)
    gold_out["pred_tfidf"] = gsystems["tfidf_linearsvc"]
    gold_out["pred_rules"] = gsystems["keyword_rules"]

    # per-class accuracy for the fine-tuned model on gold
    per_class = (
        gold_out.assign(ok=gold_out["label"] == gold_out["pred_distilbert"])
        .groupby("label")["ok"].mean().round(3).to_dict()
    )
    a2["finetuned_distilbert_per_class_acc"] = per_class

    # ---------------- Part B: retrieval ----------------
    print("Retrieval: encoding 3,000 test queries with MiniLM ...")
    embed = SentenceTransformer(EMBED_NAME)
    index = faiss.read_index(str(INDEX_DIR / "kcc_index.faiss"))
    meta = pd.read_csv(INDEX_DIR / "corpus_metadata.csv")

    test = pd.read_csv(DATA / "test.csv").dropna(subset=["questions", "answers", "query_type"])
    test = test[test["qid"].isin(sub["qid"])].drop_duplicates("qid").reset_index(drop=True)
    q_emb = embed.encode(test["questions"].tolist(), batch_size=128, normalize_embeddings=True,
                         show_progress_bar=False).astype("float32")
    sims, idxs = index.search(q_emb, 1)
    mini_sim, mini_idx = sims[:, 0], idxs[:, 0]

    print("Retrieval: TF-IDF nearest neighbour ...")
    rv = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)
    C = rv.fit_transform(meta["questions"])
    Q = rv.transform(test["questions"])
    tf_idx = np.zeros(len(test), dtype=int)
    for s in range(0, Q.shape[0], 500):
        block = (Q[s:s + 500] @ C.T).toarray()
        tf_idx[s:s + 500] = block.argmax(axis=1)

    novel_t = mini_sim < NOVEL_SIM
    b = {"n": int(len(test)), "n_novel": int(novel_t.sum())}
    for name, ix in {"tfidf": tf_idx, "minilm_faiss": mini_idx}.items():
        ret = meta.iloc[ix]
        f1s = np.array([token_f1(r, g) for r, g in zip(ret["answers"], test["answers"])])
        match = (ret["query_type"].values == test["query_type"].values)
        b[name] = {
            "category_match_all": round(float(match.mean()), 4),
            "category_match_novel": round(float(match[novel_t].mean()), 4),
            "answer_token_f1_all": round(float(f1s.mean()), 4),
            "answer_token_f1_novel": round(float(f1s[novel_t].mean()), 4),
        }

    # gold-set retrieval (cross-lingual): category match against gold label
    g_emb = embed.encode(gold["text"].tolist(), batch_size=64, normalize_embeddings=True,
                         show_progress_bar=False).astype("float32")
    gs, gi = index.search(g_emb, 1)
    Gq = rv.transform(gold["text"])
    g_tf = np.asarray((Gq @ C.T).argmax(axis=1)).ravel()
    gold_out["minilm_top1_q"] = meta.iloc[gi[:, 0]]["questions"].values
    gold_out["minilm_top1_sim"] = np.round(gs[:, 0], 4)
    gold_out["minilm_top1_type"] = meta.iloc[gi[:, 0]]["query_type"].values
    gold_out["tfidf_top1_type"] = meta.iloc[g_tf]["query_type"].values
    gret = {}
    for st in ["en", "hinglish", "hi"]:
        m = (gold_out["style"] == st).values
        gret[st] = {
            "minilm_category_match": round(float((gold_out["minilm_top1_type"][m] == gold_out["label"][m]).mean()), 4),
            "tfidf_category_match": round(float((gold_out["tfidf_top1_type"][m] == gold_out["label"][m]).mean()), 4),
        }
    b["gold_retrieval_by_style"] = gret

    # ---------------- write ----------------
    RES.mkdir(exist_ok=True)
    (RES / "baselines.json").write_text(json.dumps({"intent_weak_label_test": a1, "retrieval": b}, indent=2))
    (RES / "gold_eval.json").write_text(json.dumps(a2, indent=2))
    gold_out.to_csv(RES / "gold_predictions.csv", index=False, encoding="utf-8")

    print("\n=== A1: weak-label test subset (n=%d, novel n=%d) ===" % (a1["n"], a1["n_novel"]))
    for k in systems:
        print(f"{k:22s} all={a1[k]['all']}  novel={a1[k]['novel_only']}")
    print("\n=== A2: independent gold set (n=%d) ===" % a2["n"])
    for k in gsystems:
        e = a2[k]
        print(f"{k:22s} all={e['all']['accuracy']:.3f}  en={e['en']['accuracy']:.3f}  "
              f"hinglish={e['hinglish']['accuracy']:.3f}  hi={e['hi']['accuracy']:.3f}")
    print("per-class (distilbert):", per_class)
    print("\n=== B: retrieval ===")
    print(json.dumps(b, indent=2))


if __name__ == "__main__":
    main()
