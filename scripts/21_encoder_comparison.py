"""Compare the encoder choices named in the project synopsis (MuRIL / IndicBERT) with the
deployed multilingual MiniLM, on the same data and the same evaluation sets.

All systems train on annotated-train + spoken-style augmentation (4,200 rows) and are scored on
the silver test (n=496) and the independent gold set (en / hinglish / hi, n=150).

Frozen encoders  : mean-pooled embedding -> [embedding || char-n-gram TF-IDF] -> logistic regression
                   (identical head to the deployed model), plus an embedding-only head.
Fine-tuned (--ft): the encoder is fine-tuned end to end with a linear classification layer
                   (what the synopsis proposed). CPU only, so a short schedule.

Outputs: results/encoder_comparison.json
Usage:   .venv/Scripts/python.exe scripts/21_encoder_comparison.py [--ft]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from intent_model import fit_head  # noqa: E402

ANN = ROOT / "data" / "annotation"
RES = ROOT / "results"

FROZEN = {
    "minilm (deployed)": ("st", "paraphrase-multilingual-MiniLM-L12-v2"),
    "labse": ("st", "sentence-transformers/LaBSE"),
    "muril": ("hf", "google/muril-base-cased"),
    "indicbertv2": ("hf", "ai4bharat/IndicBERTv2-MLM-only"),
}
FINETUNE = {"muril (fine-tuned)": "google/muril-base-cased",
            "indicbertv2 (fine-tuned)": "ai4bharat/IndicBERTv2-MLM-only"}


def hf_embed(name):
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModel.from_pretrained(name).eval()

    def enc(texts):
        out = []
        texts = list(texts)
        with torch.no_grad():
            for i in range(0, len(texts), 64):
                b = tok(texts[i:i + 64], truncation=True, padding=True, max_length=64, return_tensors="pt")
                h = model(**b).last_hidden_state
                m = b["attention_mask"].unsqueeze(-1).float()
                v = (h * m).sum(1) / m.sum(1)
                out.append(torch.nn.functional.normalize(v, dim=-1).numpy())
        return np.vstack(out)
    return enc


def st_embed(name):
    m = SentenceTransformer(name)
    return lambda xs: m.encode(list(xs), batch_size=64, normalize_embeddings=True, show_progress_bar=False)


def finetune(name, train, evals, epochs=3, bs=16, lr=3e-5, seed=0):
    torch.manual_seed(seed)
    classes = sorted(train["label"].unique())
    idx = {c: i for i, c in enumerate(classes)}
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForSequenceClassification.from_pretrained(name, num_labels=len(classes))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    texts, y = list(train["text"]), np.array([idx[c] for c in train["label"]])
    steps = epochs * int(np.ceil(len(texts) / bs))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / (0.1 * steps)) * max(0.0, (steps - s) / steps))
    rng = np.random.default_rng(seed)
    t0, step = time.time(), 0
    model.train()
    for ep in range(epochs):
        order = rng.permutation(len(texts))
        for i in range(0, len(order), bs):
            j = order[i:i + bs]
            b = tok([texts[k] for k in j], truncation=True, padding=True, max_length=48, return_tensors="pt")
            loss = model(**b, labels=torch.tensor(y[j])).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad()
            step += 1
            if step % 50 == 0:
                print(f"    {name} step {step}/{steps} loss {loss.item():.3f} ({time.time() - t0:.0f}s)", flush=True)
    model.eval()
    preds = {}
    with torch.no_grad():
        for key, xs in evals.items():
            out = []
            xs = list(xs)
            for i in range(0, len(xs), 64):
                b = tok(xs[i:i + 64], truncation=True, padding=True, max_length=48, return_tensors="pt")
                out += model(**b).logits.argmax(-1).tolist()
            preds[key] = np.array([classes[k] for k in out])
    return preds


def boot_ci(correct, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    c = np.asarray(correct, float)
    m = [c[rng.integers(0, len(c), len(c))].mean() for _ in range(n)]
    return [round(float(np.percentile(m, 2.5)), 4), round(float(np.percentile(m, 97.5)), 4)]


def score(silver, gold, ps, pg):
    ok_s, ok_g = (ps == silver["label"].values), (pg == gold["label"].values)
    r = {"silver": round(float(ok_s.mean()), 4), "silver_ci": boot_ci(ok_s),
         "gold": round(float(ok_g.mean()), 4), "gold_ci": boot_ci(ok_g)}
    for st in ["en", "hinglish", "hi"]:
        m = (gold["style"] == st).values
        r[f"gold_{st}"] = round(float(ok_g[m].mean()), 4)
    return r, ok_s, ok_g


def mcnemar_exact(a, b):
    from scipy.stats import binomtest
    n01, n10 = int(((~a) & b).sum()), int((a & (~b)).sum())
    n = n01 + n10
    return {"only_other_right": n01, "only_deployed_right": n10,
            "p": round(float(binomtest(min(n01, n10), n, 0.5).pvalue), 4) if n else 1.0}


def save(results, correct, train, silver, gold, ft):
    ref = "minilm (deployed) | emb + char-tfidf LR"
    for k, (os_, og) in correct.items():
        if k != ref and ref in correct:
            results[k]["mcnemar_vs_deployed_gold"] = mcnemar_exact(correct[ref][1], og)
            results[k]["mcnemar_vs_deployed_silver"] = mcnemar_exact(correct[ref][0], os_)
    out = {"n_train": int(len(train)), "n_silver": int(len(silver)), "n_gold": int(len(gold)),
           "reference": ref, "results": results, "fine_tuned_included": bool(ft)}
    RES.mkdir(exist_ok=True)
    (RES / "encoder_comparison.json").write_text(json.dumps(out, indent=2), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ft", action="store_true", help="also fine-tune MuRIL and IndicBERTv2 end to end (slow on CPU)")
    args = ap.parse_args()

    ann = pd.read_csv(ANN / "annotated.csv")
    train = pd.concat([ann[ann["pool"] == "train"][["text", "label"]],
                       pd.read_csv(ANN / "augmented.csv")[["text", "label"]]], ignore_index=True)
    silver = ann[ann["pool"] == "test"].reset_index(drop=True)
    gold = pd.read_csv(ROOT / "data/gold/gold_queries.csv")
    y = train["label"].values
    results, correct = {}, {}

    for name, (kind, ident) in FROZEN.items():
        print(f"[frozen] {name}", flush=True)
        enc = st_embed(ident) if kind == "st" else hf_embed(ident)
        Xt, Xs, Xg = enc(train["text"]), enc(silver["text"]), enc(gold["text"])
        for head_name, use_tf, C in [("emb-only LR", False, 30), ("emb + char-tfidf LR", True, 10)]:
            h = fit_head(Xt, train["text"], y, C=C, use_tfidf=use_tf)
            ps = np.array(h.classes)[h.decision(Xs, silver["text"]).argmax(1)]
            pg = np.array(h.classes)[h.decision(Xg, gold["text"]).argmax(1)]
            r, ok_s, ok_g = score(silver, gold, ps, pg)
            key = f"{name} | {head_name}"
            results[key] = r
            correct[key] = (ok_s, ok_g)
            save(results, correct, train, silver, gold, args.ft)
            print(f"   {head_name:22s} silver {r['silver']:.3f}  gold {r['gold']:.3f}  "
                  f"en {r['gold_en']:.2f} hing {r['gold_hinglish']:.2f} hi {r['gold_hi']:.2f}", flush=True)

    if args.ft:
        for name, ident in FINETUNE.items():
            print(f"[fine-tune] {name}", flush=True)
            p = finetune(ident, train, {"silver": silver["text"], "gold": gold["text"]})
            r, ok_s, ok_g = score(silver, gold, p["silver"], p["gold"])
            key = f"{name} | classification head"
            results[key] = r
            correct[key] = (ok_s, ok_g)
            save(results, correct, train, silver, gold, args.ft)
            print(f"   silver {r['silver']:.3f}  gold {r['gold']:.3f}  en {r['gold_en']:.2f} "
                  f"hing {r['gold_hinglish']:.2f} hi {r['gold_hi']:.2f}", flush=True)

    save(results, correct, train, silver, gold, args.ft)
    print("saved results/encoder_comparison.json")


if __name__ == "__main__":
    main()
