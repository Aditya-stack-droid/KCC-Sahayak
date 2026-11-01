"""Intent head used in production: logistic regression over
[multilingual sentence embedding || character-n-gram TF-IDF], with temperature-
calibrated probabilities.

* The embedding carries cross-lingual meaning (Hindi <-> English).
* Character n-grams (char_wb 2-5) carry in-language lexical evidence and are
  robust to the spelling variation typical of Roman-script Hindi
  ("baarish" / "barish" / "barsat" share most n-grams).
"""
from pathlib import Path

import joblib
import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


class IntentHead:
    def __init__(self, lr, classes, temperature=1.0, tfidf=None, emb_weight=1.0, embed_model=None):
        self.lr = lr
        self.classes = list(classes)
        self.temperature = float(temperature)
        self.tfidf = tfidf
        self.emb_weight = float(emb_weight)
        self.embed_model = embed_model

    # ---- features
    def features(self, emb: np.ndarray, texts):
        emb = np.atleast_2d(emb)
        dense = sp.csr_matrix(emb * self.emb_weight)
        if self.tfidf is None:
            return dense
        return sp.hstack([dense, self.tfidf.transform(list(texts))], format="csr")

    def decision(self, emb, texts):
        return self.lr.decision_function(self.features(emb, texts))

    def proba(self, emb, texts):
        z = self.decision(emb, texts) / self.temperature
        z = z - z.max(1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(1, keepdims=True)

    # ---- persistence
    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"lr": self.lr, "classes": self.classes, "temperature": self.temperature,
                     "tfidf": self.tfidf, "emb_weight": self.emb_weight,
                     "embed_model": self.embed_model}, path)

    @classmethod
    def load(cls, path: Path):
        b = joblib.load(path)
        return cls(b["lr"], b["classes"], b.get("temperature", 1.0), b.get("tfidf"),
                   b.get("emb_weight", 1.0), b.get("embed_model"))


def make_tfidf():
    return TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True,
                           dtype=np.float32)


def fit_head(emb, texts, y, C=30.0, use_tfidf=True, emb_weight=1.0):
    tf = make_tfidf().fit(list(texts)) if use_tfidf else None
    head = IntentHead(None, [], 1.0, tf, emb_weight)
    X = head.features(emb, texts)
    lr = LogisticRegression(C=C, max_iter=3000).fit(X, y)
    head.lr, head.classes = lr, list(lr.classes_)
    return head
