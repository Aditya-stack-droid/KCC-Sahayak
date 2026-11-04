"""Two standalone out-of-scope detectors.

A. ``KNNRejector`` — mean cosine similarity to the k nearest agricultural
   questions in Batch A's grouped index (train+val only).
B. ``EmbeddingLRDetector`` — logistic regression on the project's MiniLM
   embeddings, trained on agricultural corpus queries plus authored synthetic
   out-of-scope queries.

Both output an in-scope score in ascending order of "looks agricultural", plus a
three-way decision (in_scope / out_of_scope / ambiguous) and a reason. Neither
touches the production classifier, retriever or safety gate.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import faiss
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression

import oos_data
from glossary import CROPS, PESTS_SYMPTOMS, QUERY_TYPE_RULES
from ner import extract_entities

_TERMS = set(CROPS) | set(PESTS_SYMPTOMS)
for _rules in QUERY_TYPE_RULES.values():
    _TERMS |= set(_rules)
_TERMS = sorted((t for t in _TERMS if t), key=len, reverse=True)
_TERM_RE = re.compile(r"\b(?:" + "|".join(re.escape(t) for t in _TERMS) + r")\b")

DECISIONS = ("in_scope", "out_of_scope", "ambiguous")


def agricultural_evidence(text) -> list:
    """Crop/pest/agronomy terms present in a query (English lexicon only).

    Conservative: exact word matches from the project's own glossary. Devanagari
    and Roman-Hindi farm words are not covered, which is a documented limitation.
    """
    if not isinstance(text, str) or not text.strip():
        return []
    low = text.lower()
    hits = set(extract_entities(low)["crop"]) | set(extract_entities(low)["symptom"])
    hits |= set(_TERM_RE.findall(low))
    return sorted(hits)[:8]


@dataclass
class Decision:
    score: float
    threshold: float
    decision: str
    accepted: bool
    reason: str
    evidence: list


def decide(score: float, threshold: float, margin: float, text: str) -> Decision:
    """Map a score to a decision; near-threshold and evidence cases are flagged."""
    score = float(score)
    accepted = score >= threshold
    if score >= threshold + margin:
        return Decision(score, threshold, "in_scope", True,
                        f"score {score:.3f} >= threshold {threshold:.3f} + margin {margin:.2f}",
                        [])
    if score > threshold - margin:
        return Decision(score, threshold, "ambiguous", accepted,
                        f"score {score:.3f} inside review band around threshold "
                        f"{threshold:.3f}", [])
    evidence = agricultural_evidence(text)
    if evidence:
        return Decision(score, threshold, "ambiguous", accepted,
                        f"low similarity but agricultural terms present: "
                        f"{', '.join(evidence)}", evidence)
    return Decision(score, threshold, "out_of_scope", False,
                    f"score {score:.3f} <= threshold {threshold:.3f} - margin "
                    f"{margin:.2f} and no agricultural terms", [])


class KNNRejector:
    """Mean cosine to the k nearest agricultural questions."""

    def __init__(self, index=None, index_path=None, k: int = 5):
        if index is None:
            path = index_path or (oos_data.GROUPED_ART / "grouped_retrieval_index.faiss")
            index = faiss.read_index(str(path))
        self.index = index
        self.k = int(k)

    def scores(self, embeddings, batch: int = 2048) -> np.ndarray:
        embeddings = np.ascontiguousarray(embeddings, dtype="float32")
        out = np.empty(len(embeddings), dtype=np.float32)
        for start in range(0, len(embeddings), batch):
            stop = min(start + batch, len(embeddings))
            sims, _ = self.index.search(embeddings[start:stop], self.k)
            out[start:stop] = sims.mean(axis=1)
        return out


class EmbeddingLRDetector:
    """Logistic regression over MiniLM embeddings (0 = in scope, 1 = out)."""

    def __init__(self, C: float = 1.0, seed: int = 0):
        self.C = float(C)
        self.seed = int(seed)
        self.model = LogisticRegression(C=self.C, class_weight="balanced",
                                        max_iter=2000, random_state=self.seed)

    def fit(self, in_scope_embeddings, out_of_scope_embeddings):
        X = np.vstack([np.asarray(in_scope_embeddings, dtype="float32"),
                       np.asarray(out_of_scope_embeddings, dtype="float32")])
        y = np.r_[np.zeros(len(in_scope_embeddings), dtype=int),
                  np.ones(len(out_of_scope_embeddings), dtype=int)]
        self.model.fit(X, y)
        return self

    def scores(self, embeddings) -> np.ndarray:
        X = np.asarray(embeddings, dtype="float32")
        proba = self.model.predict_proba(X)
        return proba[:, list(self.model.classes_).index(0)]

    def save(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": self.model, "C": self.C, "seed": self.seed}, path)
