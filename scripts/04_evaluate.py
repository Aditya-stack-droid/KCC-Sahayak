"""Evaluate the full pipeline on the held-out test split:

1. Intent classification accuracy / F1 (already reported by the training
   script, re-loaded here for the combined report).
2. Retrieval quality: for each test question, does nearest-neighbour search
   over the train+val corpus return an answer from the *same weakly-labeled
   query_type* as the test question (a proxy for "is the retrieved answer
   actually relevant" in the absence of human relevance judgments).
3. The accuracy-automation trade-off curve: as the confidence gate's
   thresholds tighten, what fraction of test queries get auto-answered
   (coverage) vs held back (safety), and what is retrieval-type-match
   accuracy *within* the auto-answered subset at each threshold -- this is
   the selective-classification risk-coverage curve the literature survey
   grounds the confidence-gate design in (Chow 1970; Geifman & El-Yaniv 2017).

Usage:
    .venv/Scripts/python.exe scripts/04_evaluate.py
"""
import json
import sys
from pathlib import Path

import faiss
import matplotlib
import numpy as np
import pandas as pd
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from confidence_gate import GateThresholds, decide  # noqa: E402

DATA_DIR = ROOT / "data" / "processed"
INTENT_MODEL_DIR = ROOT / "models" / "intent_classifier"
INDEX_DIR = ROOT / "models" / "retrieval_index"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

RETRIEVAL_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
MAX_EVAL_SAMPLES = 3000  # cap for tractable CPU evaluation


def classify_batch(texts, tokenizer, model, labels, batch_size=64):
    preds, confs = [], []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            inputs = tokenizer(
                batch, truncation=True, padding=True, max_length=64, return_tensors="pt"
            )
            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)
            top_conf, top_idx = probs.max(dim=-1)
            preds.extend([labels[i] for i in top_idx.tolist()])
            confs.extend(top_conf.tolist())
    return preds, confs


def main():
    test_df = pd.read_csv(DATA_DIR / "test.csv").dropna(subset=["questions", "query_type"])
    if len(test_df) > MAX_EVAL_SAMPLES:
        test_df = test_df.sample(MAX_EVAL_SAMPLES, random_state=42)
    print(f"Evaluating on {len(test_df)} test queries")

    labels = json.loads((INTENT_MODEL_DIR / "label_encoder.json").read_text())
    tokenizer = AutoTokenizer.from_pretrained(INTENT_MODEL_DIR)
    intent_model = AutoModelForSequenceClassification.from_pretrained(INTENT_MODEL_DIR)

    print("Running intent classifier on test set ...")
    pred_types, pred_confs = classify_batch(
        test_df["questions"].tolist(), tokenizer, intent_model, labels
    )
    test_df = test_df.copy()
    test_df["pred_query_type"] = pred_types
    test_df["pred_confidence"] = pred_confs

    intent_acc = (test_df["pred_query_type"] == test_df["query_type"]).mean()
    print(f"Intent classification accuracy on eval subset: {intent_acc:.4f}")

    print("Loading retrieval index ...")
    embed_model = SentenceTransformer(RETRIEVAL_MODEL_NAME)
    index = faiss.read_index(str(INDEX_DIR / "kcc_index.faiss"))
    corpus = pd.read_csv(INDEX_DIR / "corpus_metadata.csv")

    print("Encoding test queries and searching index ...")
    test_emb = embed_model.encode(
        test_df["questions"].tolist(), batch_size=128, show_progress_bar=True,
        normalize_embeddings=True,
    ).astype("float32")
    sims, idxs = index.search(test_emb, 1)
    test_df["retrieval_similarity"] = sims[:, 0]
    matched_type = corpus.iloc[idxs[:, 0]]["query_type"].values
    test_df["retrieved_query_type"] = matched_type
    test_df["retrieval_type_match"] = test_df["retrieved_query_type"] == test_df["query_type"]

    retrieval_top1_type_match = test_df["retrieval_type_match"].mean()
    print(f"Retrieval top-1 query_type match rate: {retrieval_top1_type_match:.4f}")

    print("\nSweeping confidence-gate thresholds (risk-coverage curve) ...")
    rows = []
    for conf_th in np.arange(0.3, 1.0, 0.05):
        for sim_th in np.arange(0.3, 0.9, 0.1):
            th = GateThresholds(intent_confidence=conf_th, retrieval_similarity=sim_th)
            decisions = [
                decide(qt, c, s, th).action
                for qt, c, s in zip(
                    test_df["query_type"], test_df["pred_confidence"], test_df["retrieval_similarity"]
                )
            ]
            test_df["_decision"] = decisions
            auto_mask = test_df["_decision"] == "AUTO_ANSWER"
            coverage = auto_mask.mean()
            if auto_mask.sum() > 0:
                risk = 1 - test_df.loc[auto_mask, "retrieval_type_match"].mean()
            else:
                risk = np.nan
            rows.append(
                {
                    "conf_threshold": round(conf_th, 2),
                    "sim_threshold": round(sim_th, 2),
                    "coverage": coverage,
                    "risk_on_automated": risk,
                }
            )

    curve_df = pd.DataFrame(rows)
    curve_df.to_csv(RESULTS_DIR / "risk_coverage_curve.csv", index=False)

    default_th = GateThresholds()
    decisions = [
        decide(qt, c, s, default_th).action
        for qt, c, s in zip(
            test_df["query_type"], test_df["pred_confidence"], test_df["retrieval_similarity"]
        )
    ]
    test_df["decision"] = decisions
    auto_mask = test_df["decision"] == "AUTO_ANSWER"
    default_coverage = auto_mask.mean()
    default_risk = (
        1 - test_df.loc[auto_mask, "retrieval_type_match"].mean() if auto_mask.sum() else float("nan")
    )
    print(
        f"\nDefault thresholds (conf>={default_th.intent_confidence}, "
        f"sim>={default_th.retrieval_similarity}): "
        f"coverage={default_coverage:.3f}, risk_on_automated={default_risk:.3f}"
    )

    summary = {
        "n_eval": len(test_df),
        "intent_classification_accuracy": float(intent_acc),
        "retrieval_top1_query_type_match_rate": float(retrieval_top1_type_match),
        "default_thresholds": {
            "intent_confidence": default_th.intent_confidence,
            "retrieval_similarity": default_th.retrieval_similarity,
        },
        "default_coverage": float(default_coverage),
        "default_risk_on_automated": float(default_risk),
    }
    (RESULTS_DIR / "evaluation_summary.json").write_text(json.dumps(summary, indent=2))
    test_df.to_csv(RESULTS_DIR / "test_predictions.csv", index=False)

    # Plot: for a fixed sim_threshold sweep, coverage vs risk
    plt.figure(figsize=(7, 5))
    for sim_th in sorted(curve_df["sim_threshold"].unique()):
        sub = curve_df[curve_df["sim_threshold"] == sim_th].sort_values("coverage")
        plt.plot(sub["coverage"], sub["risk_on_automated"], marker="o", label=f"sim>={sim_th:.1f}")
    plt.xlabel("Coverage (fraction auto-answered)")
    plt.ylabel("Risk on automated subset (1 - type-match rate)")
    plt.title("KCC-Sahayak: Accuracy-Automation Trade-off (Risk-Coverage Curve)")
    plt.legend(fontsize=8)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "risk_coverage_curve.png", dpi=150)
    print(f"\nSaved results to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
