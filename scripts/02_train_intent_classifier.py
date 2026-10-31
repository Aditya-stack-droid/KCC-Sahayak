"""Fine-tune a transformer encoder (default: distilbert-base-multilingual-cased)
for KCC QueryType intent classification on the weak-labeled splits produced
by 01_prepare_data.py.

The brief's first-choice backbones, ai4bharat/indic-bert and
google/muril-base-cased, are both real options here (pass --model to use
either) -- indic-bert is currently a gated HF repo requiring an
authenticated, approved account, which this pipeline does not set up
(no login/account-creation automation), and muril-base-cased is a much
larger mBERT-scale model where a full fine-tune is CPU-hours away.
distilbert-base-multilingual-cased is used by default as the tractable,
ungated, CPU-friendly stand-in with the same "one multilingual/Indic-capable
encoder, not a generic English-first model" design rationale from the
literature survey.

Usage:
    .venv/Scripts/python.exe scripts/02_train_intent_classifier.py
    .venv/Scripts/python.exe scripts/02_train_intent_classifier.py --model google/muril-base-cased --epochs 2
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models" / "intent_classifier"
RESULTS_DIR = ROOT / "results"


class QueryDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=64):
        self.enc = tokenizer(
            list(texts), truncation=True, padding="max_length", max_length=max_len
        )
        self.labels = list(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.enc.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro"),
        "f1_weighted": f1_score(labels, preds, average="weighted"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="distilbert-base-multilingual-cased")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--max_len", type=int, default=64)
    ap.add_argument(
        "--max_train_samples",
        type=int,
        default=20000,
        help="cap training set size for a tractable CPU run; use -1 for all data",
    )
    ap.add_argument(
        "--resume_from_checkpoint",
        default=None,
        help="path to a checkpoint dir to resume an interrupted run from",
    )
    args = ap.parse_args()

    train_df = pd.read_csv(DATA_DIR / "train.csv")
    val_df = pd.read_csv(DATA_DIR / "val.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")

    for d in (train_df, val_df, test_df):
        d.dropna(subset=["questions", "query_type"], inplace=True)

    if args.max_train_samples > 0 and len(train_df) > args.max_train_samples:
        n_total = len(train_df)
        parts = []
        for _, g in train_df.groupby("query_type"):
            n = min(len(g), max(1, args.max_train_samples * len(g) // n_total))
            parts.append(g.sample(n, random_state=42))
        train_df = pd.concat(parts, ignore_index=True)
        print(f"Subsampled train set to {len(train_df)} rows for tractable CPU training")

    le = LabelEncoder()
    le.fit(train_df["query_type"])
    labels_path = MODEL_DIR / "label_encoder.json"
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    labels_path.write_text(json.dumps(list(le.classes_)))

    def encode_labels(d):
        mask = d["query_type"].isin(le.classes_)
        return d[mask], le.transform(d.loc[mask, "query_type"])

    train_df, y_train = encode_labels(train_df)
    val_df, y_val = encode_labels(val_df)
    test_df, y_test = encode_labels(test_df)

    print(f"Loading tokenizer/model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=len(le.classes_)
    )

    train_ds = QueryDataset(train_df["questions"], y_train, tokenizer, args.max_len)
    val_ds = QueryDataset(val_df["questions"], y_val, tokenizer, args.max_len)
    test_ds = QueryDataset(test_df["questions"], y_test, tokenizer, args.max_len)

    training_args = TrainingArguments(
        output_dir=str(MODEL_DIR / "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
    )

    print("Training ...")
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    print("Evaluating on held-out test set ...")
    test_metrics = trainer.evaluate(test_ds)
    print(test_metrics)

    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "intent_classifier_test_metrics.json").write_text(
        json.dumps(test_metrics, indent=2)
    )

    print(f"Saving model to {MODEL_DIR}")
    model.save_pretrained(MODEL_DIR)
    tokenizer.save_pretrained(MODEL_DIR)


if __name__ == "__main__":
    main()
