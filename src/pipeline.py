"""End-to-end KCC-Sahayak inference pipeline:

    audio or text query
        -> (ASR if audio)
        -> ONE multilingual sentence encoder (paraphrase-multilingual-MiniLM-L12-v2)
             |-> intent: logistic regression on the embedding, temperature-calibrated
             `-> retrieval: hybrid dense (FAISS) + TF-IDF over ~125k logged Q/A pairs,
                 merged with answers human advisors have resolved since deployment
        -> entity extraction (crop / symptom glossary)
        -> confidence gate (type, confidence, similarity, answer screening)
           -> AUTO_ANSWER or ESCALATE

If models/intent_embed_lr/ is absent the pipeline falls back to the legacy
fine-tuned DistilBERT head (models/intent_classifier/).
"""
import json
import sys
import threading
from pathlib import Path

import joblib
import numpy as np
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from confidence_gate import GateThresholds, decide  # noqa: E402
from intent_model import IntentHead  # noqa: E402
from lang import detect_register  # noqa: E402
from ner import extract_entities  # noqa: E402
from query_match import coverage_problem  # noqa: E402
from retrieval import HybridRetriever  # noqa: E402

LEGACY_INTENT_DIR = ROOT / "models" / "intent_classifier"
LR_PATH = ROOT / "models" / "intent_embed_lr" / "model.joblib"
RETRIEVAL_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"


def _softmax(z):
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


class KCCSahayakPipeline:
    def __init__(self, thresholds: GateThresholds | None = None):
        self.thresholds = thresholds or GateThresholds()
        self._lock = threading.Lock()

        print("Loading multilingual encoder ...")
        self.embed_model = SentenceTransformer(RETRIEVAL_MODEL_NAME)

        self.temperature = 1.0
        if LR_PATH.exists():
            self.head = IntentHead.load(LR_PATH)
            self.lr = self.head.lr
            self.labels = list(self.head.classes)
            self.temperature = self.head.temperature
            self.intent_backend = (
                "MiniLM embedding + char-n-gram TF-IDF -> logistic regression"
                if self.head.tfidf is not None else "MiniLM embedding -> logistic regression"
            )
            self._legacy = None
        else:  # legacy fine-tuned DistilBERT head
            import torch  # noqa: F401
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            self.lr = None
            self.tokenizer = AutoTokenizer.from_pretrained(LEGACY_INTENT_DIR)
            self._legacy = AutoModelForSequenceClassification.from_pretrained(LEGACY_INTENT_DIR).eval()
            self.labels = json.loads((LEGACY_INTENT_DIR / "label_encoder.json").read_text())
            self.intent_backend = "distilbert-base-multilingual-cased (legacy)"
        print(f"Intent backend: {self.intent_backend}  (T={self.temperature:.2f})")

        print("Loading retrieval index ...")
        self.retriever = HybridRetriever()
        self.corpus = self.retriever.meta
        self.retrieval_mode = "hybrid (dense+tf-idf)" if self.retriever.is_hybrid else "dense"

        # Answers resolved by human advisors after deployment (human-in-the-loop memory)
        self.mem_emb = np.zeros((0, self.retriever.index.d), dtype="float32")
        self.mem_items: list[dict] = []

    # ---------------------------------------------------------------- encoder
    def _embed(self, text: str) -> np.ndarray:
        with self._lock:
            return self.embed_model.encode([text], normalize_embeddings=True,
                                           show_progress_bar=False).astype("float32")[0]

    # ----------------------------------------------------------------- intent
    def classify_intent(self, text: str, emb: np.ndarray | None = None):
        """Return (label, calibrated confidence, top-3 [(label, prob), ...])."""
        if self.lr is not None:
            emb = self._embed(text) if emb is None else emb
            probs = self.head.proba(emb.reshape(1, -1), [text])[0]
        else:
            import torch

            enc = self.tokenizer(text, truncation=True, padding=True, max_length=64, return_tensors="pt")
            with self._lock, torch.no_grad():
                logits = self._legacy(**enc).logits[0]
            probs = torch.softmax(logits / self.temperature, dim=-1).numpy()
        order = np.argsort(-probs)[:3]
        top3 = [(self.labels[int(i)], float(probs[i])) for i in order]
        return top3[0][0], top3[0][1], top3

    # -------------------------------------------------------------- retrieval
    def add_memory(self, question: str, answer: str, query_id: int | None = None):
        """Add an advisor-verified Q/A pair so future similar queries can see it."""
        emb = self._embed(question).reshape(1, -1)
        self.mem_emb = np.vstack([self.mem_emb, emb])
        self.mem_items.append({"question": question, "answer": answer, "query_id": query_id})

    def _types_of(self, embs: np.ndarray, texts: list[str]) -> list[str | None]:
        """Predicted topic of each candidate question (None if no classifier head)."""
        if self.lr is None or not len(texts):
            return [None] * len(texts)
        p = self.head.proba(embs, texts)
        return [self.labels[int(i)] for i in p.argmax(1)]

    def retrieve(self, text: str, emb: np.ndarray | None = None, k: int = 3,
                 query_type: str | None = None, pool: int = 8):
        """Top-k logged answers. When `query_type` is given, candidates whose own topic
        matches it are ranked first: retrieval alone often returns an off-topic answer,
        especially for Roman-script Hindi."""
        emb = self._embed(text) if emb is None else emb
        hits = self.retriever.search(emb, text, k=pool)
        results = []
        for h in hits:
            row = self.corpus.iloc[h["id"]]
            results.append({
                "similarity": h["similarity"], "fused": h["fused"], "_emb": np.asarray(self.retriever.emb[h["id"]]),
                "matched_question": row["questions"], "answer": row["answers"], "source": "corpus",
            })
        if self.mem_items:
            msims = self.mem_emb @ emb
            for i, (sim, item) in enumerate(zip(msims.tolist(), self.mem_items)):
                # advisor-verified answers rank on dense similarity alone (a mild boost)
                results.append({
                    "similarity": float(sim), "fused": float(sim), "_emb": self.mem_emb[i],
                    "matched_question": item["question"], "answer": item["answer"], "source": "advisor",
                })
        if results:
            types = self._types_of(np.vstack([r["_emb"] for r in results]), [r["matched_question"] for r in results])
            for r, t in zip(results, types):
                r["type"] = t
        for r in results:
            r.pop("_emb", None)
        if query_type is not None:
            results.sort(key=lambda r: (r.get("type") == query_type, r["fused"]), reverse=True)
        else:
            results.sort(key=lambda r: r["fused"], reverse=True)
        return results[:k]

    # -------------------------------------------------------------------- run
    def run(self, text: str | None, from_audio: str | None = None,
            thresholds: GateThresholds | None = None) -> dict:
        th = thresholds or self.thresholds
        if from_audio:
            from asr import transcribe

            asr_result = transcribe(from_audio)
            text = asr_result["text"]
        else:
            asr_result = None

        emb = self._embed(text)
        query_type, intent_confidence, top3_intents = self.classify_intent(text, emb)
        entities = extract_entities(text)
        retrieved = self.retrieve(text, emb, k=3, query_type=query_type)
        if not retrieved:
            retrieved = [{"similarity": 0.0, "fused": 0.0, "matched_question": "",
                          "answer": "", "source": "none"}]
        top = retrieved[0]

        gate = decide(
            query_type=query_type,
            intent_confidence=intent_confidence,
            retrieval_similarity=top["similarity"],
            thresholds=th,
            answer=top["answer"],
            retrieved_type=top.get("type"),
            coverage_problem=coverage_problem(text, top.get("matched_question", ""), top.get("answer", "")) if top.get("matched_question") else None,
        )
        return {
            "input_text": text,
            "register": detect_register(text),
            "asr": asr_result,
            "query_type": query_type,
            "intent_confidence": intent_confidence,
            "intent_top3": top3_intents,
            "entities": entities,
            "retrieved_top3": retrieved,
            "decision": gate.action,
            "decision_code": gate.code,
            "decision_reason": gate.reason,
            "suggested_answer": top["answer"] if gate.action == "AUTO_ANSWER" else None,
        }


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")  # Windows consoles default to cp1252
    p = KCCSahayakPipeline()
    for q in [
        "what is the control measure for aphid infestation in mustard crop",
        "gehu mein peela rust aa gaya hai kya spray karna chahiye",
        "आज मंडी में सरसों का भाव क्या चल रहा है?",
        "Is it going to rain in the next three days?",
    ]:
        r = p.run(q)
        print("\n" + "=" * 80)
        print(f"QUERY: {q}  [{r['register']}]")
        print(f"  Intent: {r['query_type']} (confidence={r['intent_confidence']:.2f})")
        print(f"  Entities: {r['entities']}")
        print(f"  Decision: {r['decision']} -- {r['decision_reason']}")
        if r["suggested_answer"]:
            print(f"  Answer: {r['suggested_answer']}")
