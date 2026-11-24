"""FastAPI backend for KCC-Sahayak.

Public API (all JSON under /api):
  GET  /health                     liveness + what is loaded
  GET  /model-info                 model names, labels, default thresholds
  GET  /samples                    example queries (English / Hinglish / Hindi)
  POST /query                      text query -> full pipeline result (logged)
  POST /transcribe                 audio upload -> Whisper -> pipeline (logged)
  GET  /queue?status=pending       escalated queries awaiting a human advisor
  POST /queue/{id}/resolve         advisor answers an escalated query
  POST /queries/{id}/feedback      farmer thumbs up / down on an auto-answer
  GET  /ops/summary                live operational statistics from the log
  GET  /stats                      offline evaluation results (static files)

The frontend is served from the same origin at "/".
"""
import json
import sys
import tempfile
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "backend"))

import db  # noqa: E402
import gemini  # noqa: E402
import rag  # noqa: E402
from answer_quality import is_informative  # noqa: E402
from confidence_gate import GateThresholds, answer_risk  # noqa: E402
from lang import detect_register  # noqa: E402
from query_match import coverage_problem  # noqa: E402
from pipeline import KCCSahayakPipeline  # noqa: E402

FRONTEND_DIR = ROOT / "frontend"
RESULTS_DIR = ROOT / "results"
GOLD_PATH = ROOT / "data" / "gold" / "gold_queries.csv"

MAX_AUDIO_BYTES = 15 * 1024 * 1024

state: dict = {"pipeline": None, "stats": {}}


# ----------------------------------------------------------------- models
class QueryRequest(BaseModel):
    text: str = Field(min_length=2, max_length=1000)
    intent_confidence_threshold: float = Field(0.70, ge=0.0, le=1.0)
    retrieval_similarity_threshold: float = Field(0.55, ge=0.0, le=1.0)
    screen_answer: bool = True


class ResolveRequest(BaseModel):
    answer: str = Field(min_length=3, max_length=2000)
    add_to_memory: bool = True


class FeedbackRequest(BaseModel):
    value: int = Field(description="+1 helpful, -1 not helpful")


# --------------------------------------------------------------- lifecycle
def _load_json(name: str):
    p = RESULTS_DIR / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def build_stats() -> dict:
    s = {
        "evaluation": _load_json("evaluation_summary.json"),
        "baselines": _load_json("baselines.json"),
        "gold_eval": _load_json("gold_eval.json"),
        "intent_v2": _load_json("intent_v2_eval.json"),
        "intent_v2_aug": _load_json("intent_v2_eval_aug.json"),
        "annotation": _load_json("annotation_summary.json"),
        "geography": _load_json("corpus_geography.json"),
        "calibration": _load_json("calibration.json"),
        "answer_screening": _load_json("answer_screening.json"),
        "research_eval": _load_json("research_eval.json"),
        "selective_eval": _load_json("selective_eval.json"),
        "hybrid_retrieval": _load_json("hybrid_retrieval.json"),
        "voice_eval": _load_json("voice_eval.json"),
        "model_selection": _load_json("model_selection.json"),
        "encoder_comparison": _load_json("encoder_comparison.json"),
        "retrieval_baselines": _load_json("retrieval_baselines.json"),
    }
    curve = RESULTS_DIR / "risk_coverage_curve.csv"
    if curve.exists():
        import pandas as pd

        s["risk_coverage_curve"] = pd.read_csv(curve).to_dict(orient="records")
    return {k: v for k, v in s.items() if v is not None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    db.init()
    needed = [ROOT / "models" / "intent_embed_lr" / "model.joblib",
              ROOT / "models" / "retrieval_index" / "kcc_index.faiss",
              ROOT / "models" / "retrieval_index" / "embeddings.npy"]
    missing = [str(p.relative_to(ROOT)) for p in needed if not p.exists()]
    if missing:
        raise RuntimeError("Model files missing: " + ", ".join(missing)
                           + ". First-time setup: run `python scripts/setup_models.py` (about 10-20 minutes, once), then start again.")
    print("Loading KCC-Sahayak pipeline ...")
    pipe = KCCSahayakPipeline()
    for r in db.resolved_for_memory():
        pipe.add_memory(r["input_text"], r["advisor_answer"], r["id"])
    state["pipeline"] = pipe
    state["stats"] = build_stats()
    print(f"Pipeline ready ({len(pipe.mem_items)} advisor answers in memory).")
    yield


app = FastAPI(title="KCC-Sahayak API", version="2.0", lifespan=lifespan)


def _pipe() -> KCCSahayakPipeline:
    if state["pipeline"] is None:
        raise HTTPException(503, "Pipeline is still loading")
    return state["pipeline"]


def _serialize(result: dict, source: str, latency_ms: float) -> dict:
    return {
        "input_text": result["input_text"],
        "source": source,
        "register": result["register"],
        "asr": result["asr"],
        "query_type": result["query_type"],
        "intent_confidence": round(result["intent_confidence"], 4),
        "intent_top3": [{"label": l, "prob": round(p, 4)} for l, p in result["intent_top3"]],
        "entities": result["entities"],
        "retrieved_top3": [
            {
                "similarity": round(r["similarity"], 4),
                "matched_question": r["matched_question"],
                "answer": r["answer"],
                "source": r.get("source", "corpus"),
                "topic": r.get("type"),
                # fit to show a farmer as a reference: real advice, no dose/chemical text, and it addresses the question
                "reference_ok": bool(r["answer"]) and is_informative(r["answer"]) and answer_risk(r["answer"]) is None
                                and coverage_problem(result["input_text"], r.get("matched_question", ""), r["answer"]) is None,
            }
            for r in result["retrieved_top3"]
        ],
        "decision": result["decision"],
        "decision_code": result["decision_code"],
        "decision_reason": result["decision_reason"],
        "suggested_answer": result["suggested_answer"],
        "latency_ms": round(latency_ms, 1),
    }


# ------------------------------------------------------------------ routes
@app.get("/api/health")
def health():
    p = state["pipeline"]
    return {
        "status": "ok",
        "pipeline_loaded": p is not None,
        "advisor_memory": len(p.mem_items) if p else 0,
    }


@app.get("/api/model-info")
def model_info():
    p = _pipe()
    return {
        "intent_backend": p.intent_backend,
        "retrieval_model": "paraphrase-multilingual-MiniLM-L12-v2",
        "labels": p.labels,
        "temperature": p.temperature,
        "corpus_size": int(len(p.corpus)),
        "default_thresholds": {"intent_confidence": 0.70, "retrieval_similarity": 0.55},
        "high_stakes_types": ["Plant Protection", "Nutrient Management", "Government Scheme"],
    }


@app.get("/api/samples")
def samples():
    if not GOLD_PATH.exists():
        return {"samples": []}
    import pandas as pd

    g = pd.read_csv(GOLD_PATH)
    # first query of each (style, label) pair keeps the chips diverse and stable
    picked = g.groupby(["style", "label"], sort=False).head(1)
    return {
        "samples": [
            {"text": r.text, "label": r.label, "style": r.style} for r in picked.itertuples()
        ]
    }


def _run_and_log(text: str | None, audio_path: str | None, th: GateThresholds, source: str,
                 register: str | None = None) -> dict:
    t0 = time.perf_counter()
    result = _pipe().run(text, from_audio=audio_path, thresholds=th)
    latency_ms = (time.perf_counter() - t0) * 1000
    out = _serialize(result, source, latency_ms)
    if register:
        out["register"] = register
    out["query_id"] = db.insert_query(out)
    return out


@app.post("/api/query")
def query(req: QueryRequest):
    th = GateThresholds(
        intent_confidence=req.intent_confidence_threshold,
        retrieval_similarity=req.retrieval_similarity_threshold,
        screen_answer=req.screen_answer,
    )
    text = req.text.strip()
    reg = detect_register(text)
    if reg == "en":
        return _run_and_log(text, None, th, "text")
    # The corpus is English. Matching an English question against it is more precise than matching Hindi or Assamese
    # directly (a Hindi "late sowing in December" question retrieved a normal-sowing answer), so the question is
    # translated first and the original is kept for display. If translation is unavailable we fall back to the
    # original text, which the multilingual encoder can still handle.
    tr = gemini.translate(text, "en")
    out = _run_and_log(tr["text"] or text, None, th, "text", register=reg)
    out["original_text"] = text
    out["translation_error"] = None if tr["text"] else tr["error"]
    return out


class GeminiRequest(BaseModel):
    text: str = Field(min_length=2, max_length=1000)


@app.get("/api/gemini/status")
def gemini_status():
    return {"configured": gemini.configured(), "model": gemini.model_name()}


class TranslateRequest(BaseModel):
    text: str = Field(min_length=1, max_length=2000)
    lang: str = Field(min_length=2, max_length=10)


@app.post("/api/translate")
def translate(req: TranslateRequest):
    """Machine translation of an English answer into the farmer's language. Display only."""
    return gemini.translate(req.text, req.lang)


class RagRequest(BaseModel):
    text: str = Field(min_length=2, max_length=1000)           # question as used for retrieval (English)
    original_text: str | None = Field(default=None, max_length=1000)   # what the farmer typed, sets the reply language


@app.post("/api/rag")
def rag_answer(req: RagRequest):
    """Gemini reply grounded in retrieved KCC answers, with sources. Display only: never changes the gate."""
    return rag.answer(_pipe(), req.text.strip(), (req.original_text or "").strip() or None)


@app.post("/api/gemini")
def gemini_suggestion(req: GeminiRequest):
    """Second opinion only. Never affects the gate decision and is not stored."""
    return gemini.suggest(req.text)


@app.post("/api/transcribe")
async def transcribe(file: UploadFile = File(...)):
    data = await file.read()
    if not data:
        raise HTTPException(400, "Empty audio upload")
    if len(data) > MAX_AUDIO_BYTES:
        raise HTTPException(413, "Audio too large (15 MB max)")
    suffix = Path(file.filename or "audio.webm").suffix or ".webm"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(data)
        path = tmp.name
    try:
        return _run_and_log(None, path, GateThresholds(), "voice")
    except HTTPException:
        raise
    except Exception as e:  # ffmpeg missing, undecodable audio, ...
        raise HTTPException(422, f"Could not process audio: {e}")
    finally:
        Path(path).unlink(missing_ok=True)


@app.get("/api/queue")
def queue(status: str = Query("pending", pattern="^(pending|resolved|answered)$"), limit: int = 100):
    return {"items": db.list_queue(status, min(limit, 200))}


@app.post("/api/queue/{query_id}/resolve")
def resolve(query_id: int, req: ResolveRequest):
    row = db.get(query_id)
    if row is None:
        raise HTTPException(404, "Query not found")
    if not db.resolve(query_id, req.answer.strip()):
        raise HTTPException(409, "Query is not pending (already resolved or auto-answered)")
    if req.add_to_memory:
        _pipe().add_memory(row["input_text"], req.answer.strip(), query_id)
    return {"ok": True, "query_id": query_id, "added_to_memory": req.add_to_memory}


@app.post("/api/queries/{query_id}/feedback")
def feedback(query_id: int, req: FeedbackRequest):
    if req.value not in (1, -1):
        raise HTTPException(422, "value must be +1 or -1")
    if not db.set_feedback(query_id, req.value):
        raise HTTPException(404, "Query not found")
    return {"ok": True}


@app.get("/api/queries/recent")
def recent(limit: int = 20):
    return {"items": db.recent(min(limit, 100))}


@app.get("/api/ops/summary")
def ops_summary(days: int = 14):
    return db.summary(days)


@app.get("/api/stats")
def stats():
    # re-read on every request: result files change whenever an experiment is re-run
    state["stats"] = build_stats()
    return state["stats"]


@app.exception_handler(Exception)
async def unhandled(_, exc: Exception):
    return JSONResponse(status_code=500, content={"detail": f"Internal error: {type(exc).__name__}"})


app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
