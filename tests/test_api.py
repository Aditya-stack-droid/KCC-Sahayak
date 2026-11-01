"""API tests with a stub pipeline: exercises validation, logging, the advisor queue,
memory feedback and error handling without loading any model."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "backend"))

from fastapi.testclient import TestClient  # noqa: E402

import app as app_module  # noqa: E402
import db  # noqa: E402


class StubPipeline:
    """Returns an escalation for 'pest' queries and an auto-answer otherwise."""

    labels = ["Plant Protection", "Varietal Information"]
    intent_backend = "stub"
    temperature = 1.0
    corpus = [0, 1, 2]

    def __init__(self):
        self.mem_items = []

    def add_memory(self, question, answer, query_id=None):
        self.mem_items.append((question, answer, query_id))

    def run(self, text, from_audio=None, thresholds=None):
        pest = "pest" in (text or "").lower()
        cand = [{"similarity": 0.8, "matched_question": "asking about pests", "answer": "spray something", "source": "corpus"}]
        return {
            "input_text": text, "register": "en", "asr": None,
            "query_type": "Plant Protection" if pest else "Varietal Information",
            "intent_confidence": 0.95, "intent_top3": [("Plant Protection", 0.9), ("Other", 0.05)],
            "entities": {"crop": [], "symptom": []}, "retrieved_top3": cand,
            "decision": "ESCALATE" if pest else "AUTO_ANSWER",
            "decision_code": "high_stakes" if pest else "ok",
            "decision_reason": "stub reason", "suggested_answer": None if pest else "use variety x",
        }


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "test.db")
    db.init()
    stub = StubPipeline()
    monkeypatch.setitem(app_module.state, "pipeline", stub)
    c = TestClient(app_module.app)  # no context manager -> lifespan (model loading) is skipped
    c.stub = stub
    return c


def test_health_reports_pipeline(client):
    r = client.get("/api/health").json()
    assert r["status"] == "ok" and r["pipeline_loaded"] is True


def test_query_is_validated(client):
    assert client.post("/api/query", json={"text": "x"}).status_code == 422          # too short
    assert client.post("/api/query", json={"text": "a" * 1001}).status_code == 422   # too long
    assert client.post("/api/query", json={"text": "ok query", "intent_confidence_threshold": 2}).status_code == 422


def test_auto_answered_query_is_logged_and_not_queued(client):
    r = client.post("/api/query", json={"text": "good tomato variety?"}).json()
    assert r["decision"] == "AUTO_ANSWER" and r["suggested_answer"] == "use variety x" and r["query_id"]
    assert client.get("/api/queue?status=pending").json()["items"] == []
    assert client.get("/api/ops/summary").json()["auto_answered"] == 1


def test_escalated_query_enters_queue_and_can_be_resolved(client):
    r = client.post("/api/query", json={"text": "pest on my wheat"}).json()
    assert r["decision"] == "ESCALATE" and r["suggested_answer"] is None
    q = client.get("/api/queue?status=pending").json()["items"]
    assert len(q) == 1 and q[0]["id"] == r["query_id"] and q[0]["candidates"]

    ok = client.post(f"/api/queue/{r['query_id']}/resolve", json={"answer": "Use neem oil weekly.", "add_to_memory": True})
    assert ok.status_code == 200
    assert client.stub.mem_items == [("pest on my wheat", "Use neem oil weekly.", r["query_id"])]
    assert client.get("/api/queue?status=pending").json()["items"] == []
    assert client.get("/api/queue?status=resolved").json()["items"][0]["advisor_answer"] == "Use neem oil weekly."


def test_memory_is_optional(client):
    r = client.post("/api/query", json={"text": "pest again"}).json()
    client.post(f"/api/queue/{r['query_id']}/resolve", json={"answer": "Do X carefully.", "add_to_memory": False})
    assert client.stub.mem_items == []


def test_resolving_twice_or_unknown_is_rejected(client):
    r = client.post("/api/query", json={"text": "pest here"}).json()
    assert client.post(f"/api/queue/{r['query_id']}/resolve", json={"answer": "First answer."}).status_code == 200
    assert client.post(f"/api/queue/{r['query_id']}/resolve", json={"answer": "Second answer."}).status_code == 409
    assert client.post("/api/queue/99999/resolve", json={"answer": "Whatever works."}).status_code == 404
    assert client.post(f"/api/queue/{r['query_id']}/resolve", json={"answer": "x"}).status_code == 422


def test_feedback_validation(client):
    r = client.post("/api/query", json={"text": "good variety please"}).json()
    assert client.post(f"/api/queries/{r['query_id']}/feedback", json={"value": 1}).status_code == 200
    assert client.post(f"/api/queries/{r['query_id']}/feedback", json={"value": 5}).status_code == 422
    assert client.post("/api/queries/99999/feedback", json={"value": 1}).status_code == 404
    s = client.get("/api/ops/summary").json()
    assert s["feedback_up"] == 1 and s["feedback_down"] == 0


def test_queue_rejects_unknown_status(client):
    assert client.get("/api/queue?status=bogus").status_code == 422


def test_empty_audio_upload_is_rejected(client):
    r = client.post("/api/transcribe", files={"file": ("a.webm", b"", "audio/webm")})
    assert r.status_code == 400


def test_pipeline_not_ready_returns_503(client, monkeypatch):
    monkeypatch.setitem(app_module.state, "pipeline", None)
    assert client.post("/api/query", json={"text": "anything at all"}).status_code == 503


def test_reference_answers_with_chemical_text_are_marked_not_ok(client):
    r = client.post("/api/query", json={"text": "pest on my wheat"}).json()
    assert r["retrieved_top3"][0]["answer"] == "spray something"
    assert r["retrieved_top3"][0]["reference_ok"] is False
