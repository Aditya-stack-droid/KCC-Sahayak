"""SQLite persistence for the KCC-Sahayak app.

One table, `queries`, is both the audit log of every query the system has
seen and the work queue for human Farm Tele Advisors: escalated queries start
as status='pending' and become 'resolved' when an advisor answers them.
Advisor answers are fed back into the retrieval memory (see app.py), closing
the human-in-the-loop cycle.
"""
import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path

DB_PATH = Path(__file__).resolve().parents[1] / "data" / "app.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS queries (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at        TEXT NOT NULL,
    source            TEXT NOT NULL,
    register          TEXT,
    input_text        TEXT NOT NULL,
    detected_language TEXT,
    query_type        TEXT,
    intent_confidence REAL,
    top_similarity    REAL,
    crops             TEXT,
    symptoms          TEXT,
    candidates_json   TEXT,
    decision          TEXT NOT NULL,
    decision_reason   TEXT,
    suggested_answer  TEXT,
    status            TEXT NOT NULL,
    advisor_answer    TEXT,
    resolved_at       TEXT,
    feedback          INTEGER,
    latency_ms        REAL
);
CREATE INDEX IF NOT EXISTS idx_queries_status ON queries(status);
CREATE INDEX IF NOT EXISTS idx_queries_created ON queries(created_at);
"""


@contextmanager
def connect():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init():
    with connect() as c:
        c.executescript(SCHEMA)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def insert_query(r: dict) -> int:
    """Log a pipeline result. r is the serialized result from app._serialize_result
    plus source/register/latency_ms."""
    status = "answered" if r["decision"] == "AUTO_ANSWER" else "pending"
    with connect() as c:
        cur = c.execute(
            """INSERT INTO queries (created_at, source, register, input_text, detected_language,
                   query_type, intent_confidence, top_similarity, crops, symptoms, candidates_json,
                   decision, decision_reason, suggested_answer, status, latency_ms)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                _now(), r["source"], r.get("register"), r["input_text"],
                (r.get("asr") or {}).get("detected_language"),
                r["query_type"], r["intent_confidence"],
                r["retrieved_top3"][0]["similarity"] if r["retrieved_top3"] else None,
                ";".join(r["entities"].get("crop", [])), ";".join(r["entities"].get("symptom", [])),
                json.dumps(r["retrieved_top3"], ensure_ascii=False),
                r["decision"], r["decision_reason"], r["suggested_answer"], status,
                r.get("latency_ms"),
            ),
        )
        return cur.lastrowid


def _row(r: sqlite3.Row) -> dict:
    d = dict(r)
    d["candidates"] = json.loads(d.pop("candidates_json") or "[]")
    return d


def get(query_id: int):
    with connect() as c:
        row = c.execute("SELECT * FROM queries WHERE id=?", (query_id,)).fetchone()
    return _row(row) if row else None


def list_queue(status: str = "pending", limit: int = 100):
    with connect() as c:
        rows = c.execute(
            "SELECT * FROM queries WHERE status=? ORDER BY id DESC LIMIT ?", (status, limit)
        ).fetchall()
    return [_row(r) for r in rows]


def recent(limit: int = 20):
    with connect() as c:
        rows = c.execute("SELECT * FROM queries ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    return [_row(r) for r in rows]


def resolve(query_id: int, answer: str) -> bool:
    with connect() as c:
        cur = c.execute(
            "UPDATE queries SET status='resolved', advisor_answer=?, resolved_at=? "
            "WHERE id=? AND status='pending'",
            (answer, _now(), query_id),
        )
        return cur.rowcount == 1


def set_feedback(query_id: int, value: int) -> bool:
    with connect() as c:
        cur = c.execute("UPDATE queries SET feedback=? WHERE id=?", (value, query_id))
        return cur.rowcount == 1


def resolved_for_memory():
    """Advisor-resolved rows, used to rebuild the retrieval memory at startup."""
    with connect() as c:
        rows = c.execute(
            "SELECT id, input_text, advisor_answer FROM queries WHERE status='resolved' "
            "AND advisor_answer IS NOT NULL"
        ).fetchall()
    return [dict(r) for r in rows]


def summary(days: int = 14) -> dict:
    since = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat(timespec="seconds")
    with connect() as c:
        tot = c.execute(
            """SELECT COUNT(*) n,
                      SUM(decision='AUTO_ANSWER') auto,
                      SUM(decision='ESCALATE') esc,
                      SUM(status='pending') pending,
                      SUM(status='resolved') resolved,
                      AVG(intent_confidence) avg_conf,
                      AVG(latency_ms) avg_latency,
                      SUM(feedback=1) up, SUM(feedback=-1) down
               FROM queries"""
        ).fetchone()
        by_type = c.execute(
            "SELECT query_type k, COUNT(*) n FROM queries GROUP BY query_type ORDER BY n DESC"
        ).fetchall()
        by_reg = c.execute(
            "SELECT COALESCE(register,'en') k, COUNT(*) n FROM queries GROUP BY 1 ORDER BY n DESC"
        ).fetchall()
        by_day = c.execute(
            """SELECT substr(created_at,1,10) day, COUNT(*) n, SUM(decision='AUTO_ANSWER') auto
               FROM queries WHERE created_at>=? GROUP BY 1 ORDER BY 1""",
            (since,),
        ).fetchall()
    n = tot["n"] or 0
    return {
        "total": n,
        "auto_answered": tot["auto"] or 0,
        "escalated": tot["esc"] or 0,
        "pending": tot["pending"] or 0,
        "resolved": tot["resolved"] or 0,
        "automation_rate": round((tot["auto"] or 0) / n, 4) if n else None,
        "avg_intent_confidence": round(tot["avg_conf"], 4) if tot["avg_conf"] is not None else None,
        "avg_latency_ms": round(tot["avg_latency"], 1) if tot["avg_latency"] is not None else None,
        "feedback_up": tot["up"] or 0,
        "feedback_down": tot["down"] or 0,
        "by_type": {r["k"]: r["n"] for r in by_type},
        "by_register": {r["k"]: r["n"] for r in by_reg},
        "by_day": [{"day": r["day"], "total": r["n"], "auto": r["auto"] or 0} for r in by_day],
    }
