"""RAG answer: grounding, safety checks and fallbacks. No network and no models: Gemini and the retriever are faked."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "backend"))

import gemini  # noqa: E402
import rag  # noqa: E402


class FakePipe:
    def __init__(self, qtype, cands):
        self.qtype, self.cands = qtype, cands

    def _embed(self, text):
        return None

    def classify_intent(self, text, emb=None):
        return self.qtype, 0.95, []

    def retrieve(self, text, emb=None, k=3, query_type=None, pool=8):
        return self.cands


def cand(q, a, sim=0.8):
    return {"matched_question": q, "answer": a, "similarity": sim}


GOOD = [cand("asking about sowing time of mustard", "advised him to sow seeds from mid oct- mid nov"),
        cand("asking about mustard sowing", "sow in the first fortnight of october")]


@pytest.fixture(autouse=True)
def fakes(monkeypatch):
    calls = {"gen": [], "plain": []}

    def fake_generate(system, text, temperature=0.3, tag=""):
        calls["gen"].append((system, text, tag))
        return {"available": True, "text": calls.get("reply", "Sow mustard from mid October to mid November [1]."),
                "model": "m", "error": None}
    monkeypatch.setattr(gemini, "_generate", fake_generate)
    monkeypatch.setattr(gemini, "suggest", lambda t: calls["plain"].append(t) or
                        {"available": True, "text": "general advice", "model": "m", "error": None})
    return calls


def test_grounded_reply_carries_sources_and_citations(fakes):
    r = rag.answer(FakePipe("Cultural Practices", GOOD), "what is the sowing time for mustard")
    assert r["grounded"] and r["text"].startswith("Sow mustard") and r["cited"] == [1] and not r["notes"]
    system, prompt, tag = fakes["gen"][0]
    assert tag == "rag" and "ONLY the numbered past advisor answers" in system
    assert "[1] Past question: asking about sowing time of mustard" in prompt and "mid oct- mid nov" in prompt
    assert [s["n"] for s in r["sources"]] == [1, 2]


def test_unsafe_and_irrelevant_sources_never_reach_the_prompt(fakes):
    cands = [cand("asking about mustard aphids", "suggested him to spray rogor@2ml/lit at evening time"),   # dose + chemical
             cand("asking about mustard", "explained in details"),                                         # call-log note
             cand("asking about sowing time of mustard", "sow in mid october", sim=0.3),                    # too dissimilar
             cand("asking about wheat sowing time", "sow in november")]                                    # wrong crop
    r = rag.answer(FakePipe("Cultural Practices", cands), "what is the sowing time for mustard")
    assert not r["grounded"] and fakes["gen"] == [] and fakes["plain"]      # nothing to ground on -> plain Gemini
    assert "NOT grounded" in r["notes"][0] or "not grounded" in r["notes"][0]


def test_live_data_questions_skip_retrieval_and_say_so(fakes):
    pipe = FakePipe("Weather", GOOD)
    r = rag.answer(pipe, "is it going to rain tomorrow")
    assert not r["grounded"] and fakes["gen"] == [] and "Live-data" in r["notes"][0]


def test_reply_with_a_dose_is_withheld(fakes):
    fakes["reply"] = "Apply 2 ml per litre of water [1]."
    r = rag.answer(FakePipe("Cultural Practices", GOOD), "what is the sowing time for mustard")
    assert r["withheld"] and r["text"] is None and "dose" in r["notes"][0]


def test_numbers_not_in_sources_are_flagged_but_sources_and_helpline_are_not(fakes):
    fakes["reply"] = "Sow in mid October [1], about 45 days of growth. Call 1800-180-1551."
    r = rag.answer(FakePipe("Cultural Practices", GOOD), "what is the sowing time for mustard")
    assert r["unsupported_numbers"] == ["45"] and "45" in r["notes"][0]
    fakes["reply"] = "Sow seeds from mid oct to mid nov [1]. Call 1800-180-1551."
    assert rag.answer(FakePipe("Cultural Practices", GOOD), "what is the sowing time for mustard")["unsupported_numbers"] == []


def test_grouped_citations_are_read_and_not_mistaken_for_figures(fakes):
    fakes["reply"] = "Sow from mid October to mid November [1, 2], and up to November [2]."
    r = rag.answer(FakePipe("Cultural Practices", GOOD), "what is the sowing time for mustard")
    assert r["cited"] == [1, 2] and r["unsupported_numbers"] == [] and not r["notes"]


def test_devanagari_digits_are_compared_as_numbers():
    src = [{"n": 1, "question": "q", "answer": "irrigate at 25 days"}]
    assert rag.unsupported_numbers("२५ दिन पर सिंचाई करें [1]", src, "") == []
    assert rag.unsupported_numbers("३० दिन पर सिंचाई करें [1]", src, "") == ["30"]


def test_reply_without_citation_is_noted(fakes):
    fakes["reply"] = "Sow mustard in mid October."
    r = rag.answer(FakePipe("Cultural Practices", GOOD), "what is the sowing time for mustard")
    assert r["cited"] == [] and any("cites no source" in n for n in r["notes"])


def test_original_language_question_goes_to_the_prompt_but_retrieval_uses_english(fakes):
    rag.answer(FakePipe("Cultural Practices", GOOD), "what is the sowing time for mustard", original="सरसों की बुवाई कब करें")
    assert "Farmer's question: सरसों की बुवाई कब करें" in fakes["gen"][0][1]
