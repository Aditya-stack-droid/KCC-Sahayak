"""Gemini second-opinion feature: never calls the network in tests, never leaks the key."""
import io
import json
import sys
import urllib.error
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "backend"))

import gemini  # noqa: E402


@pytest.fixture(autouse=True)
def clean(monkeypatch):
    monkeypatch.setattr(gemini, "_load_dotenv", lambda: None)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_MODEL", raising=False)
    gemini._cache.clear()


class FakeResp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_not_configured_makes_no_network_call(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("network must not be touched without a key")
    monkeypatch.setattr(gemini.urllib.request, "urlopen", boom)
    r = gemini.suggest("which tomato variety?")
    assert r["available"] is False and r["text"] is None and "GEMINI_API_KEY" in r["error"]


def test_success_returns_text_and_sends_key_in_header_not_url(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "SECRET-KEY")
    seen = {}

    def fake(req, timeout):
        seen["url"], seen["headers"], seen["body"] = req.full_url, dict(req.header_items()), json.loads(req.data)
        return FakeResp(json.dumps({"candidates": [{"content": {"parts": [{"text": "Use pheromone traps."}]}}]}).encode())
    monkeypatch.setattr(gemini.urllib.request, "urlopen", fake)
    r = gemini.suggest("tomato fruit borer")
    assert r["text"] == "Use pheromone traps." and r["error"] is None
    assert "SECRET-KEY" not in seen["url"] and seen["headers"]["X-goog-api-key"] == "SECRET-KEY"
    assert "NEVER give exact pesticide" in seen["body"]["systemInstruction"]["parts"][0]["text"]


def test_http_error_is_reported_without_leaking_key_or_body(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "SECRET-KEY")

    def fake(req, timeout):
        raise urllib.error.HTTPError(req.full_url, 429, "Too Many", {}, io.BytesIO(b"quota SECRET-KEY"))
    monkeypatch.setattr(gemini.urllib.request, "urlopen", fake)
    r = gemini.suggest("anything here")
    assert r["text"] is None and "429" in r["error"] and "SECRET-KEY" not in json.dumps(r)


def test_empty_or_blocked_response_is_handled(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    monkeypatch.setattr(gemini.urllib.request, "urlopen",
                        lambda req, timeout: FakeResp(json.dumps({"promptFeedback": {"blockReason": "SAFETY"}}).encode()))
    r = gemini.suggest("some question")
    assert r["text"] is None and "SAFETY" in r["error"]


def test_network_failure_never_raises(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "k")

    def fake(req, timeout):
        raise TimeoutError()
    monkeypatch.setattr(gemini.urllib.request, "urlopen", fake)
    assert "Could not reach Gemini" in gemini.suggest("question text")["error"]


def test_identical_questions_are_cached(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    calls = []

    def fake(req, timeout):
        calls.append(1)
        return FakeResp(json.dumps({"candidates": [{"content": {"parts": [{"text": "ok"}]}}]}).encode())
    monkeypatch.setattr(gemini.urllib.request, "urlopen", fake)
    gemini.suggest("same question")
    gemini.suggest("same question")
    assert len(calls) == 1
