"""Focused tests for the rule-based multi-intent decomposer (standalone, escalate-only)."""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT / "src", ROOT / "experiments"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from glossary import HIGH_STAKES_TYPES, TIME_SENSITIVE_TYPES  # noqa: E402

from multi_intent import cases as cases_module  # noqa: E402
from multi_intent import evaluate as evaluate_module  # noqa: E402
from multi_intent import report as report_module  # noqa: E402
from multi_intent.decompose import decompose, intent_set  # noqa: E402

DATA = ROOT / "experiments" / "multi_intent" / "data" / "multi_intent_cases.csv"


def test_case_set_contract():
    assert DATA.exists(), "run experiments/multi_intent/cases.py first"
    rows = evaluate_module.load_cases()
    assert len(rows) >= 100
    assert all(r["synthetic"] == "true" for r in rows)
    assert len({r["case_id"] for r in rows}) == len(rows)
    cats = {r["category"] for r in rows}
    assert {"two_unrelated", "two_related", "three_intent", "single_control"} <= cats
    assert {"en", "devanagari", "roman_hindi", "hinglish"} <= {r["language"] for r in rows}


def test_two_unrelated_requests_are_split():
    result = decompose("gehu ka bhav kya hai aur urea kitna dalen?")
    assert result.is_multi
    assert len(result.sub_queries) >= 2
    assert set(result.distinct_intents) == {"Market Price", "Nutrient Management"}


def test_intent_labels_on_clear_cases():
    assert decompose("What is the weather forecast and which tomato variety should I sow?").distinct_intents == ["Varietal Information", "Weather"]
    assert set(decompose("Which fertiliser is best for paddy and what is the capital of Japan?").distinct_intents) == {"Nutrient Management", "Other"}
    assert set(decompose("How do I control aphids in mustard and who won yesterday cricket match?").distinct_intents) == {"Plant Protection", "Other"}


def test_mixed_safe_and_high_risk_flags_escalation():
    result = decompose("How do I sow wheat and what is the monocrotophos dose for aphids?")
    assert result.is_multi
    assert result.safety_escalation_required
    assert result.sub_queries  # sub-queries preserved


def test_single_request_is_not_flagged_multi():
    for text in [
        "Which tomato variety is good for me?",
        "Apply 50 kg urea per acre to wheat.",
        "गेहूं का भाव क्या है?",
        "Meri gaay bimar hai.",
    ]:
        result = decompose(text)
        assert not result.is_multi, text
        assert len(result.distinct_intents) == 1


def test_devanagari_two_intents():
    assert intent_set("गेहूं का भाव क्या है और यूरिया कितना डालें?") == ("Market Price", "Nutrient Management")


def test_original_text_is_preserved_verbatim():
    text = "What is the price of urea and how much should I apply to my wheat crop?"
    assert decompose(text).input_text == text


def test_bad_inputs_do_not_raise():
    for value in (None, "", "   ", 42):
        result = decompose(value)  # type: ignore[arg-type]
        assert result.is_multi is False


@pytest.fixture(scope="module")
def scored():
    return evaluate_module.evaluate(evaluate_module.load_cases())


def test_decomposer_beats_single_label_baseline(scored):
    metrics, _ = scored
    assert metrics["multi"]["micro"]["f1"] > metrics["single"]["micro"]["f1"]
    assert metrics["multi"]["exact_set_match"] > metrics["single"]["exact_set_match"]
    assert metrics["multi"]["missed_intent_rate"] < metrics["single"]["missed_intent_rate"]
    assert metrics["multi"]["decomposition_accuracy"] > 0.8


def test_safety_escalation_recall_is_high(scored):
    metrics, _ = scored
    assert metrics["multi"]["safety"]["recall"] >= 0.9
    assert metrics["multi"]["safety"]["recall"] > metrics["single"]["safety"]["recall"]


def test_failures_are_recorded_not_hidden(scored):
    metrics, _ = scored
    assert metrics["failures"]["multi_missed_intents"]
    # every recorded miss really is a gold-predicted difference
    for row in metrics["failures"]["multi_missed_intents"]:
        assert set(row["missing"]) <= set(row["gold"])
        assert not (set(row["missing"]) & set(row["predicted"]))


def test_high_risk_definition_matches_glossary(scored):
    metrics, records = scored
    risky = HIGH_STAKES_TYPES | TIME_SENSITIVE_TYPES
    for r in records:
        assert r["gold_safety"] == any(l in risky for l in r["gold"])


def test_evaluation_is_deterministic():
    a = evaluate_module.evaluate(evaluate_module.load_cases())[0]
    b = evaluate_module.evaluate(evaluate_module.load_cases())[0]
    assert a == b


def test_report_renders(scored):
    metrics, _ = scored
    text = report_module.render(metrics)
    assert "Multi-intent detection" in text
    assert "Safety-escalation recall" in text
    assert f"{metrics['multi']['micro']['f1']:.4f}" in text
