"""Tests for the risk-first multi-intent detector (v2) and its safety study."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT / "src", ROOT / "experiments"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from glossary import HIGH_STAKES_TYPES, TIME_SENSITIVE_TYPES  # noqa: E402

from multi_intent import adversarial_cases as adv  # noqa: E402
from multi_intent import evaluate_v2 as ev2  # noqa: E402
from multi_intent import report_v2 as report_module  # noqa: E402
from multi_intent.detect import detect  # noqa: E402
from multi_intent.risk import assess_risk  # noqa: E402

RISKY = set(HIGH_STAKES_TYPES) | set(TIME_SENSITIVE_TYPES)
ADV_DATA = ROOT / "experiments" / "multi_intent" / "data" / "adversarial_cases_v2.csv"


# --------------------------------------------------------------- case set
def test_adversarial_case_set_contract():
    assert ADV_DATA.exists(), "run adversarial_cases.py first"
    rows = ev2.load(ADV_DATA)
    assert len(rows) >= 100
    assert all(r["synthetic"] == "true" for r in rows)
    assert len({r["case_id"] for r in rows}) == len(rows)
    cats = {r["category"] for r in rows}
    for required in ("two_intent", "three_intent", "related", "unrelated",
                     "safe_plus_dangerous", "ambiguous_chemical", "negation",
                     "missing_context", "hi_hinglish_mix", "indirect_dangerous"):
        assert required in cats, required
    assert {"en", "devanagari", "roman_hindi", "hinglish"} <= {r["language"] for r in rows}


def test_gold_safety_is_derived_from_gold_intents():
    for r in ev2.load(ADV_DATA):
        assert r["gold_safety"] == any(i in RISKY for i in r["gold_intents"])


# --------------------------------------------------------------- risk layer
def test_risk_levels():
    assert assess_risk("Apply 50 kg urea per acre", "Nutrient Management").level == "high"
    assert assess_risk("What is the wheat price", "Market Price").level == "high"
    assert assess_risk("gehu me ratua hai kya karein", "Other").level == "high"
    assert assess_risk("Which tomato variety is good", "Varietal Information").level == "low"
    assert assess_risk("fasal kharab ho rahi hai", "Other").level == "medium"


def test_chemical_content_escalates_even_for_a_safe_intent():
    # a "Varietal Information" query that carries a dose must still be high risk
    r = assess_risk("spray 2 ml monocrotophos per litre", "Varietal Information")
    assert r.level == "high" and r.escalate


# --------------------------------------------------------------- detector
@pytest.mark.parametrize("text", [
    "सरसों में कीड़ा लगा है और बीज कौन सा बोएं?",
    "gehu me ratua hai kya karein?",
    "How do I store wheat and where can I sell it?",
    "My goat is sick and what is the price of goat milk?",
    "Which soybean variety and how do I control girdle beetle?",
    "My neighbour sprayed something and the crop improved; what should I use?",
])
def test_previously_failed_high_risk_cases_now_escalate(text):
    assert detect(text).escalate, text


@pytest.mark.parametrize("text", [
    "Which tomato variety is good for me?",
    "Meri gaay bimar hai.",
    "How do I increase milk yield in my buffalo?",
])
def test_safe_controls_do_not_escalate(text):
    assert not detect(text).escalate, text


def test_safe_subquery_cannot_hide_a_dangerous_one():
    d = detect("How do I sow wheat and what is the monocrotophos dose for aphids?")
    assert d.escalate
    assert any(s.escalate for s in d.sub_queries)
    assert d.escalation_reasons


def test_entities_are_preserved_and_unknown_not_invented():
    d = detect("Apply 2 ml monocrotophos on tomato for aphids")
    sub = d.sub_queries[0]
    assert sub.quantity.startswith("2")
    assert sub.crop == "tomato" or sub.crop == "unknown"
    assert sub.pest in ("aphid", "aphids", "unknown")
    # a query with no crop must not fabricate one
    assert detect("There is whitefly on the crop, what should I do?").sub_queries[0].crop == "unknown"


def test_detector_never_raises_on_bad_input():
    for value in (None, "", "   ", 42):
        assert detect(value).escalate is False  # type: ignore[arg-type]


def test_whole_query_risk_is_a_safety_net():
    d = detect("My neighbour sprayed something and the crop improved; what should I use?")
    assert d.whole_query_risk is not None
    assert d.escalate


# --------------------------------------------------------------- study
@pytest.fixture(scope="module")
def study():
    old = ev2.load(ROOT / "experiments" / "multi_intent" / "data" / "multi_intent_cases.csv")
    new = ev2.load(ADV_DATA)
    return ev2.evaluate(old + new)


def test_v2_has_no_high_risk_false_negatives(study):
    assert study["v2"]["high_risk_fn_count"] == 0
    assert study["v2"]["high_risk_false_negatives"] == []


def test_v2_improves_safety_over_v1_and_baseline(study):
    assert study["v2"]["safety_escalation"]["recall"] >= study["v1"]["safety_escalation"]["recall"]
    assert study["v2"]["high_risk_fn_count"] <= study["v1"]["high_risk_fn_count"]
    assert study["v2"]["high_risk_fn_count"] < study["baseline"]["high_risk_fn_count"]


def test_v1_baseline_still_has_the_original_failures(study):
    # the study must not silently "fix" v1: its recorded failures remain
    assert study["v1"]["high_risk_fn_count"] >= 4


def test_report_renders(study):
    text = report_module.render({
        "protocol": {"risk_intents": sorted(RISKY)},
        "original_set": study, "adversarial_set": study, "combined": study,
        "limitations": ["synthetic"],
    })
    assert "Multi-intent safety study" in text
    assert "High-risk false negatives" in text


def test_determinism():
    rows = ev2.load(ADV_DATA)
    assert ev2.evaluate(rows) == ev2.evaluate(rows)
