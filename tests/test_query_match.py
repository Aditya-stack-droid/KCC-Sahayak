import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from query_match import coverage_problem, temporal_problem  # noqa: E402


def test_crop_named_by_farmer_but_missing_from_match_is_flagged():
    r = coverage_problem("whats better tomato or cucumber", "what are the different types of tomatoes", "pusa ruby, arka abha")
    assert r and "cucumber" in r


def test_comparison_question_needs_a_comparison_match():
    r = coverage_problem("which is better, hybrid or desi wheat", "asking about wheat varieties for late sowing", "huw-468")
    assert r and "comparison" in r


def test_ordinary_matching_question_passes():
    assert coverage_problem("which tomato variety is good for me?", "asking about good varieties of tomato", "pusa early") is None


def test_crop_found_in_the_answer_counts():
    assert coverage_problem("sowing time of cotton", "asking about sowing time", "sowing time of cotton is april to june") is None


def test_no_crops_and_no_comparison_never_flags():
    assert coverage_problem("how to start a poultry farm", "asking about poultry farm loan", "contact bank") is None


@pytest.mark.parametrize("q", [
    "Is it going to rain in the next three days? I want to decide whether to harvest my wheat now.",
    "will it rain tomorrow",
    "agle teen din mein baarish hogi kya",
    "aaj tomato ka bhav kya hai",
    "what is the price of onion",
    "आज मंडी में सरसों का भाव क्या चल रहा है?",
    "wheat mandi rate",
])
def test_live_or_price_questions_are_flagged(q):
    assert temporal_problem(q)


@pytest.mark.parametrize("q", [
    "what is the seed rate for paddy",
    "what is the sowing time for mustard",
    "which is the best variety of wheat for sowing in november",
    "how to control aphids in mustard",
])
def test_ordinary_questions_are_not_flagged(q):
    assert temporal_problem(q) is None


def test_gate_escalates_temporal_even_for_a_safe_topic():
    from confidence_gate import decide
    d = decide("Cultural Practices", 0.99, 0.99, answer="sow in november", temporal_problem="live weather")
    assert d.action == "ESCALATE" and d.code == "time_sensitive"
