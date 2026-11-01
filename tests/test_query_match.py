import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from query_match import coverage_problem  # noqa: E402


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
