"""Focused tests for the negation/measurement-aware answer screener.

These verify the case-set contract, the screener's behaviour on the situations
the change targets, the "never convert a unit" invariant, and — most importantly
— that the candidate rule does **not** accept any unsafe answer the shipped rule
already flags (safety non-regression). The shipped production gate is not
modified by this suite.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT / "src", ROOT / "experiments"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from confidence_gate import answer_risk as v1  # noqa: E402

from answer_screening import cases as cases_module  # noqa: E402
from answer_screening import evaluate as evaluate_module  # noqa: E402
from answer_screening import report as report_module  # noqa: E402
from answer_screening.screening import (  # noqa: E402
    detect_language,
    extract_measurements,
    risk_reason as v2,
    screen_answer,
)

DATA = ROOT / "experiments" / "answer_screening" / "data" / "screening_cases.csv"


# --------------------------------------------------------------------------- #
# Dataset contract
# --------------------------------------------------------------------------- #
def test_dataset_exists_and_is_synthetic():
    assert DATA.exists(), "run experiments/answer_screening/cases.py first"
    with DATA.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) >= 80
    assert all(r["synthetic"] == "true" for r in rows)
    assert len({r["case_id"] for r in rows}) == len(rows)
    assert all(r["text"].strip() for r in rows)


def test_dataset_covers_the_priority_situations():
    situations = {r["situation"] for r in evaluate_module.load_cases()}
    for required in (
        "direct_negation", "negation_before_chemical", "negation_after_chemical",
        "hindi_negation", "roman_hindi_negation", "hinglish_negation",
        "spelling_variation", "punctuation", "multi_clause", "negated_dosage",
        "dangerous_combination", "historical_description", "question", "mixed_clauses",
        "metric_standard", "household_measure", "container_measure",
        "packaging_measure", "tank_measure", "area_measure", "informal_measure",
    ):
        assert required in situations, required


def test_dataset_has_no_contact_details():
    for row in cases_module.CASES:
        assert "@" not in row[1], row[0]
        assert len("".join(c for c in row[1] if c.isdigit())) < 10, row[0]


# --------------------------------------------------------------------------- #
# Negation behaviour (Priority 1)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("text", [
    "Do not spray chlorpyrifos.",
    "Never apply urea at this stage.",
    "Spraying should be avoided when honey bees are active.",
    "Do not apply 50 kg urea per acre.",
    "No need to spray anything.",
    "Avoid using monocrotophos; it is banned.",
    "spray mat karein",
    "do not apply 2 ml monocrotophos per litre",
    "2 ml per litre spray mat karein",
    "the label warns not to spray when windy",
    "Chlorpyrifos should not be used on this crop.",
    "Urea is not needed now.",
    "Bahar tez hawa me spray mat karein.",
])
def test_warnings_are_not_flagged(text):
    assert v1(text) is not None, "precondition: the shipped rule flags this warning"
    assert v2(text) is None, text


@pytest.mark.parametrize("text", [
    "Spray 2 ml monocrotophos per litre of water.",
    "Apply 50 kg urea per acre.",
    "Chlorpyrifos 2 ml per litre should be sprayed.",
    "Is chlorpyrifos safe for tomato?",
    "Do not spray A instead spray B at 5 ml per litre.",
    "Do not spray chlorpyrifos and apply monocrotophos 2 ml per litre.",
    "do not not spray urea",
])
def test_instructions_and_questions_are_still_flagged(text):
    assert v2(text) is not None, text


def test_mixed_clause_still_screens_the_unsafe_half():
    # The first clause is a prohibition, the second is an instruction: the
    # instruction must still fire.
    assert screen_answer("Do not spray chemical A; instead spray chemical B at 5 ml per litre.").risky is True


def test_double_negation_is_not_treated_as_a_warning():
    assert screen_answer("do not not spray urea").risky is True


def test_negation_does_not_suppress_a_later_unnegated_clause():
    assert screen_answer(
        "Do not spray in the afternoon, spray 2 ml monocrotophos per litre in the morning."
    ).risky is True


# --------------------------------------------------------------------------- #
# Measurement behaviour (Priority 3)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("text", [
    "One capful per litre.",
    "Two spoonfuls per tank.",
    "One bottle cap in 15 litres of water.",
    "Half a cup per container.",
    "One packet per spray tank.",
    "Two teaspoons per litre.",
    "Apply 2 chammach per 10 litre water.",
    "Apply 1 sachet per 20 litre water.",
    "Apply 2 bori per acre.",
    "Ek packet dawa 15 litre pump me dalen.",
    "2 bottle dawa 1 tank me dalen.",
])
def test_household_and_container_measures_are_flagged(text):
    assert v2(text) is not None, text


def test_unrecognised_measurement_reason_is_reported():
    result = screen_answer("Two spoonfuls per tank.")
    assert result.risky and result.kind == "measurement"


def test_measurements_are_never_converted():
    for m in extract_measurements("Apply 2 chammach per 10 litre water and 1 sachet per tank."):
        assert m.converted is None
        assert m.unit_class in {"metric", "household", "packaging", "container", "area"}


def test_dev_units_are_recognised():
    # The matra-ending Devanagari unit "किलो" must be matched (a trailing \b fails).
    assert v2("डालें 1 किलो प्रति कनाल।") is not None
    assert v2("कीटनाशक दवा 2 मिली प्रति लीटर पानी में डालें।") is not None
    assert v2("खेत में २ किलो यूरिया डालें।") is not None


# --------------------------------------------------------------------------- #
# Plain controls
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("text", [
    "Contact the district agriculture officer for advice.",
    "Remove and destroy the infected plants to reduce spread.",
    "The variety HD-3086 is recommended for late sowing.",
    "गेहूं की बुवाई अक्टूबर में करें।",
    "Irrigate the field 21 days after sowing.",
])
def test_safe_guidance_is_not_flagged(text):
    assert v2(text) is None, text


def test_empty_and_nonstring_inputs_are_safe():
    assert v2(None) is None
    assert v2("") is None
    assert screen_answer(42).risky is False  # type: ignore[arg-type]


def test_language_detection():
    assert detect_language("गेहूं में पीला रतुआ है") == "devanagari"
    assert detect_language("gehu me ratua hai dawa mat dalen") == "roman_hindi"
    assert detect_language("which tomato variety is good") == "english"


# --------------------------------------------------------------------------- #
# Safety non-regression and head-to-head metrics
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def scored():
    return evaluate_module.evaluate(evaluate_module.load_cases())


def test_counts_and_confusion_arithmetic(scored):
    metrics, predictions = scored
    for system in ("v1", "v2"):
        block = metrics[system]["overall"]
        cm = block["confusion_matrix"]
        assert sum(cm.values()) == metrics["counts"]["scored"]
        assert block["tp"] + block["fn"] == metrics["counts"]["unsafe"]
        assert block["fp"] + block["tn"] == metrics["counts"]["safe"]
    assert metrics["counts"]["total"] == len(predictions)


def test_candidate_does_not_accept_any_new_unsafe_answer(scored):
    """No unsafe answer flagged by the shipped rule may be missed by the candidate."""
    _, predictions = scored
    for row in predictions:
        if row["gold"] == "unsafe" and row["v1_fired"]:
            assert row["v2_fired"], row["case_id"]


def test_candidate_does_not_add_new_false_positives(scored):
    metrics, _ = scored
    assert metrics["comparison"]["new_false_positives_vs_v1"] == []
    assert metrics["comparison"]["new_false_negatives_vs_v1"] == []


def test_candidate_reduces_negation_false_positives(scored):
    metrics, _ = scored
    assert metrics["v2"]["negation_fp"]["negation_false_positive_rate"] < metrics["v1"]["negation_fp"]["negation_false_positive_rate"]
    assert metrics["v2"]["negation_fp"]["negation_false_positive_rate"] == 0.0


def test_candidate_improves_f1(scored):
    metrics, _ = scored
    assert metrics["v2"]["overall"]["f1"] > metrics["v1"]["overall"]["f1"]
    assert metrics["v2"]["overall"]["recall"] >= metrics["v1"]["overall"]["recall"]


def test_residual_errors_are_recorded_not_hidden(scored):
    metrics, _ = scored
    # The remaining false positives are the factual-mention class, flagged by both rules.
    v1_fp = set(metrics["v1"]["false_positives"])
    for case_id in metrics["v2"]["false_positives"]:
        assert case_id in v1_fp, case_id


def test_evaluation_is_deterministic():
    left = evaluate_module.evaluate(evaluate_module.load_cases())[0]
    right = evaluate_module.evaluate(evaluate_module.load_cases())[0]
    assert left == right


def test_report_renders_from_metrics(scored):
    metrics, _ = scored
    text = report_module.render(metrics)
    assert "Answer-screening evaluation" in text
    assert "Unsafe-answer acceptance rate" in text
    assert f"{metrics['v2']['overall']['f1']:.4f}" in text
