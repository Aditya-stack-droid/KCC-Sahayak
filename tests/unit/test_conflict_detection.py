"""Focused tests for Batch B Improvement 1 — answer conflict detection.

These tests verify rule behaviour and the escalation wrapper.  The synthetic
pair metrics are produced by ``evaluate.py``; nothing here is presented as a
real-world safety or correctness result.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from confidence_gate import decide  # noqa: E402
from experiments.conflict_detection import evaluate as evaluate_module  # noqa: E402
from experiments.conflict_detection.conflicts import (  # noqa: E402
    DOSE_GAP_TOLERANCE,
    detect_many,
    detect_pair,
)
from experiments.conflict_detection.gate import decide_with_conflicts  # noqa: E402
from experiments.conflict_detection.products import (  # noqa: E402
    find_products,
    negated_products,
)
from experiments.conflict_detection.units import (  # noqa: E402
    extract_intervals,
    extract_quantities,
    is_dose,
)

SAFE_ANSWER = "contact the district agriculture officer; the variety is available at the seed store"


def kinds(answer_a: str, answer_b: str) -> set[str]:
    return {conflict.kind for conflict in detect_pair(answer_a, answer_b)}


# --------------------------------------------------------------------------- #
# Quantity and unit extraction
# --------------------------------------------------------------------------- #
def test_mass_and_volume_units_are_normalised():
    quantity = extract_quantities("Apply 2 kg urea per acre.")[0]
    assert quantity.value_low == 2000 and quantity.value_high == 2000
    assert quantity.basis == "acre" and quantity.basis_kind == "area"
    assert extract_quantities("Apply 1 quintal urea.")[0].value_low == 100_000
    assert extract_quantities("Spray 1 litre water.")[0].value_low == 1000


def test_hectare_is_converted_to_acre():
    quantity = extract_quantities("Apply 50 kg urea per hectare.")[0]
    assert quantity.basis == "acre"
    assert quantity.per_amount == pytest.approx(2.471053814671653)
    low, high = quantity.normalised_bounds()
    assert low == pytest.approx(50_000 / 2.471053814671653)


def test_per_n_litre_basis_is_normalised():
    quantity = extract_quantities("Spray 2 ml per 10 litre water.")[0]
    assert quantity.basis == "litre" and quantity.per_amount == 10
    low, high = quantity.normalised_bounds()
    assert low == pytest.approx(0.2)


def test_in_n_litre_water_is_a_dilution_basis():
    quantity = extract_quantities("Dilute 1 ml imidacloprid in 2 litre water.")[0]
    assert quantity.basis == "litre" and quantity.per_amount == 2
    assert quantity.normalised_bounds()[0] == pytest.approx(0.5)


def test_ranges_devanagari_and_percent():
    quantity = extract_quantities("Apply 2 - 3 ml neem oil.")[0]
    assert (quantity.value_low, quantity.value_high) == (2, 3)
    hindi = extract_quantities("२ किलो यूरिया डालें।")[0]
    assert hindi.value_low == 2000
    percent = extract_quantities("Spray 2% urea solution.")[0]
    assert percent.dimension == "concentration" and percent.value_low == 2


def test_prices_plots_and_pack_sizes_are_not_quantities():
    for text in (
        "Urea price is Rs 266 per bag.",
        "Available in 500 gm pack.",
        "Plot number 500 gm was sampled.",
        "Token 25 kg was issued.",
    ):
        assert extract_quantities(text) == (), text


def test_dose_filter_excludes_spoons():
    assert is_dose(extract_quantities("Apply 2 kg urea.")[0])
    assert not is_dose(extract_quantities("Add 2 teaspoons honey.")[0])


def test_intervals_are_extracted():
    assert extract_intervals("Spray after 5 days.")[0].low == 5
    assert extract_intervals("Spray at an interval of 15 days.")[0].high == 15
    assert extract_intervals("Repeat at 10 day interval.")[0].low == 10


# --------------------------------------------------------------------------- #
# Products and negation
# --------------------------------------------------------------------------- #
def test_product_aliases_are_normalised_without_brand_equivalence():
    assert find_products("Apply D.A.P. and MOP.") == {"dap", "mop"}
    assert find_products("Apply potash.") == {"mop"}
    assert find_products("Spray 2,4-D.") == {"2,4-d"}
    assert find_products("Spray neem oil.") == {"neem"}
    assert find_products("Apply rogor.") != find_products("Apply dimethoate.")


def test_negation_is_clause_local():
    assert negated_products("Do not apply urea; instead spray neem oil.") == {"urea"}
    assert negated_products("Avoid chlorpyrifos.") == {"chlorpyrifos"}
    assert negated_products("Apply urea at sowing.") == set()


# --------------------------------------------------------------------------- #
# Pair rules
# --------------------------------------------------------------------------- #
def test_dosage_conflict_is_detected_with_high_severity():
    conflicts = detect_pair("Apply urea 50 kg per acre.", "Apply urea 20 kg per acre.")
    assert [conflict.kind for conflict in conflicts] == ["dosage"]
    assert conflicts[0].severity == "high"


def test_dosage_difference_within_tolerance_is_not_a_conflict():
    assert kinds("Apply urea 50 kg per acre.", "Apply urea 55 kg per acre.") == set()
    assert kinds("Apply urea 2-3 kg per acre.", "Apply urea 3 kg per acre.") == set()
    assert kinds("Apply urea 50 kg per acre.", "Apply urea 75 kg per acre.") == set()


def test_format_only_differences_are_not_conflicts():
    assert kinds("Apply 50 kg urea per acre.", "Apply urea 50 kg per acre.") == set()
    assert kinds("Apply 500 gm urea per plant.", "Apply 0.5 kg urea per plant.") == set()
    assert kinds("Apply 1 kg urea per 2 acres.", "Apply 0.5 kg urea per acre.") == set()


def test_dilution_strength_is_compared():
    assert kinds("Dilute 1 ml imidacloprid in 1 litre water.",
                 "Dilute 1 ml imidacloprid in 2 litre water.") == {"dosage"}
    assert kinds("Spray 2 ml chlorpyrifos per litre.",
                 "Spray 2 ml chlorpyrifos per 10 litre.") == {"dosage"}


def test_product_conflict_suppresses_cross_product_dosage_comparison():
    assert kinds("Spray rogor 2 ml per litre.", "Spray mancozeb 5 gm per litre.") == {"product"}
    assert kinds("Apply 50 kg urea per acre.", "Apply 50 kg DAP per acre.") == {"product"}


def test_basis_and_dimension_conflicts():
    assert kinds("Apply 50 kg urea per acre.", "Apply 50 kg urea per bigha.") == {"basis"}
    assert kinds("Apply 2 kg urea per plant.", "Apply 2 kg urea per acre.") == {"basis"}
    assert kinds("Apply 5 kg urea.", "Apply 5 litre urea.") == {"dimension"}
    assert kinds("Spray 10 gm sulphur per litre.", "Spray 10 ml sulphur per litre.") == {"dimension"}


def test_hectare_vs_acre_is_compared_after_conversion():
    assert kinds("Apply 50 kg urea per hectare.", "Apply 50 kg urea per acre.") == {"dosage"}


def test_interval_conflicts():
    assert kinds("Spray after 5 days.", "Spray after 15 days.") == {"interval"}
    assert kinds("Repeat after 10 days.", "Repeat after 12 days.") == set()


def test_polarity_conflicts_and_scope():
    assert kinds("Do not spray chlorpyrifos.", "Spray chlorpyrifos 2 ml per litre.") == {"polarity"}
    assert kinds("Avoid urea at sowing.", "Apply urea at sowing.") == {"polarity"}
    assert kinds("Do not spray chlorpyrifos.", "Do not apply chlorpyrifos.") == set()
    assert kinds("Do not apply urea; instead spray neem oil.", "Spray neem oil.") == set()


def test_detection_is_symmetric():
    for answer_a, answer_b in (
        ("Apply urea 50 kg per acre.", "Apply urea 20 kg per acre."),
        ("Apply 50 kg urea per acre.", "Apply 50 kg urea per bigha."),
        ("Do not spray chlorpyrifos.", "Spray chlorpyrifos 2 ml per litre."),
        ("Spray after 5 days.", "Spray after 15 days."),
    ):
        assert kinds(answer_a, answer_b) == kinds(answer_b, answer_a)


def test_empty_and_non_string_inputs_are_safe():
    assert detect_pair("", "Apply urea 50 kg per acre.") == ()
    assert detect_pair(None, "Apply urea.") == ()  # type: ignore[arg-type]
    report = detect_many([None, "", "  "])  # type: ignore[list-item]
    assert report.n_answers == 0 and report.escalate is False


# --------------------------------------------------------------------------- #
# Aggregation and escalation
# --------------------------------------------------------------------------- #
def test_detect_many_escalates_without_producing_an_answer():
    report = detect_many([
        "Apply urea 50 kg per acre.",
        "Apply urea 20 kg per acre.",
        "Contact the agriculture officer.",
    ])
    assert report.escalate is True
    assert report.max_severity == "high"
    assert "ANSWER_CONFLICT" in report.reasons
    assert "DOSAGE_CONFLICT" in report.reasons
    payload = report.as_dict()
    assert "answer" not in payload and "consensus" not in payload
    assert payload["conflicts"]


def test_detect_many_identical_answers_do_not_escalate():
    report = detect_many(["Apply urea 50 kg per acre.", "Apply 50 kg urea per acre."])
    assert report.escalate is False and report.reasons == ()


def test_detect_many_limits_the_number_of_answers():
    answers = [f"Apply urea {value} kg per acre." for value in (10, 20, 30, 40, 50, 60, 70)]
    report = detect_many(answers)
    assert report.n_answers == 5


def test_gate_wrapper_escalates_only_and_preserves_the_base_decision():
    answers = [
        SAFE_ANSWER,
        "Apply 50 kg urea per acre.",
        "Apply 20 kg urea per acre.",
    ]
    base = decide("Varietal Information", 0.95, 0.90, answer=SAFE_ANSWER)
    decision = decide_with_conflicts(answers, "Varietal Information", 0.95, 0.90)
    assert base.action == "AUTO_ANSWER"
    assert decision.base_action == base.action
    assert decision.action == "ESCALATE"
    assert "ANSWER_CONFLICT" in decision.reasons
    assert decision.conflict_report.escalate is True


def test_gate_wrapper_leaves_clean_answers_alone():
    decision = decide_with_conflicts(
        [SAFE_ANSWER, "Contact the KVK for soil testing."],
        "Varietal Information", 0.95, 0.90,
    )
    assert decision.action == "AUTO_ANSWER"
    assert decision.base_action == "AUTO_ANSWER"
    assert decision.reasons == ()


def test_gate_wrapper_never_downgrades_a_high_stakes_escalation():
    decision = decide_with_conflicts(
        [SAFE_ANSWER, "Apply 50 kg urea per acre.", "Apply 20 kg urea per acre."],
        "Plant Protection", 0.95, 0.90,
    )
    assert decision.base_action == "ESCALATE"
    assert decision.base_code == "high_stakes"
    assert decision.action == "ESCALATE"
    assert "ANSWER_CONFLICT" in decision.reasons


def test_dose_tolerance_is_explicit():
    assert DOSE_GAP_TOLERANCE == 0.5
    # Exactly at the tolerance is a conflict; just below it is not.
    assert kinds("Apply urea 50 kg per acre.", "Apply urea 100 kg per acre.") == {"dosage"}
    assert kinds("Apply urea 50 kg per acre.", "Apply urea 74 kg per acre.") == set()


# --------------------------------------------------------------------------- #
# Authored dataset and measured evaluation
# --------------------------------------------------------------------------- #
def test_authored_pair_dataset_is_synthetic_and_labelled():
    path = ROOT / "experiments" / "conflict_detection" / "data" / "conflict_pairs.csv"
    assert path.exists(), "run build_pairs.py first"
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) >= 60
    assert all(row["synthetic"] == "true" for row in rows)
    assert {row["subset"] for row in rows} == {"core", "challenge"}
    assert sum(1 for row in rows if row["subset"] == "challenge") >= 8
    allowed = {"dosage", "product", "basis", "dimension", "interval", "polarity"}
    for row in rows:
        assert set(row["gold_kinds"].split(";")) - {""} <= allowed
        assert row["gold_conflict"] == ("true" if row["gold_kinds"] else "false")


def test_evaluation_metrics_have_valid_structure():
    rows = evaluate_module.load_pairs()
    metrics, predictions = evaluate_module.evaluate(rows)
    binary = metrics["binary"]
    assert binary["tp"] + binary["fp"] + binary["fn"] + binary["tn"] == metrics["n_pairs"]
    assert 0.0 <= binary["precision"] <= 1.0
    assert 0.0 <= binary["recall"] <= 1.0
    assert 0.0 <= binary["f1"] <= 1.0
    assert metrics["n_gold_conflict"] + metrics["n_gold_no_conflict"] == metrics["n_pairs"]
    assert len(predictions) == metrics["n_pairs"]
    assert set(metrics["subsets"]) == {"core", "challenge"}
    assert metrics["subsets"]["core"]["n"] + metrics["subsets"]["challenge"]["n"] == metrics["n_pairs"]
    for kind in ("dosage", "product", "polarity", "basis", "interval"):
        assert metrics["per_kind"][kind]["support"] > 0
