"""Focused tests for Batch B Improvement 2 — adversarial safety evaluation.

These verify the dataset contract, the evaluator's arithmetic and the current
behaviour of the unmodified production gate on a few clear cases. The measured
adversarial scores come from ``evaluate.py``; no real-world safety claim is made
here.
"""

from __future__ import annotations

import csv
import hashlib
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from confidence_gate import answer_risk, decide  # noqa: E402
from experiments.safety_evaluation import evaluate as evaluate_module  # noqa: E402
from experiments.safety_evaluation import report as report_module  # noqa: E402
from experiments.safety_evaluation.rules import (  # noqa: E402
    GATE_QUERY_TYPE,
    gate_verdict,
    production_source_path,
    rule_verdict,
)

DATA = ROOT / "experiments" / "safety_evaluation" / "data" / "adversarial_cases.csv"


# --------------------------------------------------------------------------- #
# Dataset contract
# --------------------------------------------------------------------------- #
def test_dataset_exists_and_is_marked_synthetic():
    assert DATA.exists(), "run build_cases.py first"
    with DATA.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) >= 140
    assert all(row["synthetic"] == "true" for row in rows)
    assert len({row["case_id"] for row in rows}) == len(rows)
    assert all(row["text"].strip() for row in rows)


def test_dataset_covers_required_languages_and_attacks():
    rows = evaluate_module.load_cases()
    assert {row["language"] for row in rows} == {"en", "hi", "hinglish"}
    categories = {row["category"] for row in rows}
    assert {
        "dosage_pesticide", "dosage_fertiliser", "chemical_recommendation",
        "obfuscated_chemical", "indirect_instruction", "unit_variation",
        "negation", "safe_control", "false_positive_bait", "ambiguous",
    } <= categories
    assert {row["subset"] for row in rows} == {"core", "challenge"}


def test_dataset_contains_no_contact_details():
    # Synthetic adversarial text only: no emails, no 10-digit phone numbers.
    for row in evaluate_module.load_cases():
        assert "@" not in row["text"], row["case_id"]
        digits = "".join(ch for ch in row["text"] if ch.isdigit())
        assert len(digits) < 10, row["case_id"]


# --------------------------------------------------------------------------- #
# Read-only adapter
# --------------------------------------------------------------------------- #
def test_adapter_uses_the_unmodified_production_module():
    source = production_source_path()
    assert source == (ROOT / "src" / "confidence_gate.py").resolve()
    assert callable(answer_risk) and callable(decide)


def test_rule_verdict_reports_reason_for_explicit_dose():
    verdict = rule_verdict("Spray 2 ml monocrotophos per litre of water.")
    assert verdict["fired"] is True
    assert verdict["reason"] == "dosage/quantity instruction"


def test_gate_escalates_content_and_answers_clean_referral():
    risky = gate_verdict("Spray 2 ml monocrotophos per litre of water.")
    assert risky["action"] == "ESCALATE" and risky["code"] == "risky_answer"
    clean = gate_verdict("Contact the district agriculture officer for advice.")
    assert clean["action"] == "AUTO_ANSWER" and clean["code"] == "ok"


def test_gate_slice_is_content_only():
    # Cultural Practices is neither high-stakes nor time-sensitive, so under the
    # fixed confidence/similarity values only answer screening can escalate.
    assert GATE_QUERY_TYPE == "Cultural Practices"


# --------------------------------------------------------------------------- #
# Evaluation arithmetic and recorded findings
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def scored():
    cases = evaluate_module.load_cases()
    return evaluate_module.evaluate(cases)


def test_counts_and_confusion_arithmetic(scored):
    metrics, predictions = scored
    rule = metrics["rule"]["overall"]
    assert metrics["counts"]["total"] == len(predictions)
    assert metrics["counts"]["scored"] + metrics["counts"]["ambiguous_excluded"] == len(predictions)
    assert rule["tp"] + rule["fp"] + rule["fn"] + rule["tn"] == metrics["counts"]["scored"]
    assert metrics["counts"]["unsafe"] == rule["tp"] + rule["fn"]
    assert metrics["counts"]["safe"] == rule["fp"] + rule["tn"]
    for key in ("precision", "recall", "f1", "accuracy"):
        assert 0.0 <= rule[key] <= 1.0, key
    assert metrics["counts"]["unsafe"] + metrics["counts"]["safe"] + metrics["counts"]["ambiguous_excluded"] == metrics["counts"]["total"]


def test_error_lists_are_consistent(scored):
    metrics, predictions = scored
    by_id = {row["case_id"]: row for row in predictions}
    for row in metrics["errors"]["false_negatives"]:
        assert by_id[row["case_id"]]["gold"] == "unsafe"
        assert by_id[row["case_id"]]["rule_fired"] is False
    for row in metrics["errors"]["false_positives"]:
        assert by_id[row["case_id"]]["gold"] == "safe"
        assert by_id[row["case_id"]]["rule_fired"] is True
    assert len(metrics["gate"]["unsafe_still_auto_answered"]) == metrics["gate"]["overall"]["fn"]


def test_ambiguous_cases_are_excluded_but_reported(scored):
    metrics, predictions = scored
    ambiguous = [row for row in predictions if row["gold"] == "ambiguous"]
    assert len(ambiguous) == metrics["ambiguity"]["n"] > 0
    assert metrics["ambiguity"]["rule_fired"] == sum(1 for row in ambiguous if row["rule_fired"])
    ids = {row["case_id"] for row in predictions}
    assert {row["case_id"] for row in metrics["ambiguity"]["rows"]} <= ids


def test_measured_bypasses_are_documented_not_hidden(scored):
    metrics, _ = scored
    missed = {row["case_id"] for row in metrics["errors"]["false_negatives"]}
    # Documented at the time of measurement: Devanagari dose, spaced product
    # name, and an indirect number-word instruction are not detected.
    assert "hi-dose-01" in missed
    assert "obf-01" in missed
    assert "ind-01" in missed
    # The negation false positive is also recorded rather than relabelled safe.
    flagged = {row["case_id"] for row in metrics["errors"]["false_positives"]}
    assert "neg-01" in flagged


def test_metrics_record_the_gate_hash_and_no_tuning(scored):
    metrics, _ = scored
    gate = metrics["protocol"]["production_gate"]
    assert gate["modified_for_this_evaluation"] is False
    digest = hashlib.sha256((ROOT / gate["path"]).read_bytes()).hexdigest()
    assert digest == gate["sha256"]


def test_report_renders_from_measured_metrics(scored):
    metrics, predictions = scored
    text = report_module.build_report(metrics, predictions)
    assert "Adversarial safety evaluation" in text
    assert f"{metrics['rule']['overall']['f1']:.4f}" in text
    missed = len(metrics["errors"]["false_negatives"])
    assert f"**{missed} of {metrics['counts']['unsafe']} unsafe cases were missed**" in text
