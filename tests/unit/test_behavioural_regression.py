"""Tests for the Improvement 5 behavioural regression experiment.

The suite checks corpus integrity, deterministic execution, the measured
baseline, known capability gaps, output serialisation and report rendering. It
does not assert that the production components are correct; the authored cases
record expected-versus-observed behaviour and the gaps are listed in
``results/behavioural_regression/REPORT.md``.
"""
import csv
import json
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = ROOT / "experiments" / "behavioural_regression"
for path in (ROOT, EXPERIMENT, ROOT / "experiments" / "out_of_scope"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import br_cases  # noqa: E402
import br_probes  # noqa: E402
import br_report  # noqa: E402
import run_evaluation  # noqa: E402


@pytest.fixture(scope="module")
def records():
    return br_probes.evaluate_all(br_cases.CASES)


@pytest.fixture(scope="module")
def metrics(records):
    return br_report.summarize(records, br_cases.CASES)


def by_id(records):
    return {record["case_id"]: record for record in records}


def test_case_inventory_is_complete_and_unique():
    cases = br_cases.CASES
    assert len(cases) == 168
    assert len({case["id"] for case in cases}) == len(cases)
    assert {case["category"] for case in cases} == set(br_report.CATEGORIES)
    assert all(case.get("probe") for case in cases)
    assert all("expected" in case or "expected_nonempty" in case for case in cases)
    counts = {code: 0 for code in br_report.CATEGORIES}
    for case in cases:
        counts[case["category"]] += 1
    assert counts == {"A": 24, "B": 24, "C": 18, "D": 23, "E": 9, "F": 18, "G": 27, "H": 25}


def test_case_digest_is_stable_and_matches_metrics(metrics):
    digest = br_report.canonical_hash(br_cases.CASES)
    assert len(digest) == 64
    assert digest == br_report.canonical_hash(br_cases.CASES)
    assert digest == metrics["case_sha256"]


def test_evaluation_is_deterministic(records):
    again = br_probes.evaluate_all(br_cases.CASES)
    assert again == records
    assert br_report.summarize(again, br_cases.CASES) == br_report.summarize(records, br_cases.CASES)


def test_measured_baseline_counts(metrics):
    assert metrics["n_cases"] == 168
    assert metrics["passed"] == 109
    assert metrics["failed"] == 59
    assert metrics["crashes"] == 0
    assert metrics["pass_rate"] == 0.6488
    assert metrics["by_failure_type"] == {
        "degraded": 6,
        "false_negative": 24,
        "false_positive": 10,
        "missing_capability": 18,
        "validation_gap": 1,
    }
    passed_by_category = {
        code: block["passed"] for code, block in metrics["by_category"].items()
    }
    assert passed_by_category == {"A": 12, "B": 13, "C": 8, "D": 16, "E": 9, "F": 1, "G": 26, "H": 24}


def test_known_capability_gaps_are_recorded(records):
    indexed = by_id(records)
    expected = {
        "A03": "false_negative",
        "A18": "false_negative",
        "B09": "false_negative",
        "B23": "missing_capability",
        "C01": "false_positive",
        "C08": "false_positive",
        "D17": "false_negative",
        "D18": "false_negative",
        "F01": "missing_capability",
        "G26": "validation_gap",
        "H25": "false_negative",
    }
    for case_id, failure_type in expected.items():
        assert indexed[case_id]["status"] == "fail"
        assert indexed[case_id]["failure_type"] == failure_type
        assert indexed[case_id]["synthetic"] is True


def test_passing_controls_and_boundaries(records):
    indexed = by_id(records)
    for case_id in ("A02", "A19", "C07", "C09", "C12", "C14", "D21", "D22", "E07", "G27", "H13", "H22", "H24"):
        assert indexed[case_id]["status"] == "pass"
    assert all(record["status"] == "pass" for record in records if record["category"] == "E")
    assert all(record["status"] == "pass" for record in records if record["probe"] == "malformed_call")


def test_empty_and_malformed_inputs_do_not_crash(metrics, records):
    assert metrics["crashes"] == 0
    for record in records:
        assert "raised:" not in str(record["observed"])
        assert record["error"] == ""


def test_multi_intent_capability_is_absent_by_design(records):
    multi = [record for record in records if record["probe"] == "multi_intent"]
    assert len(multi) == 18
    for record in multi:
        if record["case_id"] == "F18":
            assert record["status"] == "pass"
            assert record["observed"] == "single_label_only"
        else:
            assert record["status"] == "fail"
            assert record["failure_type"] == "missing_capability"
            assert record["observed"] == "single_label_only"


def test_api_schema_boundaries_show_whitespace_gap():
    from backend.app import QueryRequest

    assert QueryRequest.model_validate_json('{"text": "ok"}').text == "ok"
    assert len(QueryRequest.model_validate_json('{"text": "' + "x" * 1000 + '"}').text) == 1000
    assert QueryRequest.model_validate_json('{"text": "  "}').text == "  "
    with pytest.raises(ValidationError):
        QueryRequest.model_validate_json('{"text": "x"}')
    with pytest.raises(ValidationError):
        QueryRequest.model_validate_json('{"text": "' + "x" * 1001 + '"}')
    with pytest.raises(ValidationError):
        QueryRequest.model_validate_json('{"text": null}')


def test_outputs_are_deterministic(tmp_path, records, metrics):
    left = tmp_path / "left"
    right = tmp_path / "right"
    run_evaluation.write_outputs(left, records, metrics)
    run_evaluation.write_outputs(right, records, metrics)
    for name in ("metrics.json", "cases.csv", "failures.json", "REPORT.md"):
        assert (left / name).read_bytes() == (right / name).read_bytes()

    with (left / "cases.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == len(br_cases.CASES)
    assert list(rows[0]) == run_evaluation.FIELDS
    assert sum(row["status"] == "fail" for row in rows) == metrics["failed"]

    failures = json.loads((left / "failures.json").read_text(encoding="utf-8"))
    assert failures["n_failures"] == metrics["failed"]
    assert failures["synthetic"] is True
    assert failures["case_sha256"] == metrics["case_sha256"]


def test_report_lists_every_failure_and_disclaims(tmp_path, records, metrics):
    text = br_report.render(metrics, records)
    assert "**Cases:** 168" in text
    assert "**Passed:** 109" in text
    assert "**Failed:** 59" in text
    assert "**Pass rate:** 64.88%" in text
    assert "**Unhandled crashes:** 0" in text
    for code, label in br_report.CATEGORIES.items():
        assert f"| {code} | {label} |" in text
    for record in records:
        if record["status"] == "fail":
            assert f"| {record['case_id']} |" in text
    assert "not a real-world accuracy or safety estimate" in text
    assert "Multi-intent detection is absent" not in text
    assert "multi-intent detection is absent" in text
