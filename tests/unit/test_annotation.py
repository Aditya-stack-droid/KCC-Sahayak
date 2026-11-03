"""Unit tests for the Batch B Phase 2 annotation infrastructure.

These tests never fabricate study results.  Agreement values are asserted only
against small hand-computed worked examples used as software tests; the study
report itself stays PENDING until real annotators submit sheets.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.annotation import agreement, quality, report, schema, ui  # noqa: E402
from experiments.annotation import store as store_mod  # noqa: E402

ITEMS = [
    {
        "item_id": f"kcc-{index:04d}",
        "query": f"pest attack on wheat number {index}",
        "language": "en",
        "candidates": [
            {"candidate_id": "c1", "text": "spray neem oil"},
            {"candidate_id": "c2", "text": "apply urea after sowing"},
            {"candidate_id": "c3", "text": "sell at the mandi"},
        ],
    }
    for index in range(3)
]


def make_key(items=ITEMS):
    return {
        item["item_id"]: {
            "item_id": item["item_id"],
            "stratum": "common",
            "llm_label": "Plant Protection",
            "source": {"dataset": "test", "qid": index, "annotation_id": f"a{index}",
                       "annotation_row": index, "pool": "test"},
            "max_corpus_similarity": 0.5,
            "corpus_group_size": 1,
            "near_duplicate_neighbours": 0,
            "candidates": [{"candidate_id": c["candidate_id"], "provenance": "logged",
                            "truncated": False, "original_length": len(c["text"])}
                           for c in item["candidates"]],
        }
        for index, item in enumerate(items)
    }


def make_store(tmp_path, items=ITEMS):
    store = store_mod.Store(tmp_path / "data")
    store_mod.write_jsonl(store.items_path, items)
    store_mod.write_jsonl(store.key_path, list(make_key(items).values()))
    return store


def rec(item_id, annotator="A001", intent="PP", **overrides):
    base = {
        "annotator": annotator,
        "item_id": item_id,
        "intent": intent,
        "out_of_scope": False,
        "multi_intent": [],
        "safety": "safe",
        "missing_context": ["none"],
        "ambiguous": False,
        "uncertain": False,
        "certainty": 3,
        "notes": "",
        "retrieval_relevance": {"c1": "relevant", "c2": "irrelevant", "c3": "partially_relevant"},
        "answer_quality": {"c1": "adequate", "c2": "inadequate", "c3": "partially_adequate"},
        "answer_safety": {"c1": "safe", "c2": "safe", "c3": "caution"},
    }
    base.update(overrides)
    return base


# --------------------------------------------------------------------------- #
# Schema
# --------------------------------------------------------------------------- #
def test_schema_accepts_a_valid_record_and_normalises_names():
    record = rec("kcc-0000", intent="Plant Protection", missing_context="None")
    assert schema.validate_record(record, ["c1", "c2", "c3"]) == []
    cleaned = schema.normalise_record(record)
    assert cleaned["intent"] == "PP"
    assert cleaned["missing_context"] == ["none"]


def test_schema_rejects_unknown_intent_and_candidate():
    assert any("intent" in error for error in schema.validate_record(
        rec("kcc-0000", intent="XX"), ["c1", "c2", "c3"]))
    assert any("unknown candidate" in error for error in schema.validate_record(
        rec("kcc-0000", retrieval_relevance={"c9": "relevant"}), ["c1", "c2", "c3"]))


def test_schema_keeps_out_of_scope_and_intent_consistent():
    assert schema.validate_record(rec("kcc-0000", intent="OOS", out_of_scope=True),
                                  ["c1", "c2", "c3"]) == []
    assert any("out_of_scope" in error for error in schema.validate_record(
        rec("kcc-0000", intent="OOS", out_of_scope=False), ["c1", "c2", "c3"]))
    assert any("OOS" in error for error in schema.validate_record(
        rec("kcc-0000", intent="PP", out_of_scope=True), ["c1", "c2", "c3"]))


def test_schema_requires_primary_intent_inside_multi_intent():
    errors = schema.validate_record(rec("kcc-0000", multi_intent=["NM"]), ["c1", "c2", "c3"])
    assert any("primary intent" in error for error in errors)
    assert schema.validate_record(rec("kcc-0000", multi_intent=["PP", "NM"]),
                                  ["c1", "c2", "c3"]) == []


def test_schema_rejects_none_combined_with_other_missing_context():
    errors = schema.validate_record(rec("kcc-0000", missing_context=["none", "crop"]),
                                    ["c1", "c2", "c3"])
    assert any("'none'" in error for error in errors)


def test_schema_counts_incomplete_tasks():
    record = rec("kcc-0000")
    record["retrieval_relevance"] = {"c1": "relevant"}
    record["safety"] = ""
    missing = schema.is_complete(record, ["c1", "c2", "c3"])
    assert "safety" in missing
    assert "retrieval_relevance:c2" in missing


# --------------------------------------------------------------------------- #
# Cohen / Fleiss (hand-computed examples)
# --------------------------------------------------------------------------- #
def test_cohen_kappa_known_example_is_0_8():
    # a: 5 A / 5 B, b: 4 A / 6 B, 9 of 10 positions agree.
    a = list("AAAAABBBBB")
    b = list("AAAABBBBBB")
    result = agreement.cohen_kappa(a, b, ["A", "B"])
    assert result["observed_agreement"] == pytest.approx(0.9)
    assert result["expected_agreement"] == pytest.approx(0.5)
    assert result["kappa"] == pytest.approx(0.8)


def test_cohen_kappa_perfect_agreement_and_full_disagreement():
    assert agreement.cohen_kappa(["A", "B"], ["A", "B"], ["A", "B"])["kappa"] == 1.0
    assert agreement.cohen_kappa(["A", "B"], ["B", "A"], ["A", "B"])["kappa"] == -1.0


def test_cohen_kappa_is_undefined_when_expected_agreement_is_one():
    result = agreement.cohen_kappa(["A", "A"], ["A", "A"], ["A", "B"])
    assert result["kappa"] is None
    assert result["kappa_defined"] is False


def test_cohen_kappa_rejects_labels_outside_categories():
    with pytest.raises(ValueError):
        agreement.cohen_kappa(["A"], ["C"], ["A", "B"])


def test_fleiss_kappa_known_example_is_one_third():
    result = agreement.fleiss_kappa([[3, 0], [2, 1], [1, 2], [0, 3]])
    assert result["observed_agreement"] == pytest.approx(2 / 3)
    assert result["expected_agreement"] == pytest.approx(0.5)
    assert result["kappa"] == pytest.approx(1 / 3, abs=1e-6)


def test_fleiss_requires_equal_raters_per_item():
    with pytest.raises(ValueError):
        agreement.fleiss_kappa([[2, 0], [1, 0]])


def test_task_report_excludes_missing_labels_per_pair():
    per_item = {
        "a": {"A001": "PP", "A002": "PP"},
        "b": {"A001": "NM"},
        "c": {"A001": "PP", "A002": "NM"},
    }
    result = agreement.task_report(per_item, schema.INTENT_ORDER)
    pair = result["pairs"]["A001|A002"]
    assert pair["n_common"] == 2
    assert pair["n_missing_left"] == 1
    assert pair["n_missing_right"] == 0
    assert pair["kappa"] == pytest.approx(0.0)
    assert result["status"] == "OK"


def test_task_report_is_pending_with_one_annotator():
    result = agreement.task_report({"a": {"A001": "PP"}}, schema.INTENT_ORDER)
    assert result["status"] == "PENDING"
    assert result["pairs"] == {}


def test_set_report_uses_exact_match_and_jaccard():
    per_item = {
        "a": {"A001": ("PP", "NM"), "A002": ("NM", "PP")},
        "b": {"A001": ("PP",), "A002": ("PP", "NM")},
    }
    result = agreement.set_report(per_item)
    pair = result["pairs"]["A001|A002"]
    assert pair["exact_set_agreement"] == pytest.approx(0.5)
    assert pair["mean_jaccard"] == pytest.approx((1.0 + 1 / 2) / 2)


# --------------------------------------------------------------------------- #
# Store, import/export, blinding, adjudication
# --------------------------------------------------------------------------- #
def test_store_is_append_only_and_tracks_revisions(tmp_path):
    store = make_store(tmp_path)
    store.append_records("A001", [rec("kcc-0000", intent="PP")])
    store.append_records("A001", [rec("kcc-0000", intent="NM")])
    assert len(store.load_records()) == 2
    assert store.latest()[("A001", "kcc-0000")]["intent"] == "NM"
    assert store.revision_counts()[("A001", "kcc-0000")] == 2


def test_export_and_reimport_csv_round_trip(tmp_path):
    store = make_store(tmp_path)
    store.append_records("A001", [rec("kcc-0000"), rec("kcc-0001", intent="NM")])
    exported = store.export_annotator("A001")
    assert Path(exported["csv"]).exists()

    fresh = make_store(tmp_path / "roundtrip")
    result = fresh.import_file("A001", exported["csv"])
    assert result["imported"] == 2 and result["rejected"] == 0
    latest = fresh.latest()
    assert latest[("A001", "kcc-0001")]["intent"] == "NM"
    assert latest[("A001", "kcc-0000")]["retrieval_relevance"]["c2"] == "irrelevant"


def test_import_rejects_the_whole_file_when_a_row_is_invalid(tmp_path):
    store = make_store(tmp_path)
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"records": [rec("kcc-0000"), rec("kcc-0001", intent="ZZ")]}),
                    encoding="utf-8")
    result = store.import_file("A001", path)
    assert result["imported"] == 0 and result["rejected"] == 1
    assert store.load_records() == []
    assert Path(result["reject_report"]).exists()


def test_import_skips_untouched_ui_placeholders(tmp_path):
    store = make_store(tmp_path)
    placeholder = {"annotator": "A001", "item_id": "kcc-0000", "intent": "",
                   "out_of_scope": None, "multi_intent": [], "safety": "",
                   "missing_context": [], "ambiguous": False, "uncertain": False,
                   "certainty": None, "notes": "",
                   "retrieval_relevance": {}, "answer_quality": {}, "answer_safety": {}}
    path = tmp_path / "placeholder.json"
    path.write_text(json.dumps({"records": [placeholder]}), encoding="utf-8")
    result = store.import_file("A001", path)
    assert result["imported"] == 0
    assert result["skipped_empty"] == 1
    assert store.load_records() == []


def test_exports_are_blinded_between_annotators(tmp_path):
    store = make_store(tmp_path)
    store.append_records("A001", [rec("kcc-0000", intent="PP")])
    store.append_records("A002", [rec("kcc-0001", intent="NM")])
    a1 = json.loads(Path(store.export_annotator("A001")["json"]).read_text())
    assert {row["item_id"] for row in a1["records"]} == {"kcc-0000"}
    assert all(row["annotator"] == "A001" for row in a1["records"])


def test_ui_contains_no_key_fields_or_network_urls():
    html = ui.render_ui("A001", ITEMS)
    assert "kcc-0000" in html
    assert "pest attack on wheat number 0" in html
    for secret in ('"llm_label"', '"distractor_same_topic"', '"distractor_other_topic"',
                   '"stratum"', '"provenance"', '"http:"', '"https:"'):
        assert secret not in html


def test_adjudication_requires_a_named_human_and_valid_label(tmp_path):
    store = make_store(tmp_path)
    record = store.append_adjudication("kcc-0000", "intent", "PP", "A003", note="pest first")
    assert record["status"] == "resolved"
    assert store.adjudication_index()[("kcc-0000", "intent", "")]["label"] == "PP"
    with pytest.raises(ValueError):
        store.append_adjudication("kcc-0000", "intent", "ZZ", "A003")
    with pytest.raises(ValueError):
        store.append_adjudication("kcc-0000", "intent", "PP", "")


# --------------------------------------------------------------------------- #
# Agreement orchestration, QC and reports stay PENDING without people
# --------------------------------------------------------------------------- #
def test_agreement_report_is_pending_with_one_annotator(tmp_path):
    store = make_store(tmp_path)
    store.append_records("A001", [rec("kcc-0000")])
    result = agreement.agreement_for(store)
    assert result["status"] == "PENDING"
    for info in result["tasks"].values():
        pairs = info.get("pairs", info.get("aggregate", {}).get("pairs", {}))
        assert pairs == {}


def test_agreement_report_computes_pair_and_human_vs_llm(tmp_path):
    store = make_store(tmp_path)
    store.append_records("A001", [rec("kcc-0000"), rec("kcc-0001")])
    store.append_records("A002", [rec("kcc-0000"), rec("kcc-0001", intent="NM")])
    result = agreement.agreement_for(store)
    assert result["status"] == "OK"
    pair = result["tasks"]["intent"]["pairs"]["A001|A002"]
    assert pair["n_common"] == 2
    assert pair["kappa"] == pytest.approx(0.0)
    assert result["human_vs_llm"]["status"] == "OK"
    assert "LLM" in result["human_vs_llm"]["reference"]


def test_qc_reports_coverage_and_pending_disagreements(tmp_path):
    store = make_store(tmp_path)
    store.append_records("A001", [rec("kcc-0000", intent="PP")])
    store.append_records("A002", [rec("kcc-0000", intent="NM")])
    qc = quality.qc_report(store)
    assert qc["status"] == "IN_PROGRESS"
    assert qc["items_uncovered_by_any_annotator"] == ["kcc-0001", "kcc-0002"]
    assert qc["items_with_two_or_more_annotators"] == 1
    assert qc["adjudication"]["pending_count"] == 1
    assert qc["per_annotator"]["A001"]["coverage"] == pytest.approx(1 / 3, abs=1e-4)


def test_qc_accepts_a_unanimous_item_without_creating_adjudication_work(tmp_path):
    store = make_store(tmp_path)
    store.append_records("A001", [rec("kcc-0000")])
    store.append_records("A002", [rec("kcc-0000")])
    qc = quality.qc_report(store)
    assert qc["adjudication"]["pending_count"] == 0


def test_report_writes_pending_agreement_without_kappa(tmp_path):
    store = make_store(tmp_path)
    out = tmp_path / "results"
    result = report.write_report(store, out, overwrite=True)
    assert Path(result["report"]).exists()
    agreement_json = json.loads((out / "agreement.json").read_text())
    assert agreement_json["status"] == "PENDING"
    for task in agreement_json["tasks"].values():
        pairs = task.get("pairs", task.get("aggregate", {}).get("pairs", {}))
        assert pairs == {}
    text = (out / "REPORT.md").read_text()
    assert "PENDING" in text
    assert "No Cohen's or Fleiss' kappa is reported" in text


def test_report_methods_section_is_not_claimed_as_a_result(tmp_path):
    store = make_store(tmp_path)
    out = tmp_path / "results"
    report.write_report(store, out, overwrite=True)
    text = (out / "QC_REPORT.md").read_text()
    assert "No value in this report is an estimate of human agreement" in text


# --------------------------------------------------------------------------- #
# Committed sample artefacts (frozen by build-sample)
# --------------------------------------------------------------------------- #
def test_committed_sample_has_300_blind_english_items():
    store = store_mod.Store()
    items = store.load_items()
    key = store.load_key()
    assert len(items) == 300
    assert len(key) == 300
    assert len({item["item_id"] for item in items}) == 300
    assert all(item["language"] == "en" for item in items)
    assert all(len(item["candidates"]) == 3 for item in items)
    assert all("stratum" not in item and "llm_label" not in item for item in items)
    assert all(set(c["provenance"] for c in entry["candidates"]) ==
               {"logged", "distractor_same_topic", "distractor_other_topic"} for entry in key.values())
    assert all(isinstance(entry["source"]["qid"], int) for entry in key.values())


def test_committed_sample_covers_the_declared_strata():
    summary = store_mod.Store().load_summary()
    assert summary["realised_n"] == 300
    assert set(summary["realised_strata"]) == {
        "potentially_unsafe", "insufficient_context", "ambiguous", "rare_intent",
        "novel", "near_duplicate", "common",
    }
    assert summary["sample_label_counts"]
