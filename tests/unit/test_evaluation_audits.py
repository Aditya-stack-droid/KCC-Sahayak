"""Tests for the leakage, answer-quality and language capability audits."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT / "src", ROOT / "experiments"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from answer_quality_eval import framework as aq  # noqa: E402
from leakage_audit import audit as leakage  # noqa: E402
from language_eval import run as lang  # noqa: E402


# ---------------------------------------------------------------- leakage
def test_leakage_audit_values():
    m = leakage.audit()
    assert m["sizes"]["train"] + m["sizes"]["val"] == m["sizes"]["index"]
    e = m["exact_question_leakage"]
    assert e["test_rows"] == 13888
    assert e["test_question_in_train"] == 6247
    assert 0.4 < e["test_question_in_index_share"] < 0.5
    assert m["record_leakage"]["share"] == 0.0
    assert m["grouped_reference"]["near_duplicate_share_ge_0.95"] is not None


def test_leakage_report_renders():
    text = leakage.render(leakage.audit())
    assert "Dataset leakage audit" in text
    assert "Grouped split" in text


# ---------------------------------------------------------------- answer quality
def test_answer_quality_dimensions_and_unavailable_marked():
    r = aq.build()
    dims = r["dimensions"]
    for name in ("retrieval_relevance", "answer_completeness", "semantic_similarity",
                 "factual_consistency", "crop_entity_consistency", "dosage_consistency",
                 "unit_consistency", "safety_compliance", "escalation_correctness",
                 "unsupported_claim_rate", "agronomic_correctness"):
        assert name in dims, name
    assert dims["agronomic_correctness"]["status"] == "unavailable"
    assert dims["retrieval_relevance"]["status"] == "measured"
    # no fabricated correctness score
    assert "score" not in dims["agronomic_correctness"]


def test_answer_quality_headline_and_render():
    r = aq.build()
    assert r["headline"]["agronomic_correctness"] == "unavailable"
    assert 0.0 <= r["headline"]["answer_own_token_f1"] <= 1.0
    text = aq.render(r)
    assert "Answer-quality evaluation framework" in text
    assert "unavailable" in text


# ---------------------------------------------------------------- language
def test_language_capability_findings():
    m = lang.build()
    reg = m["register_and_entities"]
    assert reg["en"]["register_detection_accuracy"] == 1.0
    assert reg["hi"]["register_detection_accuracy"] == 1.0
    # English-only gazetteer cannot tag Devanagari entities
    assert reg["hi"]["entity_coverage"] == 0.0
    assert reg["hinglish"]["entity_coverage"] < reg["en"]["entity_coverage"]


def test_language_dosage_recall_candidate_dominates():
    m = lang.build()
    d = m["dosage_detection_recall"]
    assert d["devanagari"]["v1_recall"] == 0.0
    assert d["devanagari"]["v2_recall"] == 1.0
    for lang_block in d.values():
        assert lang_block["v2_recall"] >= lang_block["v1_recall"]


def test_language_report_renders():
    text = lang.render(lang.build())
    assert "Language capability report" in text
    assert "entity coverage" in text
