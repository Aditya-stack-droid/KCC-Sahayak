import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from answer_quality import is_informative  # noqa: E402
from confidence_gate import GateThresholds, answer_risk, decide  # noqa: E402

SAFE_ANSWER = "contact the district agriculture officer; the variety is available at the seed store"


def test_high_stakes_always_escalates_even_at_full_confidence():
    d = decide("Plant Protection", 1.0, 1.0, answer=SAFE_ANSWER)
    assert d.action == "ESCALATE" and d.code == "high_stakes"


def test_all_high_stakes_types_escalate():
    for t in ["Plant Protection", "Nutrient Management", "Government Scheme"]:
        assert decide(t, 0.99, 0.99, answer=SAFE_ANSWER).action == "ESCALATE"


def test_time_sensitive_types_escalate_because_logged_answers_are_stale():
    for t in ["Weather", "Market Price"]:
        d = decide(t, 0.99, 0.99, answer=SAFE_ANSWER)
        assert d.action == "ESCALATE" and d.code == "time_sensitive"


def test_time_sensitive_rule_can_be_disabled_for_ablation():
    th = GateThresholds(block_time_sensitive=False)
    assert decide("Weather", 0.99, 0.99, th, answer=SAFE_ANSWER).action == "AUTO_ANSWER"


def test_low_intent_confidence_escalates():
    d = decide("Cultural Practices", 0.40, 0.95, answer=SAFE_ANSWER)
    assert d.action == "ESCALATE" and d.code == "low_intent"


def test_low_retrieval_similarity_escalates():
    d = decide("Cultural Practices", 0.95, 0.20, answer=SAFE_ANSWER)
    assert d.action == "ESCALATE" and d.code == "low_retrieval"


def test_clean_static_knowledge_query_is_auto_answered():
    d = decide("Varietal Information", 0.95, 0.90, answer=SAFE_ANSWER)
    assert d.action == "AUTO_ANSWER" and d.code == "ok"


def test_risky_answer_escalates_even_for_safe_category():
    # the failure seen in the first demo: a Varietal Information query retrieved
    # an answer containing a fertiliser dose.
    ans = "varieties- punjab chhuhara, pusa ruby. fertilizer dose- 24kg urea, 50kg ssp per bigha"
    d = decide("Varietal Information", 0.95, 0.90, answer=ans)
    assert d.action == "ESCALATE" and d.code == "risky_answer"


def test_call_log_note_is_not_sent_as_an_answer():
    # the failure seen in the second demo: "unable to resolve" was auto-sent
    for note in ["unable to resolve", "explained in details", "transfered to expert"]:
        d = decide("Varietal Information", 0.95, 0.90, answer=note)
        assert d.action == "ESCALATE" and d.code == "uninformative_answer", note


def test_answer_screening_can_be_disabled():
    th = GateThresholds(screen_answer=False)
    assert decide("Varietal Information", 0.95, 0.90, th, answer="apply 24kg urea per bigha").action == "AUTO_ANSWER"


def test_thresholds_are_respected():
    th = GateThresholds(intent_confidence=0.9, retrieval_similarity=0.9)
    assert decide("Cultural Practices", 0.85, 0.95, th, answer=SAFE_ANSWER).action == "ESCALATE"
    assert decide("Cultural Practices", 0.95, 0.95, th, answer=SAFE_ANSWER).action == "AUTO_ANSWER"


def test_answer_risk_detects_doses_and_chemicals():
    assert answer_risk("spray rogor@2ml/lit.at evening time")
    assert answer_risk("apply 50 kg ssp")
    assert answer_risk("use copper oxychloride")
    assert answer_risk("apply urea in two splits")


def test_answer_risk_passes_plain_information():
    assert answer_risk("contact the district agriculture officer for details") is None
    assert answer_risk("") is None
    assert answer_risk(None) is None


# ---- informative-answer filter
def test_call_log_notes_are_not_informative():
    for note in ["explain in details", "explian in details", "given him in detail.", "replied accordingly",
                 "trasfer to agri. expert.", "forward to expert for further information",
                 "expert not available", "unable to resolve", "forwarded", "ok"]:
        assert not is_informative(note), note


def test_terse_real_advice_is_informative():
    for adv in ["spray maltahion", "use variety ts-67", "told him to visit kvk",
                "suggested him to spray rogor @ 2ml / l of water", "visit district agriculture office"]:
        assert is_informative(adv), adv


def test_generic_reference_answers_are_not_informative():
    for note in ["given in details about the package of practices",
                 "advised regarding different managerial aspects of pond preparation.",
                 "explained in details as per package & practice"]:
        assert not is_informative(note), note


def test_specific_advice_that_mentions_package_of_practices_is_kept():
    assert is_informative("apply 14 kg urea per bigha as per package of practices")


def test_answer_quality_source_has_no_control_characters():
    src = (Path(__file__).resolve().parents[1] / "src" / "answer_quality.py").read_text(encoding="utf-8")
    assert not [c for c in src if ord(c) < 32 and c not in "\n\t\r"]


def test_topic_mismatch_between_query_and_retrieved_question_escalates():
    d = decide("Animal Husbandry", 0.95, 0.9, answer=SAFE_ANSWER, retrieved_type="Government Scheme")
    assert d.action == "ESCALATE" and d.code == "topic_mismatch"


def test_matching_topic_is_answered():
    d = decide("Varietal Information", 0.95, 0.9, answer=SAFE_ANSWER, retrieved_type="Varietal Information")
    assert d.action == "AUTO_ANSWER"


def test_topic_check_can_be_disabled():
    th = GateThresholds(require_topic_match=False)
    d = decide("Varietal Information", 0.95, 0.9, th, answer=SAFE_ANSWER, retrieved_type="Weather")
    assert d.action == "AUTO_ANSWER"


def test_single_word_transfer_note_is_not_informative():
    assert not is_informative("transfer")
