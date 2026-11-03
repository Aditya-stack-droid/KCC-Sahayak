"""Unit tests for the Batch B PII recognizers and redaction helpers.

These tests are additive and write no files.  The evaluation numbers live in
``results/privacy_audit/`` and come from the synthetic dataset, not from these
unit tests.
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.privacy.audit import evaluate, load_rows  # noqa: E402
from experiments.privacy.recognizers import (  # noqa: E402
    detect,
    normalize_digits,
    verhoeff_check_digit,
    verhoeff_valid,
)
from experiments.privacy.redact import RedactionConfig, redact, redact_spans  # noqa: E402


def _aadhaar(base: str = "23456789012") -> str:
    return base + verhoeff_check_digit(base)


A1 = _aadhaar()
A1_FMT = f"{A1[:4]} {A1[4:8]} {A1[8:]}"


def _types(text: str, policy: str = "balanced") -> list[str]:
    return [d.type for d in detect(text, policy=policy)]


def _values(text: str, policy: str = "balanced") -> list[str]:
    return [d.value for d in detect(text, policy=policy)]


# --- checksum ---------------------------------------------------------------
def test_verhoeff_known_vectors_and_generated_identifier():
    assert verhoeff_valid("2363")
    assert verhoeff_valid("999941057058")
    assert not verhoeff_valid("2364")
    assert verhoeff_valid(A1)
    assert len(A1) == 12


# --- phones -----------------------------------------------------------------
def test_indian_mobile_with_context():
    dets = detect("Call 9876543210 for the spray schedule.")
    assert [(d.type, d.value) for d in dets] == [("PHONE", "9876543210")]


def test_mobile_with_country_prefix_and_spacing():
    dets = detect("Contact +91-98765-43210 tomorrow.")
    assert len(dets) == 1
    assert dets[0].type == "PHONE"
    assert dets[0].value == "+91-98765-43210"


def test_bare_mobile_balanced_accepts_but_strict_rejects():
    assert _types("Reach 9876543210.") == ["PHONE"]
    assert "PHONE" not in _types("Reach 9876543210.", policy="strict")


def test_malformed_phone_lengths_and_prefix_are_rejected():
    for text in (
        "The code is 987654321.",
        "The code is 98765432101.",
        "The code is 5876543210.",
    ):
        assert "PHONE" not in _types(text), text


def test_phone_negative_context_is_respected():
    for text in (
        "Plot number 9876543210 is near the canal.",
        "Token 9876543210 was issued.",
        "The sample ID is 9876543210-A.",
        "Rs 9876543210 is the total budget.",
    ):
        assert "PHONE" not in _types(text), text


def test_later_aadhaar_or_email_does_not_suppress_a_phone():
    text = f"My number is 9876543210, my Aadhaar is {A1_FMT}."
    assert "PHONE" in _types(text)
    assert "AADHAAR" in _types(text)


def test_dot_grouped_phone_is_a_documented_limitation():
    assert "PHONE" not in _types("Call 98765.43210 now.")


def test_trunk_zero_phone_is_a_documented_limitation():
    assert "PHONE" not in _types("Call 09876543210 now.")


def test_international_phone():
    dets = detect("Call +1 (415) 555-0132 anytime.")
    assert [(d.type, d.value) for d in dets] == [("PHONE", "+1 (415) 555-0132")]


def test_devanagari_digit_phone():
    dets = detect("मोबाइल ९८७६५४३२१० है")
    assert len(dets) == 1
    assert dets[0].type == "PHONE"
    assert dets[0].value == "९८७६५४३२१०"
    assert normalize_digits(dets[0].value) == "9876543210"


# --- email / URL / bank ------------------------------------------------------
def test_email_and_uppercase_email():
    assert _values("Email rahul.sharma@example.com please.") == ["rahul.sharma@example.com"]
    assert _values("Email RAHUL@EXAMPLE.COM now.") == ["RAHUL@EXAMPLE.COM"]


def test_url_and_ifsc():
    assert _types("Profile https://example.com/farmer/rahul") == ["URL"]
    assert _values("IFSC SBIN0001234 for the transfer.") == ["SBIN0001234"]


def test_bank_account_requires_context_and_is_not_aadhaar():
    assert _types("The number 123456789012 was logged.") == []
    dets = detect("Account number 123456789012 at the bank.")
    assert [d.type for d in dets] == ["BANK_ACCOUNT"]
    assert dets[0].value == "123456789012"


# --- Aadhaar ----------------------------------------------------------------
def test_aadhaar_valid_checksum():
    dets = detect(f"Aadhaar {A1_FMT} is linked to the account.")
    assert [(d.type, d.value) for d in dets] == [("AADHAAR", A1_FMT)]


def test_aadhaar_invalid_without_cue_is_not_detected():
    assert _types("Reference 1234 5678 9012 was logged.") == []


def test_aadhaar_invalid_with_explicit_cue_is_redacted():
    dets = detect("mera aadhaar number 1234 5678 9012 hai")
    assert [d.type for d in dets] == ["AADHAAR"]
    assert dets[0].confidence < 0.9


def test_aadhaar_batch_context_is_not_treated_as_identity():
    assert _types(f"Batch {A1_FMT} was processed.") == []


# --- names ------------------------------------------------------------------
def test_english_name_anchor_and_honorific():
    assert _values("My name is Rahul Sharma and I grow wheat.") == ["Rahul Sharma"]
    assert _values("Mr. Verma advised a neem spray.") == ["Verma"]


def test_name_initials_are_a_documented_limitation():
    assert "PERSON" not in _types("My name is R. K. Sharma.")


def test_lowercase_honorific_is_a_documented_limitation():
    assert "PERSON" not in _types("mr. verma advised a neem spray.")


def test_hindi_name_is_trimmed_before_danda():
    assert _values("मेरा नाम राहुल शर्मा है।") == ["राहुल शर्मा"]


def test_hinglish_name_and_ji():
    assert _values("mera naam Rahul hai, call 98765 43210") == ["Rahul", "98765 43210"]
    assert "PERSON" in _types("Rahul ji ka number 9876543210 hai")


def test_unanchored_names_are_not_detected():
    assert "PERSON" not in _types("Rahul said the mandi rate is good.")
    assert "PERSON" not in _types("रमेश ने कहा कि फसल ठीक है।")


def test_organisation_and_crop_after_name_anchor_are_rejected():
    assert "PERSON" not in _types("My name is Kisan Call Centre.")
    assert "PERSON" not in _types("Dr. Wheat visited the field.")


# --- multiple entities and spans -------------------------------------------
def test_multiple_entities_in_one_sentence():
    text = "My name is Rahul, call 9876543210, email rahul@example.com."
    assert _types(text) == ["PERSON", "PHONE", "EMAIL"]


def test_every_detection_span_matches_its_value():
    text = f"My name is Rahul, call 9876543210, email rahul@example.com, Aadhaar {A1_FMT}."
    for det in detect(text):
        assert text[det.start:det.end] == det.value
        assert 0 <= det.start < det.end <= len(text)


def test_empty_and_whitespace_input():
    assert detect("") == ()
    assert detect("   ") == ()
    assert redact("").redacted == ""


# --- redaction --------------------------------------------------------------
def test_redaction_placeholders_preserve_surrounding_text():
    text = "My name is Rahul, call 9876543210 or email rahul@example.com."
    assert redact(text).redacted == "My name is [PERSON], call [PHONE] or email [EMAIL]."


def test_redaction_is_idempotent():
    text = f"Call 9876543210, Aadhaar {A1_FMT}, IFSC SBIN0001234."
    once = redact(text).redacted
    twice = redact(once).redacted
    assert once == twice
    assert "9876543210" not in once
    assert A1_FMT not in once


def test_redaction_can_be_limited_to_selected_types():
    text = "Call 9876543210 or email rahul@example.com."
    result = redact(text, RedactionConfig(enabled_types=("EMAIL",)))
    assert "9876543210" in result.redacted
    assert "rahul@example.com" not in result.redacted
    assert result.redacted == "Call 9876543210 or email [EMAIL]."


def test_custom_placeholder():
    result = redact("Call 9876543210.", RedactionConfig(placeholders={"PHONE": "<phone>"}))
    assert result.redacted == "Call <phone>."


def test_redaction_does_not_leak_detected_values_in_result_counts():
    result = redact("Call 9876543210 or email rahul@example.com.")
    assert result.counts() == {"PHONE": 1, "EMAIL": 1}
    assert "9876543210" not in result.redacted


def test_redact_spans_masks_missed_gold_spans_and_merges_overlaps():
    text = "My name is R. K. Sharma."
    assert redact_spans(text, [(11, 23, "PERSON")]) == "My name is [PERSON]."
    # Overlapping detected/gold spans must not corrupt the offsets.
    assert redact_spans("mera naam Rahul sharma hai", [(10, 15, "PERSON"), (10, 22, "PERSON")]) == (
        "mera naam [PERSON] hai"
    )


# --- evaluation -------------------------------------------------------------
def test_strict_matching_metrics_on_toy_rows():
    rows = [
        {
            "id": "a",
            "language": "en",
            "text": "Call 9876543210",
            "entities": [{"start": 5, "end": 15, "type": "PHONE", "value": "9876543210"}],
            "notes": "",
        },
        {
            "id": "b",
            "language": "en",
            "text": "Call 9876543210",
            "entities": [],
            "notes": "",
        },
        {
            "id": "c",
            "language": "en",
            "text": "The dose is 2 kg urea.",
            "entities": [],
            "notes": "",
        },
    ]
    result = evaluate(rows, policy="balanced")
    overall = result["overall"]
    assert (overall["tp"], overall["fp"], overall["fn"]) == (1, 1, 0)
    assert overall["precision"] == 0.5
    assert overall["recall"] == 1.0
    # Doc a is exact; doc b has one FP; doc c is exact.
    assert overall["exact_document_rate"] == pytest.approx(2 / 3, abs=1e-4)


def test_dataset_is_synthetic_and_covers_all_supported_types():
    rows = load_rows()
    assert len(rows) >= 80
    assert all(row["synthetic"] == "true" for row in rows)
    assert {row["language"] for row in rows} == {"en", "hi", "hinglish"}
    observed = {entity["type"] for row in rows for entity in row["entities"]}
    assert observed == {"PHONE", "EMAIL", "PERSON", "AADHAAR", "IFSC", "BANK_ACCOUNT", "URL"}
