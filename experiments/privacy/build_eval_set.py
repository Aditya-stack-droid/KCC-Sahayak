"""Build the synthetic PII evaluation set.

Every example in this file is hand-authored and synthetic.  Names, phone
numbers, emails and identifiers are fake or use the reserved example.com
domain.  No real farmer data is used.

The dataset stores exact character spans.  Spans are computed by locating each
declared entity value in the text, so there is no manual offset bookkeeping
that could silently drift.  ``build`` refuses to write the dataset if a value
is missing or if expected spans overlap.

Run:
    .venv312/bin/python experiments/privacy/build_eval_set.py
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.privacy.recognizers import verhoeff_check_digit  # noqa: E402


def _aadhaar(base: str) -> str:
    """Return a 12-digit Verhoeff-valid synthetic identifier from an 11-digit base."""
    number = base + verhoeff_check_digit(base)
    assert len(number) == 12
    return number


def _fmt_aadhaar(number: str) -> str:
    return f"{number[:4]} {number[4:8]} {number[8:]}"


A1 = _aadhaar("23456789012")  # 234567890124
A2 = _aadhaar("34567890123")  # 345678901238
A3 = _aadhaar("45678901234")  # 456789012341
F1, F2, F3 = _fmt_aadhaar(A1), _fmt_aadhaar(A2), _fmt_aadhaar(A3)


# Each case: id, language, text, entities [(type, exact substring)], notes.
# "entities" is empty for negatives and ambiguous-number cases.
CASES: list[dict] = [
    # ------------------------------------------------------------------ English
    {
        "id": "en-phone-context",
        "language": "en",
        "text": "Call me at 9876543210 for the spray schedule.",
        "entities": [("PHONE", "9876543210")],
        "notes": "positive: strong phone cue",
    },
    {
        "id": "en-phone-spaced",
        "language": "en",
        "text": "My mobile number is 98765 43210.",
        "entities": [("PHONE", "98765 43210")],
        "notes": "positive: 5+5 separator grouping",
    },
    {
        "id": "en-phone-country",
        "language": "en",
        "text": "Contact +91-98765-43210 tomorrow morning.",
        "entities": [("PHONE", "+91-98765-43210")],
        "notes": "positive: +91 country prefix and separators",
    },
    {
        "id": "en-phone-bare",
        "language": "en",
        "text": "Reach 9876543210.",
        "entities": [("PHONE", "9876543210")],
        "notes": "ambiguous: standalone 10-digit number; balanced policy accepts",
    },
    {
        "id": "en-phone-whatsapp",
        "language": "en",
        "text": "WhatsApp 98765-43210 for seeds.",
        "entities": [("PHONE", "98765-43210")],
        "notes": "positive: WhatsApp cue",
    },
    {
        "id": "en-phone-intl",
        "language": "en",
        "text": "Call +1 (415) 555-0132 anytime.",
        "entities": [("PHONE", "+1 (415) 555-0132")],
        "notes": "positive: international number with explicit +1",
    },
    {
        "id": "en-email",
        "language": "en",
        "text": "Email me at rahul.sharma@example.com please.",
        "entities": [("EMAIL", "rahul.sharma@example.com")],
        "notes": "positive: email with dot and plus-safe local part",
    },
    {
        "id": "en-name",
        "language": "en",
        "text": "My name is Rahul Sharma and I grow wheat.",
        "entities": [("PERSON", "Rahul Sharma")],
        "notes": "positive: anchored English name",
    },
    {
        "id": "en-honorific",
        "language": "en",
        "text": "Mr. Verma advised a neem oil spray.",
        "entities": [("PERSON", "Verma")],
        "notes": "positive: honorific-anchored name",
    },
    {
        "id": "en-name-colon",
        "language": "en",
        "text": "Name: Rahul Sharma",
        "entities": [("PERSON", "Rahul Sharma")],
        "notes": "positive: form-style name label",
    },
    {
        "id": "en-name-lower",
        "language": "en",
        "text": "my name is rahul",
        "entities": [("PERSON", "rahul")],
        "notes": "positive: lowercase Roman-Hindi/English name after anchor",
    },
    {
        "id": "en-aadhaar",
        "language": "en",
        "text": f"Aadhaar {F1} is linked to the account.",
        "entities": [("AADHAAR", F1)],
        "notes": "positive: valid Verhoeff checksum plus Aadhaar cue",
    },
    {
        "id": "en-aadhaar-uid",
        "language": "en",
        "text": f"UID {F2} was recorded.",
        "entities": [("AADHAAR", F2)],
        "notes": "positive: UID cue",
    },
    {
        "id": "en-aadhaar-checksum-only",
        "language": "en",
        "text": f"The number {F3} belongs to the beneficiary.",
        "entities": [("AADHAAR", F3)],
        "notes": "positive: valid checksum without an explicit Aadhaar cue",
    },
    {
        "id": "en-ifsc",
        "language": "en",
        "text": "IFSC SBIN0001234 for the transfer.",
        "entities": [("IFSC", "SBIN0001234")],
        "notes": "positive: IFSC pattern",
    },
    {
        "id": "en-bank",
        "language": "en",
        "text": "Account number 123456789012 at the bank.",
        "entities": [("BANK_ACCOUNT", "123456789012")],
        "notes": "positive: account cue; not treated as Aadhaar",
    },
    {
        "id": "en-url",
        "language": "en",
        "text": "Profile: https://example.com/farmer/rahul",
        "entities": [("URL", "https://example.com/farmer/rahul")],
        "notes": "positive: URL",
    },
    {
        "id": "en-multi-name-phone-email",
        "language": "en",
        "text": "My name is Rahul Sharma, call 9876543210 or email rahul@example.com.",
        "entities": [
            ("PERSON", "Rahul Sharma"),
            ("PHONE", "9876543210"),
            ("EMAIL", "rahul@example.com"),
        ],
        "notes": "positive: multiple PII entities",
    },
    {
        "id": "en-multi-aadhaar-phone",
        "language": "en",
        "text": f"Aadhaar {F2}; mobile 98765 43210.",
        "entities": [("AADHAAR", F2), ("PHONE", "98765 43210")],
        "notes": "positive: multiple identifiers",
    },
    {
        "id": "en-multi-all",
        "language": "en",
        "text": (
            f"My name is Rahul, my number is 9876543210, my email is "
            f"rahul@example.com, my Aadhaar is {F1}."
        ),
        "entities": [
            ("PERSON", "Rahul"),
            ("PHONE", "9876543210"),
            ("EMAIL", "rahul@example.com"),
            ("AADHAAR", F1),
        ],
        "notes": "positive: four entity types in one sentence",
    },
    {
        "id": "en-multi-aadhaar-ifsc",
        "language": "en",
        "text": f"Aadhaar {F3} and IFSC HDFC0000123.",
        "entities": [("AADHAAR", F3), ("IFSC", "HDFC0000123")],
        "notes": "positive: identifier plus bank code",
    },
    {
        "id": "en-email-uppercase",
        "language": "en",
        "text": "Email RAHUL.SHARMA@EXAMPLE.COM now.",
        "entities": [("EMAIL", "RAHUL.SHARMA@EXAMPLE.COM")],
        "notes": "positive: uppercase email",
    },
    {
        "id": "en-bank-ac",
        "language": "en",
        "text": "A/c 123456789012 at the branch.",
        "entities": [("BANK_ACCOUNT", "123456789012")],
        "notes": "positive: A/c abbreviation",
    },
    {
        "id": "en-aadhaar-hyphen",
        "language": "en",
        "text": f"Aadhaar {A1[:4]}-{A1[4:8]}-{A1[8:]} was verified.",
        "entities": [("AADHAAR", f"{A1[:4]}-{A1[4:8]}-{A1[8:]}")],
        "notes": "positive: hyphen-grouped Aadhaar",
    },
    # Deliberately hard examples.  These are expected to expose limitations.
    {
        "id": "en-name-initials",
        "language": "en",
        "text": "My name is R. K. Sharma.",
        "entities": [("PERSON", "R. K. Sharma")],
        "notes": "hard: initials are not supported by the name heuristics",
    },
    {
        "id": "en-honorific-lowercase",
        "language": "en",
        "text": "mr. verma advised a neem spray.",
        "entities": [("PERSON", "verma")],
        "notes": "hard: honorific matching is case-sensitive",
    },
    {
        "id": "en-phone-dots",
        "language": "en",
        "text": "Call 98765.43210 now.",
        "entities": [("PHONE", "98765.43210")],
        "notes": "hard: dot-grouped mobile is not supported",
    },
    {
        "id": "en-phone-trunk-zero",
        "language": "en",
        "text": "Call 09876543210 now.",
        "entities": [("PHONE", "09876543210")],
        "notes": "hard: Indian trunk prefix 0 is not supported",
    },
    {
        "id": "hin-name-lowercase-surname",
        "language": "hinglish",
        "text": "mera naam Rahul sharma hai",
        "entities": [("PERSON", "Rahul sharma")],
        "notes": "hard: lowercase surname is only partially captured",
    },
    {
        "id": "hi-phone-plus91",
        "language": "hi",
        "text": "संपर्क +91 98765 43210",
        "entities": [("PHONE", "+91 98765 43210")],
        "notes": "positive: +91 with a space before the national number",
    },
    {
        "id": "hi-name-unanchored",
        "language": "hi",
        "text": "रमेश ने कहा कि फसल ठीक है।",
        "entities": [],
        "notes": "scope limit: unanchored Devanagari names are not detected",
    },
    # Negatives and hard cases
    {
        "id": "en-neg-dose",
        "language": "en",
        "text": "The dose is 2 kg urea per bigha.",
        "entities": [],
        "notes": "negative: unit/quantity, not a phone number",
    },
    {
        "id": "en-neg-year",
        "language": "en",
        "text": "In 2024 the rainfall was low.",
        "entities": [],
        "notes": "negative: year",
    },
    {
        "id": "en-neg-price",
        "language": "en",
        "text": "Price is Rs 50000 per quintal.",
        "entities": [],
        "notes": "negative: currency/quantity",
    },
    {
        "id": "en-neg-plot",
        "language": "en",
        "text": "Plot number 9876543210 is near the canal.",
        "entities": [],
        "notes": "ambiguous: 10 digits but plot context; must not be a phone",
    },
    {
        "id": "en-neg-token",
        "language": "en",
        "text": "Token 9876543210 was issued.",
        "entities": [],
        "notes": "ambiguous: token context",
    },
    {
        "id": "en-neg-order-id",
        "language": "en",
        "text": "The sample ID is 9876543210-A.",
        "entities": [],
        "notes": "ambiguous: sample ID",
    },
    {
        "id": "en-neg-budget",
        "language": "en",
        "text": "Rs 9876543210 is the total budget.",
        "entities": [],
        "notes": "ambiguous: currency context",
    },
    {
        "id": "en-neg-9digit",
        "language": "en",
        "text": "The code is 987654321.",
        "entities": [],
        "notes": "negative: 9 digits",
    },
    {
        "id": "en-neg-11digit",
        "language": "en",
        "text": "The code is 98765432101.",
        "entities": [],
        "notes": "negative: 11 digits, must not match a 10-digit sub-span",
    },
    {
        "id": "en-neg-wrong-start",
        "language": "en",
        "text": "The code is 5876543210.",
        "entities": [],
        "notes": "negative: Indian mobiles start 6-9",
    },
    {
        "id": "en-neg-pincode",
        "language": "en",
        "text": "PIN 110001",
        "entities": [],
        "notes": "negative: 6-digit PIN code",
    },
    {
        "id": "en-neg-invalid-aadhaar",
        "language": "en",
        "text": "Reference 1234 5678 9012 was logged.",
        "entities": [],
        "notes": "negative: 12 digits, invalid checksum, no Aadhaar cue",
    },
    {
        "id": "en-neg-batch-valid-aadhaar",
        "language": "en",
        "text": f"Batch {F1} was processed.",
        "entities": [],
        "notes": "ambiguous: valid checksum but batch context; not treated as Aadhaar",
    },
    {
        "id": "en-neg-org-name",
        "language": "en",
        "text": "My name is Kisan Call Centre.",
        "entities": [],
        "notes": "negative: organisation after a name anchor",
    },
    {
        "id": "en-neg-crop-honorific",
        "language": "en",
        "text": "Dr. Wheat visited the field.",
        "entities": [],
        "notes": "negative: crop word after an honorific",
    },
    {
        "id": "en-neg-bare-name",
        "language": "en",
        "text": "Rahul said the mandi rate is good.",
        "entities": [],
        "notes": "scope limit: unanchored names are not detected",
    },
    {
        "id": "en-neg-unknown-name",
        "language": "en",
        "text": "my name is not known",
        "entities": [],
        "notes": "negative: stopword phrase after anchor",
    },
    {
        "id": "en-neg-name-of-crop",
        "language": "en",
        "text": "The name of the crop is wheat.",
        "entities": [],
        "notes": "negative: 'name of' is not a person",
    },
    {
        "id": "en-neg-date",
        "language": "en",
        "text": "The meeting is on 12/05/2024.",
        "entities": [],
        "notes": "negative: date",
    },
    # ------------------------------------------------------------------ Hinglish
    {
        "id": "hin-phone-name",
        "language": "hinglish",
        "text": "mera naam Rahul hai, call 98765 43210",
        "entities": [("PERSON", "Rahul"), ("PHONE", "98765 43210")],
        "notes": "positive: Roman-Hindi anchor plus phone",
    },
    {
        "id": "hin-ji-number",
        "language": "hinglish",
        "text": "Rahul ji ka number 9876543210 hai",
        "entities": [("PERSON", "Rahul"), ("PHONE", "9876543210")],
        "notes": "positive: 'ji' name and weak number cue",
    },
    {
        "id": "hin-email",
        "language": "hinglish",
        "text": "mera email rahul.kcc@example.com hai",
        "entities": [("EMAIL", "rahul.kcc@example.com")],
        "notes": "positive: email in Roman-Hindi",
    },
    {
        "id": "hin-aadhaar",
        "language": "hinglish",
        "text": f"mera aadhaar {F1} hai",
        "entities": [("AADHAAR", F1)],
        "notes": "positive: Aadhaar cue in Roman-Hindi",
    },
    {
        "id": "hin-aadhaar-invalid-context",
        "language": "hinglish",
        "text": "mera aadhaar number 1234 5678 9012 hai",
        "entities": [("AADHAAR", "1234 5678 9012")],
        "notes": "positive: explicit Aadhaar cue, invalid checksum (test-number case)",
    },
    {
        "id": "hin-bank",
        "language": "hinglish",
        "text": "account number 123456789012 par paisa bhejo",
        "entities": [("BANK_ACCOUNT", "123456789012")],
        "notes": "positive: account cue",
    },
    {
        "id": "hin-phone-sep",
        "language": "hinglish",
        "text": "mobile number 98765-43210",
        "entities": [("PHONE", "98765-43210")],
        "notes": "positive: mobile cue with separator",
    },
    {
        "id": "hin-name-comma",
        "language": "hinglish",
        "text": "naam Rahul Sharma, gaon Karnal",
        "entities": [("PERSON", "Rahul Sharma")],
        "notes": "positive: form-style naam label",
    },
    {
        "id": "hin-multi",
        "language": "hinglish",
        "text": "Rahul ji, 9876543210, rahul@example.com",
        "entities": [("PERSON", "Rahul"), ("PHONE", "9876543210"), ("EMAIL", "rahul@example.com")],
        "notes": "positive: multiple entities",
    },
    {
        "id": "hin-name-year",
        "language": "hinglish",
        "text": "mera naam Rahul hai aur 2024 mein fasal kharab hui",
        "entities": [("PERSON", "Rahul")],
        "notes": "positive: name; year must not be redacted",
    },
    {
        "id": "hin-name-phone",
        "language": "hinglish",
        "text": "mera naam Priya hai, mera number 98765 43210 hai",
        "entities": [("PERSON", "Priya"), ("PHONE", "98765 43210")],
        "notes": "positive: name plus weak number cue",
    },
    {
        "id": "hin-name-kisan",
        "language": "hinglish",
        "text": "kisan ka naam Ram Singh hai",
        "entities": [("PERSON", "Ram Singh")],
        "notes": "positive: name after a non-name subject",
    },
    {
        "id": "hin-devanagari-phone",
        "language": "hinglish",
        "text": "mobile ९८७६५४३२१० hai",
        "entities": [("PHONE", "९८७६५४३२१०")],
        "notes": "positive: Devanagari digits in a Roman-Hindi sentence",
    },
    {
        "id": "hin-neg-dose",
        "language": "hinglish",
        "text": "khet mein 2 kg urea dalna hai",
        "entities": [],
        "notes": "negative: quantity/unit",
    },
    {
        "id": "hin-neg-price",
        "language": "hinglish",
        "text": "aaj mandi ka bhav 5000 rupaye hai",
        "entities": [],
        "notes": "negative: price",
    },
    {
        "id": "hin-neg-wrong-start",
        "language": "hinglish",
        "text": "call 5876543210",
        "entities": [],
        "notes": "negative: wrong mobile prefix",
    },
    {
        "id": "hin-neg-order",
        "language": "hinglish",
        "text": "order id 9876543210",
        "entities": [],
        "notes": "ambiguous: order context",
    },
    {
        "id": "hin-neg-naam-verb",
        "language": "hinglish",
        "text": "naam likhwa do",
        "entities": [],
        "notes": "negative: 'naam' followed by a verb, not a person",
    },
    # ---------------------------------------------------------------------- Hindi
    {
        "id": "hi-name",
        "language": "hi",
        "text": "मेरा नाम राहुल शर्मा है।",
        "entities": [("PERSON", "राहुल शर्मा")],
        "notes": "positive: Devanagari name anchor",
    },
    {
        "id": "hi-name-colon",
        "language": "hi",
        "text": "नाम: राहुल शर्मा",
        "entities": [("PERSON", "राहुल शर्मा")],
        "notes": "positive: Devanagari form label",
    },
    {
        "id": "hi-phone",
        "language": "hi",
        "text": "मोबाइल नंबर 9876543210 है।",
        "entities": [("PHONE", "9876543210")],
        "notes": "positive: Hindi mobile cue",
    },
    {
        "id": "hi-phone-contact",
        "language": "hi",
        "text": "संपर्क करें: 98765 43210",
        "entities": [("PHONE", "98765 43210")],
        "notes": "positive: Hindi contact cue, 5+5 grouping",
    },
    {
        "id": "hi-phone-intl",
        "language": "hi",
        "text": "कॉल करें +1 (415) 555-0132",
        "entities": [("PHONE", "+1 (415) 555-0132")],
        "notes": "positive: international number in Hindi text",
    },
    {
        "id": "hi-email",
        "language": "hi",
        "text": "मेरा ईमेल rahul@example.com है।",
        "entities": [("EMAIL", "rahul@example.com")],
        "notes": "positive: email in Devanagari sentence",
    },
    {
        "id": "hi-aadhaar",
        "language": "hi",
        "text": f"आधार {F1} दर्ज है।",
        "entities": [("AADHAAR", F1)],
        "notes": "positive: Hindi Aadhaar cue",
    },
    {
        "id": "hi-honorific",
        "language": "hi",
        "text": "श्री रमेश जी से बात करें",
        "entities": [("PERSON", "रमेश")],
        "notes": "positive: honorific plus ji; honorific and ji overlap",
    },
    {
        "id": "hi-honorific-ji",
        "language": "hi",
        "text": "श्रीमती सुनीता जी",
        "entities": [("PERSON", "सुनीता")],
        "notes": "positive: feminine honorific plus ji",
    },
    {
        "id": "hi-bank",
        "language": "hi",
        "text": "खाता संख्या 123456789012 है।",
        "entities": [("BANK_ACCOUNT", "123456789012")],
        "notes": "positive: Hindi account cue",
    },
    {
        "id": "hi-multi",
        "language": "hi",
        "text": "मेरा नाम राहुल है, फोन 9876543210, ईमेल rahul@example.com",
        "entities": [("PERSON", "राहुल"), ("PHONE", "9876543210"), ("EMAIL", "rahul@example.com")],
        "notes": "positive: three entity types",
    },
    {
        "id": "hi-devanagari-phone",
        "language": "hi",
        "text": "मोबाइल ९८७६५४३२१० है",
        "entities": [("PHONE", "९८७६५४३२१०")],
        "notes": "positive: Devanagari-digit phone",
    },
    {
        "id": "hi-name-year",
        "language": "hi",
        "text": "मेरा नाम राहुल है और 2024 में सूखा पड़ा",
        "entities": [("PERSON", "राहुल")],
        "notes": "positive: name; year must not be redacted",
    },
    # Hindi negatives and hard cases
    {
        "id": "hi-neg-dose",
        "language": "hi",
        "text": "गेहूं में 2 किलो यूरिया डालें",
        "entities": [],
        "notes": "negative: quantity/unit",
    },
    {
        "id": "hi-neg-price",
        "language": "hi",
        "text": "कीमत 5000 रुपये प्रति क्विंटल",
        "entities": [],
        "notes": "negative: price",
    },
    {
        "id": "hi-neg-plot",
        "language": "hi",
        "text": "प्लॉट नंबर 9876543210",
        "entities": [],
        "notes": "ambiguous: Hindi plot context",
    },
    {
        "id": "hi-neg-invalid-aadhaar",
        "language": "hi",
        "text": "संख्या 1234 5678 9012 दर्ज है",
        "entities": [],
        "notes": "negative: 12 digits, no Aadhaar cue, invalid checksum",
    },
    {
        "id": "hi-neg-naam-verb",
        "language": "hi",
        "text": "नाम लिखवा दो",
        "entities": [],
        "notes": "negative: 'नाम' followed by a verb",
    },
    {
        "id": "hi-neg-org-name",
        "language": "hi",
        "text": "मेरा नाम किसान केंद्र है",
        "entities": [],
        "notes": "negative: organisation after a name anchor",
    },
]


def build() -> list[dict]:
    rows: list[dict] = []
    for case in CASES:
        text = case["text"]
        cursor = 0
        entities = []
        for kind, value in case["entities"]:
            start = text.find(value, cursor)
            if start < 0:
                raise ValueError(f"{case['id']}: value not found after cursor: {value!r}")
            end = start + len(value)
            if text[start:end] != value:
                raise ValueError(f"{case['id']}: span mismatch for {value!r}")
            entities.append({"start": start, "end": end, "type": kind, "value": value})
            cursor = end
        entities.sort(key=lambda e: e["start"])
        for left, right in zip(entities, entities[1:]):
            if left["end"] > right["start"]:
                raise ValueError(f"{case['id']}: overlapping expected spans")
        rows.append(
            {
                "id": case["id"],
                "language": case["language"],
                "text": text,
                "entities": json.dumps(entities, ensure_ascii=False, separators=(",", ":")),
                "notes": case["notes"],
                "synthetic": "true",
            }
        )
    return rows


def main() -> None:
    rows = build()
    out_dir = HERE / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "synthetic_pii_eval.csv"
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["id", "language", "text", "entities", "notes", "synthetic"],
        )
        writer.writeheader()
        writer.writerows(rows)

    by_language: dict[str, int] = {}
    by_type: dict[str, int] = {}
    for row in rows:
        by_language[row["language"]] = by_language.get(row["language"], 0) + 1
        for ent in json.loads(row["entities"]):
            by_type[ent["type"]] = by_type.get(ent["type"], 0) + 1
    print(f"wrote {out_path} ({len(rows)} synthetic examples)")
    print("by language:", json.dumps(by_language, sort_keys=True))
    print("by entity type:", json.dumps(by_type, sort_keys=True))
    print(f"negatives/ambiguous: {sum(1 for r in rows if r['entities'] == '[]')}")


if __name__ == "__main__":
    main()
