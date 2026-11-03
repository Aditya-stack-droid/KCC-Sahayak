"""Standalone PII recognizers for KCC-Sahayak (Batch B, Phase 1).

This module is additive.  It does not import or modify ``src/``, the backend,
the original retrieval index or any Batch A artefact.

Supported entity types
----------------------
PHONE          Indian mobile numbers (with optional +91/0091 and separators)
               and international numbers that carry an explicit +/00 prefix.
EMAIL          ASCII email addresses.
PERSON         Names only when introduced by an explicit anchor or honorific
               ("my name is", "naam", "Mr.", "श्री", "Rahul ji").  This is a
               deliberately conservative, best-effort recogniser.
AADHAAR        12-digit Aadhaar-like identifiers.  A valid Verhoeff checksum
               is strong evidence; an explicit Aadhaar cue is accepted even
               when the checksum fails (test numbers are often invalid).
IFSC           Indian bank branch IFSC codes.
BANK_ACCOUNT   9-18 digit account numbers, only when a bank/account cue is
               present.  The cue requirement keeps random long numbers out.
URL            http(s):// and www. URLs.

Design notes
------------
* Devanagari digits are transliterated 1:1 to ASCII before matching, so spans
  still refer to the original string and redaction preserves the original text.
* A 10-digit number is not automatically a phone.  Negative context (plot,
  token, order, account, units, currency, year, ...) suppresses it unless a
  strong phone cue is present.
* Overlaps are resolved by entity priority, then span length.  AADHAAR and
  BANK_ACCOUNT therefore win over a 10-digit sub-span of a 12-digit number.
* No network access, no external services, no new dependencies.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Sequence

ENTITY_TYPES = (
    "PHONE",
    "EMAIL",
    "PERSON",
    "AADHAAR",
    "IFSC",
    "BANK_ACCOUNT",
    "URL",
)

PRIORITY = {
    "AADHAAR": 100,
    "BANK_ACCOUNT": 90,
    "PHONE": 80,
    "EMAIL": 70,
    "IFSC": 60,
    "URL": 50,
    "PERSON": 40,
}

DEVANAGARI_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")


def normalize_digits(text: str) -> str:
    """Return *text* with Devanagari digits replaced by ASCII digits.

    The translation is character-for-character, so every match offset in the
    normalised string is also a valid offset in the original string.
    """
    return text.translate(DEVANAGARI_DIGITS)


@dataclass(frozen=True)
class Detection:
    """One detected span in the original (not normalised) text."""

    start: int
    end: int
    type: str
    confidence: float
    value: str

    def as_dict(self) -> dict:
        return {
            "start": self.start,
            "end": self.end,
            "type": self.type,
            "confidence": round(self.confidence, 3),
            "value": self.value,
        }


# ---------------------------------------------------------------------------
# Verhoeff checksum (used by Aadhaar).  The tables are the standard ones.
# ---------------------------------------------------------------------------

_VERHOEFF_D = (
    (0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
    (1, 2, 3, 4, 0, 6, 7, 8, 9, 5),
    (2, 3, 4, 0, 1, 7, 8, 9, 5, 6),
    (3, 4, 0, 1, 2, 8, 9, 5, 6, 7),
    (4, 0, 1, 2, 3, 9, 5, 6, 7, 8),
    (5, 9, 8, 7, 6, 0, 4, 3, 2, 1),
    (6, 5, 9, 8, 7, 1, 0, 4, 3, 2),
    (7, 6, 5, 9, 8, 2, 1, 0, 4, 3),
    (8, 7, 6, 5, 9, 3, 2, 1, 0, 4),
    (9, 8, 7, 6, 5, 4, 3, 2, 1, 0),
)

_VERHOEFF_P = (
    (0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
    (1, 5, 7, 6, 2, 8, 3, 0, 9, 4),
    (5, 8, 0, 3, 7, 9, 6, 1, 4, 2),
    (8, 9, 1, 6, 0, 4, 3, 5, 2, 7),
    (9, 4, 5, 3, 1, 2, 6, 8, 7, 0),
    (4, 2, 8, 6, 5, 7, 3, 9, 0, 1),
    (2, 7, 9, 3, 8, 0, 6, 4, 1, 5),
    (7, 0, 4, 6, 9, 1, 3, 2, 5, 8),
)

_VERHOEFF_INV = (0, 4, 3, 2, 1, 5, 6, 7, 8, 9)


def verhoeff_valid(number: str) -> bool:
    """Return True when *number* (digits only) has a valid Verhoeff check."""
    if not number.isdigit():
        return False
    checksum = 0
    for i, char in enumerate(reversed(number)):
        checksum = _VERHOEFF_D[checksum][_VERHOEFF_P[i % 8][int(char)]]
    return checksum == 0


def verhoeff_check_digit(base: str) -> str:
    """Return the Verhoeff check digit for *base* (digits only, no check digit)."""
    if not base.isdigit():
        raise ValueError("base must contain digits only")
    checksum = 0
    for i, char in enumerate(reversed(base)):
        checksum = _VERHOEFF_D[checksum][_VERHOEFF_P[(i + 1) % 8][int(char)]]
    return str(_VERHOEFF_INV[checksum])


# ---------------------------------------------------------------------------
# Patterns
# ---------------------------------------------------------------------------

# 10-digit Indian mobile, optionally with +91/0091.  Separators are allowed
# only in the common 5+5 and 3+3+4 groupings.
_MOBILE_RE = re.compile(
    r"(?<!\d)"
    r"(?:(?:\+|00)?91[\s\-.]?)?"
    r"((?:[6-9]\d{9})|(?:[6-9]\d{4}[\s\-]\d{5})|(?:[6-9]\d{2}[\s\-]\d{3}[\s\-]\d{4}))"
    r"(?!\d)"
)

# International numbers require an explicit + or 00 prefix.  This avoids
# treating arbitrary long digit strings as international phone numbers.
_INTL_PHONE_RE = re.compile(r"(?<![\w+])(?:\+|00)\d[\d\s().-]{6,18}\d(?!\d)")

_EMAIL_RE = re.compile(
    r"(?<![\w.+-])[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}(?![\w])"
)

_URL_RE = re.compile(r"(?:https?://|www\.)[^\s<>\"']+", re.IGNORECASE)

_IFSC_RE = re.compile(r"(?<![A-Z0-9])[A-Z]{4}0[A-Z0-9]{6}(?![A-Z0-9])")

_AADHAAR_RE = re.compile(r"(?<!\d)\d{4}[\s-]?\d{4}[\s-]?\d{4}(?!\d)")

_BANK_RE = re.compile(
    r"(?i)(?:a/c|account|acct|\u0916\u093e\u0924\u093e|\u0916\u093e\u0924\u0947|bank|\u092c\u0948\u0902\u0915)"
    r"\s*(?:no\.?|number|\u0938\u0902\u0916\u094d\u092f\u093e)?\s*[:\-]?\s*"
    r"(\d{9,18})"
)

# Name anchors.  The capture group is the candidate name; it is trimmed below.
_LATIN_NAME_ANCHOR_RE = re.compile(
    r"\b(?i:my name is|name is|name|mera naam|mera name|naam)\s*[:\-]?\s+"
    r"([A-Za-z][A-Za-z'.-]+(?:\s+[A-Z][A-Za-z'.-]+){0,2})"
)
_LATIN_HONORIFIC_RE = re.compile(
    r"\b(?:Mr|Mrs|Ms|Dr|Shri|Smt|Sri)\.?\s+"
    r"([A-Z][A-Za-z'.-]+(?:\s+[A-Z][A-Za-z'.-]+){0,2})"
)
_LATIN_JI_RE = re.compile(r"\b([A-Z][a-z]+)\s+ji\b")

_HINDI_NAME_ANCHOR_RE = re.compile(
    r"(?<![\u0900-\u097F])(?:\u092e\u0947\u0930\u093e\s+\u0928\u093e\u092e|\u0928\u093e\u092e)"
    r"(?![\u0900-\u097F])\s*[:\-]?\s*"
    r"([\u0900-\u097F]{2,}(?:\s+[\u0900-\u097F]{2,}){0,2})"
)
_HINDI_HONORIFIC_RE = re.compile(
    r"(?:\u0936\u094d\u0930\u0940\u092e\u0924\u0940|\u0936\u094d\u0930\u0940|\u0938\u0941\u0936\u094d\u0930\u0940|\u0921\u0949)"
    r"(?![\u0900-\u097F])\.?\s*([\u0900-\u097F]{2,}(?:\s+[\u0900-\u097F]{2,}){0,2})"
)
_HINDI_JI_RE = re.compile(r"([\u0900-\u097F]{2,})\s+\u091c\u0940(?![\u0900-\u097F])")


# Context vocabularies.  Matching is case-insensitive substring matching on a
# +/- 45 character window around the candidate span.
_PHONE_STRONG = (
    "call",
    "mobile",
    "phone",
    "whatsapp",
    "dial",
    "contact",
    "\u0938\u0902\u092a\u0930\u094d\u0915",
    "\u092b\u094b\u0928",
    "\u092e\u094b\u092c\u093e\u0907\u0932",
    "\u0935\u094d\u0939\u093e\u091f\u094d\u0938\u090f\u092a",
    "\u0915\u0949\u0932",
    "\u0935\u094d\u0939\u093e\u091f\u094d\u0938\u0910\u092a",
)
_PHONE_WEAK = (
    "number",
    "no.",
    "\u0928\u0902\u092c\u0930",
    "\u0928\u092e\u094d\u092c\u0930",
)
_PHONE_NEGATIVE = (
    "plot",
    "\u092a\u094d\u0932\u0949\u091f",
    "token",
    "order",
    "invoice",
    "serial",
    "roll",
    "gst",
    "registration",
    "account",
    "aadhaar",
    "aadhar",
    "\u0906\u0927\u093e\u0930",
    "\u0916\u093e\u0924\u093e",
    "pincode",
    "pin code",
    "sample",
    "code",
    " kg",
    "kgs",
    " gm",
    "gms",
    " gram",
    " ml",
    "litre",
    "liter",
    "quintal",
    "acre",
    "bigha",
    "%",
    "\u20b9",
    " rs",
    "rs ",
    "rs.",
    "rupees",
    "\u0930\u0941\u092a\u092f\u0947",
    "\u0915\u093f\u0932\u094b",
    "\u0932\u0940\u091f\u0930",
    "\u090f\u0915\u0921\u093c",
    "\u092c\u0940\u0918\u093e",
    "\u0935\u0930\u094d\u0937",
    "\u0938\u093e\u0932",
    "year",
)
# Negative cues that only make sense immediately after a number (units and
# currency).  Identity words after the number, such as "my Aadhaar", must not
# suppress a phone that appears earlier in the same sentence.
_PHONE_NEGATIVE_POST = (
    " kg",
    "kgs",
    " gm",
    "gms",
    " gram",
    " ml",
    "litre",
    "liter",
    "quintal",
    "acre",
    "bigha",
    "%",
    "\u20b9",
    " rs",
    "rs ",
    "rs.",
    "rupees",
    "\u0930\u0941\u092a\u092f\u0947",
    "\u0915\u093f\u0932\u094b",
    "\u0932\u0940\u091f\u0930",
    "\u090f\u0915\u0921\u093c",
    "\u092c\u0940\u0918\u093e",
)

_AADHAAR_CUES = ("aadhaar", "aadhar", "uidai", "uid", "\u0906\u0927\u093e\u0930")
_AADHAAR_NEGATIVE = (
    "batch",
    "sample",
    "order",
    "invoice",
    "token",
    "serial",
    "roll",
    "code",
    "account",
    "a/c",
)
_BANK_CUES = (
    "account",
    "a/c",
    "acct",
    "bank",
    "\u0916\u093e\u0924\u093e",
    "\u0916\u093e\u0924\u0947",
    "\u092c\u0948\u0902\u0915",
)

# A conservative blocklist for name candidates.  Names are only emitted after
# an explicit anchor; this list stops common anchors such as "name of crop"
# and "Kisan Call Centre" from being reported as people.
_NAME_STOP = {
    "not",
    "known",
    "unknown",
    "none",
    "na",
    "the",
    "of",
    "and",
    "or",
    "is",
    "are",
    "hai",
    "hain",
    "ji",
    "ka",
    "ki",
    "ke",
    "mein",
    "se",
    "ko",
    "email",
    "mobile",
    "phone",
    "number",
    "call",
    "contact",
    "\u0939\u0948",
    "\u0939\u0948\u0902",
    "\u0914\u0930",
    "\u092b\u094b\u0928",
    "\u092e\u094b\u092c\u093e\u0907\u0932",
    "\u0928\u0902\u092c\u0930",
    "\u0908\u092e\u0947\u0932",
    "\u0906\u0927\u093e\u0930",
    "\u091c\u0940",
    "\u0938\u0947",
    "\u0915\u094b",
    "\u0915\u093e",
    "\u0915\u0940",
    "\u0915\u0947",
    "\u092e\u0947\u0902",
}
_NAME_BLOCK = {
    "kisan",
    "call",
    "centre",
    "center",
    "krishi",
    "vigyan",
    "kendra",
    "department",
    "officer",
    "advisor",
    "expert",
    "farmer",
    "wheat",
    "paddy",
    "rice",
    "cotton",
    "mustard",
    "urea",
    "dap",
    "crop",
    "plant",
    "field",
    "mandi",
    "india",
    "government",
    "likhwa",
    "likhwao",
    "batao",
    "bataye",
    "kya",
    "kaun",
    "kab",
    "kahan",
    "kaise",
    "do",
    "dena",
    "\u0915\u093f\u0938\u093e\u0928",
    "\u0915\u0943\u0937\u093f",
    "\u0935\u093f\u092d\u093e\u0917",
    "\u0905\u0927\u093f\u0915\u093e\u0930\u0940",
    "\u0917\u0947\u0939\u0942\u0902",
    "\u0927\u093e\u0928",
    "\u0915\u092a\u093e\u0938",
    "\u0938\u0930\u0938\u094b\u0902",
    "\u0916\u093e\u0926",
    "\u092f\u0942\u0930\u093f\u092f\u093e",
    "\u092b\u0938\u0932",
    "\u0916\u0947\u0924",
    "\u092e\u0902\u0921\u0940",
    "\u092d\u093e\u0930\u0924",
    "\u0938\u0930\u0915\u093e\u0930",
    "\u0932\u093f\u0916\u0935\u093e",
    "\u092c\u0924\u093e\u0913",
    "\u0915\u094d\u092f\u093e",
    "\u0915\u094c\u0928",
    "\u0915\u092c",
    "\u0915\u0939\u093e\u0902",
    "\u0915\u0948\u0938\u0947",
    "\u0926\u094b",
    "\u0926\u0947\u0928\u093e",
}


def _window(text: str, start: int, end: int, radius: int = 45) -> str:
    return text[max(0, start - radius): min(len(text), end + radius)].lower()


def _has_any(haystack: str, needles: Sequence[str]) -> bool:
    return any(needle in haystack for needle in needles)


def _add(dets: list[Detection], text: str, start: int, end: int, kind: str, confidence: float) -> None:
    if end <= start:
        return
    dets.append(Detection(start, end, kind, confidence, text[start:end]))


def _detect_aadhaar(text: str, norm: str, dets: list[Detection]) -> None:
    for match in _AADHAAR_RE.finditer(norm):
        digits = re.sub(r"\D", "", match.group(0))
        if len(digits) != 12:
            continue
        window = _window(norm, match.start(), match.end())
        has_aadhaar_cue = _has_any(window, _AADHAAR_CUES)
        has_bank_cue = _has_any(window, _BANK_CUES)
        has_negative_cue = _has_any(window, _AADHAAR_NEGATIVE)
        # Real Aadhaar numbers start 2-9.  A test number with an explicit
        # Aadhaar cue is still redacted even if its first digit or checksum
        # is invalid.
        if digits[0] not in "23456789" and not has_aadhaar_cue:
            continue
        # An account number can coincidentally pass the checksum (1 in 10).
        # A bank cue without an Aadhaar cue is treated as a bank account.
        if has_bank_cue and not has_aadhaar_cue:
            continue
        if has_negative_cue and not has_aadhaar_cue:
            continue
        valid = verhoeff_valid(digits)
        if not valid and not has_aadhaar_cue:
            continue
        _add(dets, text, match.start(), match.end(), "AADHAAR", 0.99 if valid else 0.6)


def _detect_bank(text: str, norm: str, dets: list[Detection]) -> None:
    for match in _BANK_RE.finditer(norm):
        start, end = match.span(1)
        _add(dets, text, start, end, "BANK_ACCOUNT", 0.9)


def _detect_phones(text: str, norm: str, policy: str, dets: list[Detection]) -> None:
    for match in _MOBILE_RE.finditer(norm):
        full = match.group(0)
        digits = re.sub(r"\D", "", full)
        national = digits[-10:] if len(digits) >= 10 else digits
        if not re.fullmatch(r"[6-9]\d{9}", national):
            continue
        if len(digits) not in (10, 12):
            continue
        window = _window(norm, match.start(), match.end())
        # Negative cues are local to the candidate: a later email or Aadhaar
        # elsewhere in the sentence must not suppress this phone number.
        before = norm[max(0, match.start() - 30): match.start()].lower()
        after = norm[match.end(): match.end() + 12].lower()
        strong = _has_any(window, _PHONE_STRONG)
        weak = _has_any(window, _PHONE_WEAK)
        negative = _has_any(before, _PHONE_NEGATIVE) or _has_any(after, _PHONE_NEGATIVE_POST)
        has_prefix = full.lstrip().startswith(("+", "00"))
        has_separator = bool(re.search(r"[\s\-]", full))
        if negative and not strong:
            continue
        if strong or has_prefix:
            confidence = 0.95
        elif has_separator or weak:
            confidence = 0.8
        else:
            if policy == "strict":
                continue
            confidence = 0.6
        _add(dets, text, match.start(), match.end(), "PHONE", confidence)

    for match in _INTL_PHONE_RE.finditer(norm):
        digits = re.sub(r"\D", "", match.group(0))
        if not 8 <= len(digits) <= 15:
            continue
        _add(dets, text, match.start(), match.end(), "PHONE", 0.95)


def _detect_email(text: str, norm: str, dets: list[Detection]) -> None:
    for match in _EMAIL_RE.finditer(norm):
        _add(dets, text, match.start(), match.end(), "EMAIL", 0.99)


def _detect_url(text: str, norm: str, dets: list[Detection]) -> None:
    for match in _URL_RE.finditer(norm):
        start, end = match.start(), match.end()
        while end > start and norm[end - 1] in ".,;:!?)]}'\"":
            end -= 1
        _add(dets, text, start, end, "URL", 0.9)


def _detect_ifsc(text: str, norm: str, dets: list[Detection]) -> None:
    for match in _IFSC_RE.finditer(norm):
        _add(dets, text, match.start(), match.end(), "IFSC", 0.98)


def _trim_person(raw: str) -> str:
    kept: list[str] = []
    for token in raw.split():
        clean = token.strip(".,;:!?()[]{}\"'\u0964\u0965")
        key = clean.lower()
        if not clean or len(clean) < 2:
            break
        if key in _NAME_STOP or key in _NAME_BLOCK:
            break
        kept.append(clean)
    return " ".join(kept)


def _add_person(dets: list[Detection], text: str, match: re.Match) -> None:
    raw = match.group(1)
    value = _trim_person(raw)
    if not value:
        return
    offset = raw.find(value)
    if offset < 0:
        return
    start = match.start(1) + offset
    _add(dets, text, start, start + len(value), "PERSON", 0.7)


def _detect_names(text: str, norm: str, dets: list[Detection]) -> None:
    for pattern in (_LATIN_NAME_ANCHOR_RE, _LATIN_HONORIFIC_RE, _LATIN_JI_RE,
                    _HINDI_NAME_ANCHOR_RE, _HINDI_HONORIFIC_RE, _HINDI_JI_RE):
        for match in pattern.finditer(norm):
            _add_person(dets, text, match)


def _resolve(dets: Iterable[Detection]) -> tuple[Detection, ...]:
    accepted: list[Detection] = []
    for det in sorted(dets, key=lambda d: (-PRIORITY[d.type], -(d.end - d.start), d.start)):
        if any(not (det.end <= kept.start or det.start >= kept.end) for kept in accepted):
            continue
        accepted.append(det)
    return tuple(sorted(accepted, key=lambda d: (d.start, d.end)))


def detect(
    text: str,
    policy: str = "balanced",
    enabled_types: Sequence[str] | None = None,
) -> tuple[Detection, ...]:
    """Detect PII spans in *text*.

    Parameters
    ----------
    text:
        English, Hindi (Devanagari) or Hinglish/Roman-Hindi text.
    policy:
        ``"balanced"`` (default) accepts a standalone 10-digit mobile-looking
        number with no context.  ``"strict"`` requires a phone cue, +91/00
        prefix, separator grouping or an international prefix.
    enabled_types:
        Optional subset of :data:`ENTITY_TYPES`.
    """
    if policy not in ("balanced", "strict"):
        raise ValueError("policy must be 'balanced' or 'strict'")
    if not isinstance(text, str) or not text:
        return ()
    if enabled_types is not None:
        unknown = set(enabled_types) - set(ENTITY_TYPES)
        if unknown:
            raise ValueError(f"unknown entity types: {sorted(unknown)}")

    norm = normalize_digits(text)
    dets: list[Detection] = []
    _detect_aadhaar(text, norm, dets)
    _detect_bank(text, norm, dets)
    _detect_phones(text, norm, policy, dets)
    _detect_email(text, norm, dets)
    _detect_url(text, norm, dets)
    _detect_ifsc(text, norm, dets)
    _detect_names(text, norm, dets)

    resolved = _resolve(dets)
    if enabled_types is not None:
        allowed = set(enabled_types)
        resolved = tuple(d for d in resolved if d.type in allowed)
    return resolved


def detect_dicts(text: str, policy: str = "balanced", enabled_types: Sequence[str] | None = None) -> list[dict]:
    """Convenience wrapper returning JSON-serialisable dictionaries."""
    return [d.as_dict() for d in detect(text, policy=policy, enabled_types=enabled_types)]
