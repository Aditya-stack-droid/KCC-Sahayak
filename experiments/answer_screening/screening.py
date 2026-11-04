"""Negation- and measurement-aware answer-risk screening (experimental).

This module is **additive**: it does not modify the shipped rule in
``src/confidence_gate.py``. It provides a second-generation content screener for
two measured weaknesses of ``answer_risk``:

1. **Negation blind spot.** ``"Do not spray chlorpyrifos."`` is flagged risky
   because the regex sees the product and the verb but ignores the prohibition.
   Ten such false positives were recorded by the Improvement 5 behavioural
   audit (cases C01, C02, C03, C04, C05, C06, C08, C10, C11, C13).

2. **Measurement blind spot.** Dosages written with containers or household
   measures — ``"one capful per litre"``, ``"two spoonfuls per tank"``,
   ``"1 sachet per 20 litre"``, ``"2 bori per acre"`` — are not matched by the
   shipped numeric+metric regex and pass unscreened.

Design rules (deliberately conservative, because this is safety code):

* **Default verdict is RISKY.** A risk span is suppressed only when a negation
  cue clearly governs it inside the same clause.
* **Negation is clause-local and window-bounded.** An un-negated clause is still
  screened even when a neighbouring clause is negated, and a cue only reaches a
  few tokens, so ``"Do not spray A, instead spray B at 5 ml/litre"`` still flags
  the second clause.
* **No unit is ever converted.** Household/container measures are reported as an
  *unrecognised measurement in a possible instruction* and escalated; the module
  never invents an ml/g equivalent (``Measurement.converted`` is always ``None``).
* **Ambiguous text stays risky.** Questions and hedged statements are not
  suppressed; they escalate.

The production rule is imported **read-only** and reused as the base vocabulary,
so the measured difference is attributable to to the negation/measurement layer
rather than to unrelated rewording.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

# Reuse the shipped regexes read-only. Prefixing the production patterns makes the
# comparison against ``answer_risk`` attribute the change to this layer, not to a
# rewritten vocabulary.
try:  # pragma: no cover - exercised through the experiment's sys.path setup
    from confidence_gate import _CHEMICALS as _PROD_CHEM, _DOSE as _PROD_DOSE
except Exception:  # pragma: no cover
    import sys
    from pathlib import Path

    _SRC = Path(__file__).resolve().parents[2] / "src"
    if str(_SRC) not in sys.path:
        sys.path.insert(0, str(_SRC))
    from confidence_gate import _CHEMICALS as _PROD_CHEM, _DOSE as _PROD_DOSE

# --------------------------------------------------------------------------- #
# Vocabulary
# --------------------------------------------------------------------------- #
_DEV_DIGITS = "०१२३४५६७८९"
_NUMWORD = (
    r"(?:a|an|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|half|quarter|"
    r"aadha|aadhi|aadhe|adha|paav|pao|sawa|dedh|ek|do|dono|teen|char|chaar|"
    r"paanch|panch|cheh|chhe|saat|aath|nau|das|"
    r"एक|दो|तीन|चार|पांच|पाँच|छह|सात|आठ|नौ|दस|आधा|सवा|डेढ़)"
)
_NUMTOK = rf"(?:\d+(?:[.,]\d+)?|[{_DEV_DIGITS}]+|{_NUMWORD})"

_UNIT_METRIC = (
    r"(?:ml|mls|millilitres?|milliliters?|cc|gms?|gm|grams?|kgs?|kilos?|kilograms?|"
    r"ltr|litres?|liters?|lit|q|quintals?|ppm|percent|मिली|मिलीलीटर|लीटर|ग्राम|किलो|किलोग्राम)"
)
_UNIT_HOUSEHOLD = (
    r"(?:chammach|chamach|spoons?|spoonfuls?|teaspoons?|tsp|tablespoons?|tbsp|"
    r"cupfuls?|cups?|capfuls?|caps?|glassfuls?|glasses?|handfuls?|mutthi|mutti|lotahs?|"
    r"bottles?|bung|बूंदें?|बूँदें?|चम्मच|ग्लास|प्याली)"
)
_UNIT_PACK = (
    r"(?:packets?|sachets?|pouches?|packs?|boris?|bags?|tins?|boxes|box|strips?|vials?|"
    r"ampoules?|पैकेट|बोरी|डिब्बा|पुड़िया)"
)
_UNIT_TANK = r"(?:tanks?|pumps?|drums?|barrels?|sprayers?|containers?|टैंक)"
_UNIT_AREA = r"(?:acres?|hectares?|ha|bighas?|kanals?|gunthas?|marla|cents?|kathas?|ropani|एकड़|हेक्टेयर|बीघा)"

_UNIT_ANY = rf"(?:{_UNIT_METRIC}|{_UNIT_HOUSEHOLD}|{_UNIT_PACK}|{_UNIT_TANK}|{_UNIT_AREA})"

# Measurement: a number token, optional filler, then a unit token.
# Devanagari word-final vowel signs are combining marks, so a trailing ``\b``
# after e.g. "किलो" or "मिली" never matches. ``(?![A-Za-z0-9])`` is the portable
# equivalent that still rejects "kg" inside "kgs" and "litre" inside "litres".
_END = r"(?![A-Za-z0-9])"
_MEASURE = re.compile(
    rf"(?<![\w]){_NUMTOK}(?:\s+(?:a|an|of|the|full|का|की))*\s*({_UNIT_ANY}){_END}",
    re.IGNORECASE,
)

# Base dosage rule = the production dose regex, byte-identical.
_PROD_DOSE_RE = _PROD_DOSE

# Plural-tolerant metric unit (the shipped regex requires "kg"/"litre" exactly and
# misses "kgs", "litres", "grams"). Digits use it unconditionally; number words
# only count with an agricultural context, to avoid flagging "one litre of water".
_METRIC_UNIT_EXT = (
    r"(?:mls?|millilitres?|milliliters?|cc|gms?|grs?|grams?|kgs?|kilos?|kilograms?|"
    r"ltrs?|litres?|liters?|quintals?|quis?|ppm|percent|मिली|लीटर|ग्राम|किलो|किलोग्राम)"
)
_DOSE_DIGITS = re.compile(rf"(?<![\w])(?:\d+(?:[.,]\d+)?|[०-९]+)\s*{_METRIC_UNIT_EXT}{_END}", re.IGNORECASE)
_DOSE_WORDS = re.compile(rf"(?<![\w]){_NUMWORD}\s*{_METRIC_UNIT_EXT}{_END}", re.IGNORECASE)
# Extended chemical vocabulary: shipped pattern + common brands + Devanagari names.
_EXTRA_CHEM = re.compile(
    r"\b(?:confidor|dithane(?:\s*m-?45)?|bavistin|carbofuran|3g|"
    r"mono-?crotophos|m0n0crotophos|r0gor|rogor|chlorpyriphos|coragen|"
    r"actara|karate|thiodan|furadan|phorate|tricel|tilt|amistar|score|"
    r"chemicals?)\b"
    r"|(?:यूरिया|डीएपी|एसएसपी|एमओपी|पोटाश|खाद|कीटनाशक|कीटनाशी|दवा|दवाइयाँ|फफूंदीनाशक|"
    r"रोगोर|मोनोक्रोटोफॉस|क्लोरपायरीफॉस|इमिडाक्लोप्रिड|मैंकोजेब|कार्बेन्डाजिम|सल्फर|जिंक|"
    r"ज़िंक|खरपतवारनाशी|जीवाणुनाशक|नाइट्रोजन|यूरिया)",
    re.IGNORECASE,
)

# Application / instruction verbs (Latin + Roman-Hindi + Devanagari).
_APP_VERB = re.compile(
    r"(?:\bspray(?:ing|ed|s)?\b|\bsprey\b|\bdrench(?:ing)?\b|\bapply(?:ing|ied)?\b|"
    r"\bmix(?:ing|ed)?\b|\bdust(?:ing)?\b|\bsprinkl\w*\b|\binject\w*\b|\btreat\w*\b|"
    r"\badd(?:ing)?\b|\bpour(?:ing)?\b|\bdal(?:en|na|o|ein|e)?\b|\bdaal(?:na|o|ein)?\b|"
    r"\bmilaye(?:n)?\b|\bmilakar\b|\bchhidkav\b|\bchhidkao\b|"
    r"छिड़क\w*|डाल\w*|मिला\w*|प्रयोग\s*कर\w*)",
    re.IGNORECASE,
)
_DENOMINATOR = re.compile(
    r"\bper\b|\bprati\b|प्रति|/\s*(?:litre|liter|acre|ha|bigha|tank|plant|pump)",
    re.IGNORECASE,
)

# --------------------------------------------------------------------------- #
# Negation
# --------------------------------------------------------------------------- #
_DEV_B = r"(?<![ऀ-ॿ])"   # start of a Devanagari word
_DEV_A = r"(?![ऀ-ॿ])"    # end of a Devanagari word (matra-safe: no trailing \b)
_CUES = re.compile(
    r"(?:\bdo\s+not\b|\bdon'?t\b|\bdoes\s+not\b|\bdid\s+not\b|\bdo\s+nt\b|\bnever\b|"
    r"\bavoid\w*\b|\bwithout\b|\bnot\s+to\b|\bno\s+need\b|\bno\s+requirement\b|"
    r"\brefrain\w*\b|\bstop\b|\bmust\s+not\b|\bshould\s+not\b|\bshouldn'?t\b|"
    r"\bcan'?t\b|\bcannot\b|\b(?:is|are|was|were|be)\s+not\b|"
    r"\bnot\s+(?:recommended|advised|needed|required|necessary|suggested|allowed|permitted|effective)\b|"
    r"\bmat\b|\bnahi\b|\bnahin\b|\bna\b|\bbina\b|\bno\b|"
    rf"{_DEV_B}मत{_DEV_A}|{_DEV_B}नहीं{_DEV_A}|{_DEV_B}बिना{_DEV_A}|{_DEV_B}ना{_DEV_A}|{_DEV_B}न\s+कर)",
    re.IGNORECASE,
)
_DOUBLE_NEG = re.compile(
    r"\b(?:not|don'?t|does\s+not|do\s+not|never)\s+(?:not|never)\b", re.IGNORECASE
)
_TRAILING_PROHIBITION = re.compile(
    r"(?:\bmat\s+(?:karein|karo|karen|dalo|dalna|dalen|karna|karvayein)\b|"
    rf"\bmat\b|\bnahi\b|\bnahin\b|{_DEV_B}मत{_DEV_A}|{_DEV_B}नहीं{_DEV_A}|{_DEV_B}न\s+कर)",
    re.IGNORECASE,
)

_WINDOW_BEFORE = 8
_WINDOW_AFTER = 4

# Clause boundaries. Sentence terminators, contrastive and result conjunctions,
# and commas always split; coordinating conjunctions ("and"/"or") split only when
# a new application verb follows, so "do not spray A and B" stays one negated
# clause while "do not spray A and apply B" splits and screens the second half.
# Splitting at result conjunctions ("so", "therefore") keeps a negation in one
# clause from suppressing an instruction in the next ("pest is not controlled so
# spray 2 ml").
_SENT_END = re.compile(r"[.;!?।\n]+")
_CONTRAST = re.compile(
    r"\b(?:but|however|instead|although|though|whereas|yet|otherwise|"
    r"so|therefore|hence|thus|consequently|isliye|"
    r"lekin|magar|parantu|बल्कि|परन्तु|लेकिन|इसलिए)\b",
    re.IGNORECASE,
)
_COORD = re.compile(r"\b(?:and|or|aur|ya)\b", re.IGNORECASE)


def _split_clauses(text: str) -> list[str]:
    pieces: list[str] = []
    for chunk in _SENT_END.split(text):
        for sub in _CONTRAST.split(chunk):
            pieces.extend(sub.split(","))
    out: list[str] = []
    for piece in pieces:
        parts: list[str] = []
        last = 0
        for m in _COORD.finditer(piece):
            after = piece[m.end(): m.end() + 40]
            if _APP_VERB.search(after):
                parts.append(piece[last:m.start()])
                last = m.end()
        parts.append(piece[last:])
        out.extend(p.strip() for p in parts)
    return [p for p in out if p]


def _is_negated(clause: str, start: int, end: int) -> bool:
    pre = " ".join(clause[:start].split()[-_WINDOW_BEFORE:])
    post = " ".join(clause[end:].split()[:_WINDOW_AFTER])
    if _DOUBLE_NEG.search(pre) or _DOUBLE_NEG.search(post):
        return False  # double negative resolves to a positive instruction
    if _CUES.search(pre) or _CUES.search(post):
        return True
    return bool(_TRAILING_PROHIBITION.search(clause))


# --------------------------------------------------------------------------- #
# Structured measurements
# --------------------------------------------------------------------------- #
_NUMWORD_VALUE = {
    "one": 1.0, "ek": 1.0, "एक": 1.0, "a": 1.0,
    "two": 2.0, "do": 2.0, "दो": 2.0,
    "three": 3.0, "teen": 3.0, "तीन": 3.0,
    "four": 4.0, "char": 4.0, "chaar": 4.0, "चार": 4.0,
    "five": 5.0, "paanch": 5.0, "panch": 5.0, "पांच": 5.0, "पाँच": 5.0,
    "six": 6.0, "cheh": 6.0, "chhe": 6.0, "छह": 6.0,
    "seven": 7.0, "saat": 7.0, "सात": 7.0,
    "eight": 8.0, "aath": 8.0, "आठ": 8.0,
    "nine": 9.0, "nau": 9.0, "नौ": 9.0,
    "ten": 10.0, "das": 10.0, "दस": 10.0,
    "half": 0.5, "aadha": 0.5, "aadhi": 0.5, "adha": 0.5, "आधा": 0.5,
    "quarter": 0.25, "paav": 0.25, "pao": 0.25,
}
_UNIT_CLASS: list[tuple[re.Pattern, str]] = [
    (re.compile(_UNIT_METRIC, re.IGNORECASE), "metric"),
    (re.compile(_UNIT_HOUSEHOLD, re.IGNORECASE), "household"),
    (re.compile(_UNIT_PACK, re.IGNORECASE), "packaging"),
    (re.compile(_UNIT_TANK, re.IGNORECASE), "container"),
    (re.compile(_UNIT_AREA, re.IGNORECASE), "area"),
]


def _quantity_value(token: str) -> float | None:
    token = token.strip()
    if not token:
        return None
    if token.isdigit():
        return float(token)
    if all(c in _DEV_DIGITS for c in token):
        return float("".join(str(_DEV_DIGITS.index(c)) for c in token))
    try:
        return float(token.replace(",", "."))
    except ValueError:
        return _NUMWORD_VALUE.get(token.lower())


def _unit_class(unit: str) -> str:
    for pattern, name in _UNIT_CLASS:
        if pattern.fullmatch(unit):
            return name
    if unit in _DEV_DIGITS:  # defensive
        return "unknown"
    return "unknown"


@dataclass(frozen=True)
class Measurement:
    """A quantity + unit expression, kept structured and **never converted**."""

    raw: str
    quantity: str
    quantity_value: float | None
    unit: str
    unit_class: str
    denominator: str | None
    language: str
    converted: None = None  # invariant: no ml/g conversion is ever produced


def extract_measurements(text: str, language_hint: str | None = None) -> list[Measurement]:
    """Return structured quantity/unit expressions (no conversion attempted)."""
    if not isinstance(text, str) or not text.strip():
        return []
    normalised = _normalise(text)
    lang = language_hint or detect_language(normalised)
    out: list[Measurement] = []
    for m in _MEASURE.finditer(normalised):
        unit = m.group(1)
        quantity = normalised[m.start(): m.end() - len(unit)].strip()
        denom_match = re.search(
            r"\b(?:per|prati)\s+(\w+)|प्रति\s+(\w+)", normalised[m.end(): m.end() + 40], re.IGNORECASE
        )
        denominator = None
        if denom_match:
            denominator = (denom_match.group(1) or denom_match.group(2)).lower()
        out.append(Measurement(
            raw=m.group(0).strip(),
            quantity=quantity,
            quantity_value=_quantity_value(quantity),
            unit=unit.lower(),
            unit_class=_unit_class(unit),
            denominator=denominator,
            language=lang,
        ))
    return out


# --------------------------------------------------------------------------- #
# Language and normalisation
# --------------------------------------------------------------------------- #
_DEVANAGARI = re.compile(r"[ऀ-ॿ]")
_ROMAN_HI = re.compile(
    r"\b(?:dawa|dal(?:en|na|o)?|karein|karo|mat|nahi|pani|khet|fasal|gehu|bhav|"
    r"chhidkav|milaye|prati|boond|chammach|bori|packet)\b",
    re.IGNORECASE,
)


def detect_language(text: str) -> str:
    letters = [c for c in text if c.isalpha()]
    if letters and sum(1 for c in letters if _DEVANAGARI.match(c)) / len(letters) > 0.25:
        return "devanagari"
    if _ROMAN_HI.search(text):
        return "roman_hindi"
    return "english"


def _normalise(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\u00a0", " ")
    return re.sub(r"\s+", " ", text).strip()


# --------------------------------------------------------------------------- #
# Screener
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ScreenResult:
    risky: bool
    reason: str | None = None
    kind: str | None = None          # dosage | measurement | chemical
    clause: str | None = None
    language: str = "english"
    negated: bool = False


_DOSAGE_REASON = "dosage/quantity instruction"
_MEASUREMENT_REASON = "unrecognised container/household measurement in a possible application"
_CHEMICAL_REASON = "chemical or spray recommendation"


def _measurement_is_agricultural(clause: str) -> bool:
    """A household/container/word-number measure is only risky with agri context."""
    return bool(
        _DENOMINATOR.search(clause)
        or _APP_VERB.search(clause)
        or _PROD_CHEM.search(clause)
        or _EXTRA_CHEM.search(clause)
    )


def screen_answer(text: str | None, language_hint: str | None = None) -> ScreenResult:
    """Content-screen an answer text. RISKY unless a negation clearly covers it."""
    if not isinstance(text, str) or not text.strip():
        return ScreenResult(False)
    normalised = _normalise(text)
    lang = language_hint or detect_language(normalised)
    clauses = _split_clauses(normalised)

    def unnegated(matches, clause: str):
        for m in matches:
            if not _is_negated(clause, m.start(), m.end()):
                return True
        return False

    # 1. explicit numeric dosage
    for clause in clauses:
        # (a) shipped rule verbatim (keeps "@ 2" and "per bigha" forms)
        if unnegated(_PROD_DOSE_RE.finditer(clause), clause):
            return ScreenResult(True, _DOSAGE_REASON, "dosage", clause, lang)
        # (b) digits (Latin or Devanagari) + plural/Devanagari metric units
        if unnegated(_DOSE_DIGITS.finditer(clause), clause):
            return ScreenResult(True, _DOSAGE_REASON, "dosage", clause, lang)
        # (c) number words + metric units, only with an agricultural context
        if _measurement_is_agricultural(clause):
            for m in _DOSE_WORDS.finditer(clause):
                if not _is_negated(clause, m.start(), m.end()):
                    return ScreenResult(True, _DOSAGE_REASON, "dosage", clause, lang)

    # 2. household / container / packaging / area measurements with a quantity
    for clause in clauses:
        if not _measurement_is_agricultural(clause):
            continue
        for m in _MEASURE.finditer(clause):
            unit = m.group(1)
            if _unit_class(unit) == "metric":
                continue  # handled by the dosage rule above
            if not _is_negated(clause, m.start(), m.end()):
                return ScreenResult(True, _MEASUREMENT_REASON, "measurement", clause, lang)

    # 3. chemical / spray recommendation (shipped vocabulary + brands + Devanagari)
    for clause in clauses:
        if unnegated(_PROD_CHEM.finditer(clause), clause):
            return ScreenResult(True, _CHEMICAL_REASON, "chemical", clause, lang)
        if unnegated(_EXTRA_CHEM.finditer(clause), clause):
            return ScreenResult(True, _CHEMICAL_REASON, "chemical", clause, lang)

    return ScreenResult(False, language=lang)


def risk_reason(text: str | None) -> str | None:
    """Drop-in analogue of ``confidence_gate.answer_risk`` (risk string or None)."""
    result = screen_answer(text)
    return result.reason if result.risky else None
