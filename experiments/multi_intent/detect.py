"""Risk-first multi-intent detection (v2).

Supersedes :mod:`decompose` (which is kept unchanged as the v1 baseline). The
flow matches the required architecture:

    normalise -> identify multi-intent -> decompose sub-queries ->
    preserve entities -> assess risk per sub-query -> overall safety decision

Key safety properties:

* a sub-query is kept even when it has no question word but does carry
  agricultural content (the v1 decomposer silently dropped such pest statements);
* risk is assessed **per sub-query and again on the whole query**, so a
  decomposition or classification miss cannot hide a dangerous request;
* any high/medium-risk sub-query escalates the whole query — a safe sub-query can
  never override a dangerous one;
* entities that are not present are reported as ``"unknown"``, never invented.

The rule-based "confidence" is a heuristic signal strength (0.9 strong / 0.7 weak
/ 0.3 unknown), not a calibrated probability.
"""
from __future__ import annotations

import re
import sys
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT / "src", ROOT / "experiments"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from weak_labels import label_query as _english_label  # noqa: E402

from answer_screening.screening import extract_measurements  # noqa: E402
from multi_intent.risk import CROP, PEST_DISEASE, PRICE, Risk, assess_risk  # noqa: E402

# ------------------------------------------------------------------ lexicon
CUE_LEXICON: dict[str, list[str]] = {
    "Market Price": ["bhav", "bhaav", "mandi", "rate", "msp", "price", "market", "sell", "selling",
                     "bech", "bechna", "bikri", "भाव", "मंडी", "कीमत", "दाम", "बेच", "बिक्री"],
    "Nutrient Management": ["urea", "khaad", "khad", "dap", "ssp", "mop", "potash", "nitrogen",
                            "fertiliser", "fertilizer", "manure", "zinc", "यूरिया", "खाद", "उर्वरक", "डीएपी"],
    "Plant Protection": ["dawa", "keeda", "kira", "kida", "keet", "spray", "rog", "ratua", "pest",
                         "insect", "aphid", "borer", "rust", "blight", "wilt", "termite", "disease",
                         "रोग", "कीट", "कीड़ा", "रतुआ", "कीटनाशक", "दवा", "छिड़काव"],
    "Weather": ["mausam", "baarish", "barish", "weather", "rain", "rainfall", "forecast",
                "temperature", "बारिश", "मौसम", "बरसात"],
    "Varietal Information": ["beej", "variety", "varieties", "kism", "seed", "hybrid", "cultivar",
                             "बीज", "किस्म"],
    "Animal Husbandry": ["gaay", "bhains", "bakri", "cow", "buffalo", "goat", "pashu", "cattle",
                         "doodh", "milk", "poultry", "livestock", "गाय", "भैंस", "पशु", "दूध"],
    "Fisheries": ["machhli", "fish", "talab", "pond", "fishery", "मछली", "तालाब"],
    "Government Scheme": ["yojana", "scheme", "subsidy", "loan", "insurance", "pm-kisan", "pmfby",
                          "बीमा", "योजना", "सब्सिडी", "लोन"],
    "Cultural Practices": ["sinchai", "irrigation", "irrigate", "sowing", "sow", "buwai", "plough",
                           "ploughing", "transplant", "transplanting", "harvest", "harvesting",
                           "store", "storage", "spacing", "tillage", "सिंचाई", "बुवाई", "जुताई", "भंडारण"],
}
_CUE_RE = {label: re.compile(r"(?<![A-Za-z])(" + "|".join(re.escape(c) for c in cues) + r")(?![A-Za-z])",
                             re.IGNORECASE)
           for label, cues in CUE_LEXICON.items()}

_NON_AGRI = re.compile(
    r"\b(?:python|program|code|blockchain|capital of|mutual fund|interest rate|fixed deposit|"
    r"cricket|movie|award|prime minister|passport|train|ticket|recipe|poem|laptop|savings account|"
    r"boiling point)\b", re.IGNORECASE)
_REQUEST = re.compile(
    r"\b(?:what|which|how|when|where|why|who|tell|explain|write|book|give|show|list|suggest|"
    r"can|could|should|is|are|do|does|will|kya|kaise|kab|kahan|kitna|kitni|kitne|batao|bataen|"
    r"bataiye|chahiye|kaunsa|kaunsi|kaun|karo|karein)\b", re.IGNORECASE)
_AGRI_CONTENT = re.compile(
    PRICE.pattern + "|" + CROP.pattern + "|" + PEST_DISEASE.pattern
    + r"|(?:urea|dap|fertiliser|fertilizer|khaad|dawa|spray|sinchai|irrigation|beej|variety|"
    r"fish|machhli|talab|pond|doodh|milk|yojana|scheme|mausam|weather|भाव|मंडी|खाद|बीज)",
    re.IGNORECASE)

_SENT_END = re.compile(r"[?।;\n]+")
_CONTRAST = re.compile(
    r"\b(?:but|however|instead|although|though|whereas|yet|otherwise|so|therefore|hence|thus|"
    r"lekin|magar|parantu|बल्कि|लेकिन|इसलिए)\b", re.IGNORECASE)
_COORD = re.compile(r"\b(?:and\s+also|and|aur|plus|also|as\s+well\s+as|तथा|और)\b", re.IGNORECASE)
_APP_VERB = re.compile(r"\b(?:spray|spraying|apply|mix|dalen|dalna|dal|karein|milaye|chhidkav)\b|"
                       r"(?:छिड़क|डाल|मिला)", re.IGNORECASE)


@dataclass
class SubQuery:
    text: str
    start: int
    end: int
    intent: str
    confidence: float
    crop: str
    pest: str
    chemical: str
    quantity: str
    unit: str
    context: str
    risk: Risk
    escalation_reason: str = ""

    @property
    def escalate(self) -> bool:
        return self.risk.escalate


@dataclass
class Decision:
    input_text: str
    is_multi: bool
    sub_queries: list[SubQuery]
    distinct_intents: list[str]
    escalate: bool
    escalation_reasons: list[str] = field(default_factory=list)
    whole_query_risk: Risk | None = None
    reason: str = ""


def _normalise(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    return re.sub(r"\s+", " ", text.replace("\u00a0", " ")).strip()


def _label(segment: str) -> tuple[str, float]:
    """Rule-based intent + heuristic confidence for one sub-query."""
    if _NON_AGRI.search(segment):
        return "Other", 0.9
    if PRICE.search(segment):
        # A price/market request is Market Price (time-sensitive) whatever is priced.
        return "Market Price", 0.8
    english = _english_label(segment)
    counts = {label: len(rx.findall(segment)) for label, rx in _CUE_RE.items()}
    hits = sum(counts.values())
    if english != "Other":
        return english, (0.9 if hits >= 2 else 0.75)
    counts = {k: v for k, v in counts.items() if v}
    if not counts:
        return "Other", 0.3
    top = max(counts.items(), key=lambda kv: kv[1])[0]
    return top, (0.8 if counts[top] >= 2 else 0.6)


def _split(text: str) -> list[tuple[str, int]]:
    pieces: list[str] = []
    for chunk in _SENT_END.split(text):
        for sub in _CONTRAST.split(chunk):
            pieces.extend(sub.split(","))
    out: list[tuple[str, int]] = []
    # Recover offsets approximately by searching the original text.
    cursor = 0
    for piece in pieces:
        for part in _coord_split(piece):
            stripped = part.strip()
            if not stripped:
                continue
            pos = text.find(stripped, cursor)
            if pos < 0:
                pos = text.find(stripped)
            if pos < 0:
                pos = cursor
            out.append((stripped, pos))
            cursor = pos + len(stripped)
    return out


def _coord_split(piece: str) -> list[str]:
    """Split on coordinating conjunctions. For multi-intent decomposition we split
    on every conjunction (more sub-queries only ever adds risk signals), unlike the
    negation screener which must keep a negated list together."""
    parts, last = [], 0
    for m in _COORD.finditer(piece):
        parts.append(piece[last:m.start()])
        last = m.end()
    parts.append(piece[last:])
    return [p.strip() for p in parts]


def _keep(segment: str) -> bool:
    """Keep a segment if it is a request OR carries agricultural content."""
    return bool(_REQUEST.search(segment) or _AGRI_CONTENT.search(segment))


def _entities(segment: str) -> tuple[str, str, str, str, str]:
    crop_m = CROP.search(segment)
    pest_m = PEST_DISEASE.search(segment)
    measures = extract_measurements(segment)
    quantity = unit = "unknown"
    if measures:
        m = measures[0]
        quantity = m.raw.split()[0] if m.raw else "unknown"
        unit = m.unit
    chem = "unknown"
    from answer_screening.screening import screen_answer  # local import to avoid cycles

    result = screen_answer(segment)
    if result.risky and result.kind in ("chemical", "dosage", "measurement"):
        chem = result.clause or "unknown"
    return (crop_m.group(1).lower() if crop_m else "unknown",
            pest_m.group(1).lower() if pest_m else "unknown",
            chem, quantity, unit)


def detect(text: str | None) -> Decision:
    """Risk-first multi-intent decision. Never raises on bad input."""
    if not isinstance(text, str) or not text.strip():
        return Decision(text or "", False, [], [], False, reason="empty input")
    normalised = _normalise(text)

    subs: list[SubQuery] = []
    for segment, start in _split(normalised):
        if not _keep(segment):
            continue
        intent, confidence = _label(segment)
        crop, pest, chem, qty, unit = _entities(segment)
        risk = assess_risk(segment, intent)
        subs.append(SubQuery(segment, start, start + len(segment), intent, confidence,
                             crop, pest, chem, qty, unit,
                             context=(crop if crop != "unknown" else "unknown"),
                             risk=risk,
                             escalation_reason="; ".join(risk.reasons) if risk.escalate else ""))

    if not subs:  # nothing decomposed: fall back to the whole query
        intent, confidence = _label(normalised)
        crop, pest, chem, qty, unit = _entities(normalised)
        risk = assess_risk(normalised, intent)
        subs = [SubQuery(normalised, 0, len(normalised), intent, confidence, crop, pest, chem,
                         qty, unit, crop, risk, "; ".join(risk.reasons) if risk.escalate else "")]

    # Whole-query safety net: risk cannot be hidden by a bad decomposition.
    whole_intent, _ = _label(normalised)
    whole_risk = assess_risk(normalised, whole_intent)

    distinct = [l for l in CUE_LEXICON if any(s.intent == l for s in subs)]
    if any(s.intent == "Other" for s in subs):
        distinct.append("Other")
    risky = [s for s in subs if s.escalate]
    escalate = bool(risky) or whole_risk.escalate
    reasons = []
    for s in risky:
        reasons.append(f"[{s.intent}] {s.text!r}: {s.escalation_reason}")
    if whole_risk.escalate and not risky:
        reasons.append(f"[whole query] {whole_intent}: {'; '.join(whole_risk.reasons)}")
    return Decision(
        input_text=text,
        is_multi=len(set(s.intent for s in subs)) >= 2 or len(subs) >= 2,
        sub_queries=subs,
        distinct_intents=distinct,
        escalate=escalate,
        escalation_reasons=reasons,
        whole_query_risk=whole_risk,
        reason="risk-flagged" if escalate else "no risk signal",
    )


def intent_set(text: str | None) -> tuple[str, ...]:
    return tuple(detect(text).distinct_intents)
