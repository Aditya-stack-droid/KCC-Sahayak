"""Risk assessment for agricultural sub-queries (multilingual, conservative).

The v1 multi-intent decomposer treated every intent equally and derived safety
only from the classified intent, so a classification miss silently removed the
risk signal. This module adds an independent, multi-signal risk layer that runs
**per sub-query and again on the whole query**, so a decomposition or
classification failure cannot hide a dangerous request.

Signals (any one of the first four ⇒ ``high``):

* intent in the high-stakes or time-sensitive sets (``src/glossary``);
* a chemical / dosage / measurement instruction (reuses the answer screener);
* a pest/disease term together with a treatment/control request;
* a toxicity / exposure / harvest-restriction question.

Uncertainty escalates: an ``Other`` sub-query that still carries agricultural
signals is ``medium`` (escalate), never auto-approved.
"""
from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT / "src", ROOT / "experiments"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from glossary import HIGH_STAKES_TYPES, TIME_SENSITIVE_TYPES  # noqa: E402

from answer_screening.screening import risk_reason as _chemical_dose  # noqa: E402

HIGH_RISK_INTENTS = set(HIGH_STAKES_TYPES) | set(TIME_SENSITIVE_TYPES)

# ------------------------------------------------------------------ lexicons
_DEV = r"ऀ-ॿ"


def _word(*terms: str) -> re.Pattern:
    return re.compile("(?<![A-Za-z])(" + "|".join(re.escape(t) for t in terms) + r")(?![A-Za-z])", re.IGNORECASE)


PEST_DISEASE = _word(
    "rust", "ratua", "ratuwa", "blight", "wilt", "smut", "blast", "mildew", "rot",
    "aphid", "aphids", "whitefly", "borer", "bollworm", "caterpillar", "termite",
    "thrips", "mite", "hopper", "keeda", "kira", "kida", "keet",
    "beetle", "weevil", "moth", "bug", "larva", "maggot", "cutworm", "girdle",
    "रतुआ", "रोग", "कीट", "कीड़ा", "कीड़े", "इल्ली", "माहू", "फफूंद", "सड़न",
    "pest", "disease", "insect",
)
CROP = _word(
    "wheat", "paddy", "rice", "gehu", "gehun", "dhan", "mustard", "sarson", "cotton",
    "kapas", "bajra", "maize", "tomato", "brinjal", "sugarcane", "gram", "chana",
    "गेहूं", "धान", "सरसों", "कपास", "टमाटर", "बाजरा", "गन्ना", "मक्का",
)
TREATMENT_REQUEST = re.compile(
    r"\b(?:control|treatment|treat|cure|remedy|recommend|suggest|spray|sprayed|spraying|sprey|"
    r"apply|use|using|management|ilaj|upay|dawa|chhidkav|nash|hataye|kya\s+karein|kya\s+kare|"
    r"kya\s+karun|how\s+to\s+control|get\s+rid)\b"
    r"|(?:उपाय|इलाज|दवा|छिड़काव|नियंत्रण|उपचार|क्या\s*करें|क्या\s*करूँ)",
    re.IGNORECASE,
)
# Generic agricultural context (used only for the "unknown intent" medium rule).
GENERIC_AGRI = re.compile(
    r"\b(?:crop|crops|fasal|fasal|kheti|khet|farm|field|standing\s+crop)\b|(?:फसल|खेत|खेती)",
    re.IGNORECASE,
)
TOXICITY = re.compile(
    r"\b(?:toxic|toxicity|poison|poisonous|zeher|jahar|harmful|safe\s+for\s+(?:humans|cattle|animals)|"
    r"waiting\s+period|residual|residue|harvest\s+(?:after|before|restriction|interval)|"
    r"eat\s+after|खाने\s*से|जहर|जहरीला|प्रतीक्षा\s*अवधि|अवशेष)\b|"
    r"(?:जहर|जहरीला|अवशेष|प्रतीक्षा)",
    re.IGNORECASE,
)
PRICE = re.compile(
    r"\b(?:price|bhav|bhaav|rate|mandi|msp|market|sell|selling|sale|bech|bechna|bikri|"
    r"procurement|quintal\s+rate)\b|(?:भाव|मंडी|कीमत|दाम|बेच|बिक्री|मूल्य)",
    re.IGNORECASE,
)


@dataclass
class Risk:
    level: str                       # "high" | "medium" | "low"
    reasons: list[str] = field(default_factory=list)
    signals: dict = field(default_factory=dict)

    @property
    def escalate(self) -> bool:
        return self.level in ("high", "medium")


def assess_risk(text: str, intent: str | None = None) -> Risk:
    """Multi-signal risk for one (sub-)query. Conservative: uncertain ⇒ escalate."""
    if not isinstance(text, str) or not text.strip():
        return Risk("low", ["empty"])
    t = text
    reasons: list[str] = []
    signals = {
        "intent": intent,
        "pest_disease": bool(PEST_DISEASE.search(t)),
        "crop": bool(CROP.search(t)),
        "generic_agri": bool(GENERIC_AGRI.search(t)),
        "treatment_request": bool(TREATMENT_REQUEST.search(t)),
        "chemical_or_dose": _chemical_dose(t) is not None,
        "toxicity": bool(TOXICITY.search(t)),
        "price": bool(PRICE.search(t)),
    }

    level = "low"
    if intent in HIGH_STAKES_TYPES:
        level = "high"
        reasons.append(f"high-stakes intent '{intent}'")
    if intent in TIME_SENSITIVE_TYPES:
        level = "high"
        reasons.append(f"time-sensitive intent '{intent}' (logged answer is stale)")
    if signals["chemical_or_dose"]:
        level = "high"
        reasons.append(f"chemical/dosage content: {_chemical_dose(t)}")
    if signals["pest_disease"] and signals["treatment_request"]:
        level = "high"
        reasons.append("pest/disease question with a treatment request")
    if signals["toxicity"]:
        level = "high"
        reasons.append("toxicity/exposure/harvest-restriction question")

    if level == "low":
        # Unknown but agricultural and not clearly safe ⇒ escalate (medium).
        agri = (signals["pest_disease"] or signals["crop"] or signals["price"]
                or signals["generic_agri"])
        if (intent in (None, "Other")) and agri:
            level = "medium"
            reasons.append("agricultural content with an unidentified intent")
        elif signals["treatment_request"] and (signals["crop"] or signals["generic_agri"]):
            level = "medium"
            reasons.append("treatment request that could not be classified")

    return Risk(level, reasons or ["no risk signal"], signals)
