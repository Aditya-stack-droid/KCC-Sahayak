"""Rule-based multi-intent decomposition (experimental, escalate-only).

The shipped pipeline predicts a **single** intent per query. This module detects
when a query carries two or more distinct requests, splits it into sub-queries,
labels each with the existing keyword rules plus a small bilingual cue lexicon,
and flags when any sub-query is high-stakes or time-sensitive so the *whole*
query can be escalated.

It is deliberately conservative:

* the original query string is preserved verbatim;
* it only ever **adds** an escalation signal — it never removes one, and it is not
  wired into the production gate;
* when a query is not clearly multi-intent it returns ``is_multi=False`` and the
  original single-intent decision stands.

It reuses ``src/weak_labels.py`` and ``src/glossary.py`` rather than training a
new model, and adds no dependency.
"""
from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from glossary import HIGH_STAKES_TYPES, TIME_SENSITIVE_TYPES  # noqa: E402
from weak_labels import label_query as _label_query  # noqa: E402

LABELS = [
    "Animal Husbandry", "Cultural Practices", "Fisheries", "Government Scheme",
    "Market Price", "Nutrient Management", "Other", "Plant Protection",
    "Varietal Information", "Weather",
]

# Small bilingual cue lexicon. English cues are added for plurals and common words
# that the shared keyword rules miss ("aphids", "irrigate"); bilingual cues cover
# Hindi/Hinglish so sub-query labelling works beyond English.
_BILINGUAL_CUES = {
    "Market Price": ["bhav", "bhaav", "mandi", "rate", "msp", "price", "भाव", "मंडी", "कीमत", "दाम"],
    "Nutrient Management": ["urea", "khaad", "khad", "dap", "ssp", "mop", "potash", "nitrogen",
                            "fertiliser", "fertilizer", "manure", "zinc", "यूरिया", "खाद", "उर्वरक", "डीएपी"],
    "Plant Protection": ["dawa", "keeda", "keet", "kida", "spray", "rog", "pest", "pests", "insect",
                         "insects", "aphid", "aphids", "borer", "rust", "blight", "wilt", "whitefly",
                         "termite", "caterpillar", "disease", "weed", "weeds",
                         "रोग", "कीट", "कीटनाशक", "दवा", "छिड़काव", "इल्ली", "रतुआ"],
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
                           "spacing", "tillage", "सिंचाई", "बुवाई", "जुताई"],
}
_CUE_RE = {label: re.compile(r"(?<![A-Za-z])(" + "|".join(re.escape(c) for c in cues) + r")(?![A-Za-z])", re.IGNORECASE)
           for label, cues in _BILINGUAL_CUES.items()}

# Request / question markers that make a segment a "request" rather than filler.
_REQUEST = re.compile(
    r"\b(?:what|which|how|when|where|why|who|whom|whose|tell|explain|write|book|"
    r"give|show|list|suggest|can|could|should|is|are|do|does|will|kya|kaise|kab|kahan|"
    r"kitna|kitni|kitne|batao|bataen|bataiye|chahiye|kaunsa|kaunsi|kaun|karo|karein)\b",
    re.IGNORECASE,
)
# Non-agricultural request markers (programming, general knowledge, travel, ...).
_NON_AGRI = re.compile(
    r"\b(?:python|program|code|blockchain|capital of|mutual fund|interest rate|fixed deposit|"
    r"cricket|movie|award|prime minister|passport|train|ticket|recipe|poem|laptop|"
    r"savings account|boiling point|match|match)\b",
    re.IGNORECASE,
)

_SENT_END = re.compile(r"[?।;\n]+")
_CONJ = re.compile(r"\b(?:and\s+also|and|aur|plus|also|as\s+well\s+as|तथा|और)\b", re.IGNORECASE)


@dataclass
class SubQuery:
    text: str
    intent: str
    start: int
    end: int
    non_agricultural: bool = False
    high_risk: bool = False


@dataclass
class MultiIntentResult:
    input_text: str
    is_multi: bool
    sub_queries: list[SubQuery] = field(default_factory=list)
    distinct_intents: list[str] = field(default_factory=list)
    safety_escalation_required: bool = False
    reason: str = ""


def _label_segment(segment: str) -> str:
    # A clearly non-agricultural sub-request is "Other", never an agri class even
    # if it contains a shared word ("interest rate" must not become Market Price).
    if _NON_AGRI.search(segment):
        return "Other"
    english = _label_query(segment)
    if english != "Other":
        return english
    counts = {label: len(rx.findall(segment)) for label, rx in _CUE_RE.items()}
    counts = {k: v for k, v in counts.items() if v}
    if not counts:
        return "Other"
    return max(counts.items(), key=lambda kv: kv[1])[0]


def _is_request(segment: str) -> bool:
    if _NON_AGRI.search(segment):
        return True
    if _REQUEST.search(segment):
        return True
    return _label_segment(segment) != "Other"


def _split(text: str) -> list[tuple[str, int]]:
    segments: list[tuple[str, int]] = []
    for sent in re.finditer(r"[^?।;\n]+", text):
        base = sent.start()
        piece = sent.group()
        last = 0
        for m in _CONJ.finditer(piece):
            chunk = piece[last:m.start()]
            if chunk.strip():
                segments.append((chunk.strip(), base + last))
            last = m.end()
        tail = piece[last:]
        if tail.strip():
            segments.append((tail.strip(), base + last))
    return segments


def decompose(text: str | None) -> MultiIntentResult:
    """Detect and decompose a multi-request query. Never raises on bad input."""
    if not isinstance(text, str) or not text.strip():
        return MultiIntentResult(text or "", False, reason="empty input")
    parts = []
    for segment, start in _split(text):
        if not _is_request(segment):
            continue
        label = _label_segment(segment)
        non_agri = bool(_NON_AGRI.search(segment))
        parts.append(SubQuery(segment, label, start, start + len(segment),
                              non_agricultural=non_agri,
                              high_risk=label in HIGH_STAKES_TYPES or label in TIME_SENSITIVE_TYPES))
    # Keep at most one sub-query per distinct label to avoid splitting a single request
    # that merely contains a conjunction.
    by_label: dict[str, SubQuery] = {}
    for sub in parts:
        by_label.setdefault(sub.intent, sub)
    distinct = [l for l in LABELS if l in by_label]
    non_agri_hits = sum(1 for s in parts if s.non_agricultural)
    is_multi = len(distinct) >= 2 or non_agri_hits >= 1 and len(parts) >= 2
    safety = any(s.high_risk for s in parts)
    reason = (
        f"{len(distinct)} distinct intents" if is_multi
        else "single request"
    )
    return MultiIntentResult(
        input_text=text,
        is_multi=bool(is_multi),
        sub_queries=parts,
        distinct_intents=distinct,
        safety_escalation_required=bool(safety),
        reason=reason,
    )


def intent_set(text: str | None) -> tuple[str, ...]:
    """Convenience for evaluation: the (multi-)intent label set of a query."""
    return tuple(decompose(text).distinct_intents)
