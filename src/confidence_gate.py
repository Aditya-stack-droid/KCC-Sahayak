"""Confidence-gated decision layer: turns (intent confidence, retrieval
similarity, query type, retrieved answer) into an AUTO_ANSWER vs ESCALATE
decision.

This formalizes the reject-option / selective-classification framing from
the literature survey (Chow 1970; Geifman & El-Yaniv 2017) as a concrete,
tunable policy. A query is auto-answered only if ALL of these hold:

  1. its predicted type is not on the high-stakes list,
  2. its type is not time-sensitive (weather, prices): a logged answer is stale,
  3. intent-classifier confidence clears a threshold,
  4. retrieval similarity clears a threshold and the closest logged question is
     about the same topic as the query (retrieval can return an off-topic answer),
  5. the retrieved text is real advice (not a call-log note like "explained in
     details"), and
  6. it contains no dosage / chemical instruction.

Check 6 exists because the query's category does not bound the risk of the
text we would send back: a "Varietal Information" question can retrieve an
answer that ends in "...fertilizer dose 24kg urea per bigha".
"""
import re
from dataclasses import dataclass

from answer_quality import is_informative
from glossary import HIGH_STAKES_TYPES, TIME_SENSITIVE_TYPES

# Quantities with agronomic units (e.g. "2ml/lit", "50 kg", "600gm", "@ 2 g")
_DOSE = re.compile(
    r"\d+(?:\.\d+)?\s*(?:ml|mls|gm|gms|g|kg|kgs|lit|litre|liter|l|q|quintal|ppm|%)\b"
    r"|@\s*\d"
    r"|\bper\s+(?:bigha|acre|hectare|ha|kanal|plant|litre|liter)\b"
    r"|\b\d+\s*(?:kg|gm|g)\s*/\s*(?:acre|ha|bigha|plant)\b",
    re.IGNORECASE,
)

# Pesticide / fertilizer / chemical names and application verbs that signal
# "do this chemical thing" advice.
_CHEMICALS = re.compile(
    r"\b(?:urea|dap|ssp|mop|npk|potash|zinc sulphate|borax|sulphur|sulfur|"
    r"insecticide|pesticide|fungicide|herbicide|weedicide|acaricide|"
    r"chlorpyrifos|imidacloprid|cypermethrin|mancozeb|carbendazim|"
    r"copper oxychloride|copper sulphate|bordeaux|rogor|dimethoate|"
    r"monocrotophos|endosulfan|butachlor|2,4-?d|glyphosate|neem oil|"
    r"spray(?:ing)?|dusting)\b",
    re.IGNORECASE,
)


def answer_risk(answer: str | None) -> str | None:
    """Return a short reason if the answer text carries dosage/chemical
    instructions, else None."""
    if not answer:
        return None
    if _DOSE.search(answer):
        return "dosage/quantity instruction"
    if _CHEMICALS.search(answer):
        return "chemical or spray recommendation"
    return None


@dataclass
class GateThresholds:
    intent_confidence: float = 0.70
    retrieval_similarity: float = 0.55
    screen_answer: bool = True
    block_time_sensitive: bool = True
    block_high_stakes: bool = True
    require_topic_match: bool = True


@dataclass
class GateDecision:
    action: str  # "AUTO_ANSWER" or "ESCALATE"
    reason: str
    code: str = ""  # machine-readable: high_stakes | low_intent | low_retrieval | risky_answer | ok


def decide(
    query_type: str,
    intent_confidence: float,
    retrieval_similarity: float,
    thresholds: GateThresholds = GateThresholds(),
    answer: str | None = None,
    retrieved_type: str | None = None,
    coverage_problem: str | None = None,
) -> GateDecision:
    if thresholds.block_high_stakes and query_type in HIGH_STAKES_TYPES:
        return GateDecision(
            "ESCALATE",
            f"high-stakes query type '{query_type}' always routed to a human FTA",
            "high_stakes",
        )
    if thresholds.block_time_sensitive and query_type in TIME_SENSITIVE_TYPES:
        return GateDecision(
            "ESCALATE",
            f"'{query_type}' answers go stale; a logged answer cannot be reused, so live data or an advisor is needed",
            "time_sensitive",
        )
    if intent_confidence < thresholds.intent_confidence:
        return GateDecision(
            "ESCALATE",
            f"intent confidence {intent_confidence:.2f} below threshold "
            f"{thresholds.intent_confidence:.2f}",
            "low_intent",
        )
    if retrieval_similarity < thresholds.retrieval_similarity:
        return GateDecision(
            "ESCALATE",
            f"retrieval similarity {retrieval_similarity:.2f} below threshold "
            f"{thresholds.retrieval_similarity:.2f}",
            "low_retrieval",
        )
    if thresholds.require_topic_match and retrieved_type is not None and retrieved_type != query_type:
        return GateDecision(
            "ESCALATE",
            f"the closest logged question is about '{retrieved_type}', not '{query_type}': likely an off-topic answer",
            "topic_mismatch",
        )
    if thresholds.require_topic_match and coverage_problem:
        return GateDecision(
            "ESCALATE",
            f"no solid answer in the corpus: {coverage_problem}",
            "no_solid_match",
        )
    if thresholds.screen_answer and answer is not None and not is_informative(answer):
        return GateDecision(
            "ESCALATE",
            "the closest logged answer is only a call-log note (no actual advice was recorded)",
            "uninformative_answer",
        )
    if thresholds.screen_answer:
        risk = answer_risk(answer)
        if risk:
            return GateDecision(
                "ESCALATE",
                f"retrieved answer contains a {risk}; a human must confirm it fits this farmer's case",
                "risky_answer",
            )
    return GateDecision(
        "AUTO_ANSWER",
        f"intent confidence {intent_confidence:.2f} and retrieval similarity "
        f"{retrieval_similarity:.2f} cleared threshold for non-high-stakes type "
        f"'{query_type}'; answer screened clean",
        "ok",
    )
