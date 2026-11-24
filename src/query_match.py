"""Does the closest logged question actually address what the farmer asked?

Dense retrieval happily returns a near neighbour that shares the main topic word ("tomato") but answers a
different question ("what are the different types of tomatoes" for "what is better, tomato or cucumber").
Two cheap, precise checks catch the most common version of that failure:

  1. Crop coverage: every crop the farmer names must also appear in the retrieved question or answer.
  2. Comparison cue: if the farmer asks for a comparison ("better", "versus", "compare", "difference"),
     the retrieved question must be a comparison too.

Returns a short reason string when the match should NOT be trusted, otherwise None.
"""
import re

from ner import extract_entities

# Questions about NOW or the near future (live weather, today's price, tomorrow ...). The KCC logs are historical
# records, so no logged answer can be correct for them, whatever topic the classifier picks.
_TEMPORAL = re.compile(
    r"\b(?:today|tonight|tomorrow|tmrw|right now|currently|"
    r"this\s+(?:week|weekend|evening|morning)|next\s+(?:\d+\s+|few\s+|two\s+|three\s+)?(?:days?|week|weeks|hours?)|"
    r"(?:is|are)\s+(?:it|there)\s+(?:going\s+to|gonna)|will\s+(?:it|there)\s+(?:rain|be|snow|storm)|"
    r"aaj|abhi|kal|parso|parson|agle\s+\w+\s+din|agle\s+hafte|is\s+hafte)\b"
    r"|आज|अभी|कल|परसों|अगले\s+\S+\s+दिन|इस\s+हफ्ते",
    re.I,
)

# Prices move daily and differ by mandi and state, so a logged price is wrong for this farmer even when the
# question has no time word. "seed rate" / "rate of application" are doses, not prices, and do not match.
_PRICE = re.compile(
    r"\b(?:prices?|pricing|bhav|bhaav|bhaw|mandi|market\s+rate|selling\s+rate|msp\s+rate|"
    r"how\s+much\s+(?:does|do|will)\s+\w+\s+(?:cost|sell))\b|भाव|मंडी|कीमत|दाम",
    re.I,
)

_COMPARE = re.compile(r"\b(better|versus|vs\.?|compare[ds]?|comparison|difference|behtar)\b", re.I)


def temporal_problem(query: str) -> str | None:
    """Reason string if the question is about current or upcoming conditions, else None."""
    p = _PRICE.search(query or "")
    if p:
        return (f"the question is about market prices ('{p.group(0)}'); prices change daily and by mandi and region, "
                "so a logged price cannot be right for this farmer")
    m = _TEMPORAL.search(query or "")
    if m:
        return (f"the question is about current or upcoming conditions ('{m.group(0)}'); "
                "the KCC logs are historical and cannot answer it")
    return None


def coverage_problem(query: str, retrieved_question: str, retrieved_answer: str = "") -> str | None:
    q = (query or "").lower()
    rq = (retrieved_question or "").lower()
    both = f"{rq} {(retrieved_answer or '').lower()}"
    missing = [c for c in extract_entities(q)["crop"] if c not in both]
    if missing:
        return f"the farmer asked about {', '.join(missing)}, which the closest logged question does not mention"
    if _COMPARE.search(q) and not _COMPARE.search(rq):
        return "the farmer asked for a comparison, but the closest logged question is not a comparison"
    return None
