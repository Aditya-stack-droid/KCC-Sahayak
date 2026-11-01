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

_COMPARE = re.compile(r"\b(better|versus|vs\.?|compare[ds]?|comparison|difference|behtar)\b", re.I)


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
