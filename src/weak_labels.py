"""Rule-based weak labeling of KCC QueryType, used because the public corpus
ships only free-text question/answer pairs with no category column.

Distant/weak supervision from a domain keyword glossary is the same strategy
used by the KCC NER literature (bootstrapped glossary annotation) cited in
the project's literature survey -- applied here one level up, to intent
labels instead of entity spans.
"""
import re

from glossary import QUERY_TYPE_RULES

_COMPILED = {
    label: [re.compile(r"\b" + re.escape(kw) + r"\b") for kw in keywords]
    for label, keywords in QUERY_TYPE_RULES.items()
}


def label_query(text: str) -> str:
    """Return the best-matching QueryType label, or 'Other' if no rule fires.

    Ties are broken by the label with the most matched keywords (more
    specific match wins over a single generic hit).
    """
    text = text.lower()
    scores = {}
    for label, patterns in _COMPILED.items():
        count = sum(1 for p in patterns if p.search(text))
        if count:
            scores[label] = count
    if not scores:
        return "Other"
    return max(scores.items(), key=lambda kv: kv[1])[0]
