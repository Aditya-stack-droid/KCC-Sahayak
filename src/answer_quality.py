"""Is a logged KCC answer actual advice, or just a call-log note?

Many KCC "answers" are notes written by the Farm Tele Advisor saying advice was
given without recording it: "explained in details", "transferred to expert",
"expert not available". Returning one to a farmer is useless, and scoring a
system against one is meaningless. An answer is NON-informative when

  A. every word is a "note" verb or filler ("explained him in details"), or
  B. it is a pure referral (transfer/forward + expert/officer/department), or
  C. it contains a fixed non-answer phrase ("unable to resolve", "inform later").

Words are matched with typo tolerance (the log is full of "explian", "trasfer",
"detial") via difflib similarity >= 0.8 against a small vocabulary.
"""
import re
from difflib import get_close_matches
from functools import lru_cache

_VERBS = ["explain", "explained", "explaining", "describe", "described", "given", "gave", "give",
          "suggested", "suggest", "told", "advised", "advice", "informed", "discussed", "replied",
          "answered", "clarified", "guided"]
_FILLER = ["him", "her", "them", "the", "farmer", "in", "detail", "details", "detailed", "briefly",
           "properly", "clearly", "accordingly", "elaborately", "thoroughly", "with", "and", "all",
           "everything", "to", "about", "well", "satisfactorily", "fully", "a", "of", "for", "further",
           "information", "more", "need", "needed", "required", "regarding"]
_TRANSFER = ["transfer", "transferred", "forward", "forwarded", "refer", "referred", "conference",
             "connected", "registered"]
_EXPERT = ["expert", "officer", "specialist", "scientist", "veterinary", "agri", "agriculture",
           "department", "dept", "consultation"]
_PHRASES = [
    r"unable to (?:resolve|answer|understand|hear)",
    r"(?:call|line)\s+(?:was\s+)?(?:disconnect\w*|dropped|cut)",
    r"\bno\s+(?:response|reply|information|answer)\b",
    r"expert\s+(?:is\s+)?not\s+available",
    r"(?:ask|asked|told)\s+(?:him\s+)?to\s+(?:inform|call|contact)\s+(?:later|again)",
    r"inform\s+later",
    r"^\W*(?:not available|wait for some\w*|disconnec\w*|forwarded|transferred|transfered|noted|ok|okay)\W*$",
]
_PHRASE_RE = [re.compile(p, re.IGNORECASE) for p in _PHRASES]
# "given in details about the package of practices", "advised regarding different managerial
# aspects of pond preparation": the advice is deferred to a generic reference, not stated.
_GENERIC = re.compile(
    r"(?:package\s*(?:of|&|and)?\s*practices?|cultivation practices|managerial aspects?|"
    r"management practices|all aspects?|different aspects?|various aspects?)", re.IGNORECASE)
_VOCAB_ALL = _VERBS + _FILLER


@lru_cache(maxsize=200_000)
def _kind(tok: str) -> str:
    """Classify a token as 'verb', 'filler', 'transfer', 'expert' or 'other'."""
    for name, vocab in (("verb", _VERBS), ("filler", _FILLER), ("transfer", _TRANSFER), ("expert", _EXPERT)):
        if tok in vocab or get_close_matches(tok, vocab, n=1, cutoff=0.8):
            return name
    return "other"


def is_informative(answer) -> bool:
    if not isinstance(answer, str):
        return False
    a = " ".join(answer.lower().split())
    if len(a) < 5:
        return False
    if any(p.search(a) for p in _PHRASE_RE):
        return False
    if _GENERIC.search(a) and not re.search(r"\d", a):
        return False  # C2: refers to a generic reference instead of stating advice
    toks = re.findall(r"[a-z]+", a)
    if not toks:
        return True  # digits / symbols only, e.g. a phone number or a variety code
    kinds = [_kind(t) for t in toks]
    if "other" not in kinds and any(k in kinds for k in ("verb", "transfer", "expert")):
        return False  # A: only note verbs / referral words / filler ("explained in details", "transfer")
    if "transfer" in kinds and "expert" in kinds and kinds.count("other") <= 3:
        return False  # B: pure referral
    return True


def informative_mask(series):
    return series.map(is_informative).to_numpy()
