"""Product-name recognition for conflict detection.

Only spelling/abbreviation variants are normalised.  The module deliberately
does **not** map a brand to an active ingredient (for example ``rogor`` and
``dimethoate`` are kept distinct): asserting chemical equivalence without a
verified product database would create false conflicts *and* false agreements.
Different names are therefore reported as different products, with the caveat
documented in the report.
"""

from __future__ import annotations

import re

# canonical name -> aliases (all lowercase).  Kept intentionally small and
# agronomy-focused; sourced from src/confidence_gate.py's chemical vocabulary
# plus common KCC answers.  Nothing here is a recommendation.
PRODUCT_ALIASES: dict[str, tuple[str, ...]] = {
    "urea": ("urea",),
    "dap": ("dap", "d a p", "di ammonium phosphate", "diammonium phosphate"),
    "ssp": ("ssp", "single super phosphate"),
    "mop": ("mop", "muriate of potash", "potash"),
    "npk": ("npk", "n p k"),
    "zinc sulphate": ("zinc sulphate", "zinc sulfate", "zinc sulph"),
    "borax": ("borax",),
    "sulphur": ("sulphur", "sulfur", "wettable sulphur"),
    "gypsum": ("gypsum",),
    "ammonium sulphate": ("ammonium sulphate", "ammonium sulfate"),
    "fym": ("fym", "farm yard manure", "farmyard manure"),
    "vermicompost": ("vermicompost", "vermi compost"),
    "chlorpyrifos": ("chlorpyrifos",),
    "imidacloprid": ("imidacloprid",),
    "cypermethrin": ("cypermethrin",),
    "mancozeb": ("mancozeb",),
    "carbendazim": ("carbendazim",),
    "copper oxychloride": ("copper oxychloride", "copperoxy chloride"),
    "copper sulphate": ("copper sulphate", "copper sulfate"),
    "bordeaux": ("bordeaux", "bordeaux mixture"),
    "rogor": ("rogor",),
    "dimethoate": ("dimethoate",),
    "monocrotophos": ("monocrotophos",),
    "endosulfan": ("endosulfan",),
    "butachlor": ("butachlor",),
    "glyphosate": ("glyphosate",),
    "2,4-d": ("2 4 d", "2 4d", "24d", "2,4 d"),
    "neem": ("neem", "neem oil", "neem kernel", "neem kernel extract"),
    "planofix": ("planofix",),
    "malathion": ("malathion", "maltahion"),
    "captan": ("captan",),
    "carbaryl": ("carbaryl",),
    "phorate": ("phorate",),
    "quinalphos": ("quinalphos",),
    "profenofos": ("profenofos",),
    "thiamethoxam": ("thiamethoxam",),
    "fipronil": ("fipronil",),
    "hexaconazole": ("hexaconazole",),
    "propiconazole": ("propiconazole",),
    "tebuconazole": ("tebuconazole",),
    "tricyclazole": ("tricyclazole",),
    "streptocycline": ("streptocycline", "streptomycin"),
    "atrazine": ("atrazine",),
    "pendimethalin": ("pendimethalin",),
    "pretilachlor": ("pretilachlor",),
    "quizalofop": ("quizalofop",),
    "chlorantraniliprole": ("chlorantraniliprole",),
    "emamectin": ("emamectin", "emamectin benzoate"),
    "spinosad": ("spinosad",),
    "abamectin": ("abamectin",),
    "acephate": ("acephate",),
    "methomyl": ("methomyl",),
    "thiodicarb": ("thiodicarb",),
    "dichlorvos": ("dichlorvos", "dichlorovos"),
}

_NEGATION = re.compile(
    r"\b(?:do\s+not|don't|dont|does\s+not|never|avoid|not\s+to|should\s+not|no\s+need\s+to|mat|nahi)\b"
    r"|\bnot\s+(?:recommended|advised|advisable|suggested|safe|needed|required|used)\b",
    re.IGNORECASE,
)
_CLAUSE_BREAK = re.compile(r"[.;!?\n]|\bbut\b|\binstead\b|\bhowever\b", re.IGNORECASE)


def _alias_regex(alias: str) -> re.Pattern:
    tokens = [token for token in re.split(r"[\s,.\-_/]+", alias.lower()) if token]
    body = r"[\s,.\-_/]*".join(re.escape(token) for token in tokens)
    return re.compile(r"(?<![a-z0-9])" + body + r"(?![a-z0-9])", re.IGNORECASE)


_PATTERNS = [
    (canonical, _alias_regex(alias))
    for canonical, aliases in PRODUCT_ALIASES.items()
    for alias in aliases
]


def product_matches(text: str) -> tuple[tuple[str, int, int], ...]:
    """All product mentions as ``(canonical, start, end)``, longest-first at overlaps."""
    if not isinstance(text, str) or not text:
        return ()
    matches: list[tuple[str, int, int]] = []
    for canonical, pattern in _PATTERNS:
        for match in pattern.finditer(text):
            matches.append((canonical, match.start(), match.end()))
    matches.sort(key=lambda item: (item[1], -(item[2] - item[1])))
    kept: list[tuple[str, int, int]] = []
    for candidate in matches:
        if any(not (candidate[2] <= kept_item[1] or candidate[1] >= kept_item[2])
               for kept_item in kept):
            continue
        kept.append(candidate)
    return tuple(kept)


def find_products(text: str) -> set[str]:
    return {canonical for canonical, _, _ in product_matches(text)}


def _clauses(text: str) -> list[tuple[int, int]]:
    starts = [0]
    for match in _CLAUSE_BREAK.finditer(text):
        starts.append(match.end())
    starts = sorted(set(starts))
    return [(start, starts[index + 1] if index + 1 < len(starts) else len(text))
            for index, start in enumerate(starts)]


def negated_products(text: str) -> set[str]:
    """Products that appear inside a negated clause of the answer.

    Clause-local by design, so "do not apply urea; instead spray neem oil"
    negates urea only.
    """
    if not isinstance(text, str) or not text:
        return set()
    clauses = _clauses(text)
    negated: set[str] = set()
    for canonical, start, end in product_matches(text):
        midpoint = (start + end) / 2
        for clause_start, clause_end in clauses:
            if clause_start <= midpoint <= clause_end:
                if _NEGATION.search(text[clause_start:clause_end]):
                    negated.add(canonical)
                break
    return negated
