"""Number, unit and interval extraction for conflict detection.

Deterministic and dependency-free.  The goal is not general quantity parsing;
it is to pull out the handful of agronomic quantities that two answers can
contradict each other about:

* mass (g / kg / quintal / tonne),
* volume (ml / litre),
* concentration (%),
* the application basis (/acre, /hectare, /bigha, /litre, /plant).

Rules that keep this conservative:

* a quantity needs a unit — bare numbers (years, plot ids, prices) are ignored;
* currency, plot/token/order ids and pack sizes are rejected by local context;
* area bases are only compared when they are the same base, except hectare,
  which has a fixed conversion to acre (1 ha = 2.4710538 acre).  ``bigha``,
  ``kanal`` and ``guntha`` are not converted because their size varies by
  district; different bases are reported as a basis conflict instead.
* a "per N litre" / "in N litre water" or "per N acre" basis is normalised to
  per-one-unit before comparison, so ``2 ml per 10 litre`` is compared as
  ``0.2 ml/litre``.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, replace

DEVANAGARI_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")
HECTARE_TO_ACRE = 2.471053814671653

# raw unit -> (base unit, dimension, multiplier into the base unit)
_UNIT_ALIASES: dict[str, tuple[str, str, float]] = {}


def _units(base: str, dimension: str, factor: float, names: tuple[str, ...]) -> None:
    for name in names:
        _UNIT_ALIASES[name] = (base, dimension, factor)


_units("g", "mass", 1.0, ("g", "gm", "gms", "gram", "grams", "ग्राम"))
_units("g", "mass", 1_000.0, ("kg", "kgs", "kilo", "kilos", "kilogram", "kilograms", "किलो"))
_units("g", "mass", 100_000.0, ("q", "quintal", "quintals", "क्विंटल"))
_units("g", "mass", 1_000_000.0, ("tonne", "tonnes", "ton", "tons"))
_units("ml", "volume", 1.0, ("ml", "mls", "millilitre", "millilitres", "milliliter", "milliliters", "मिलीलीटर"))
_units("ml", "volume", 1_000.0, ("l", "lt", "ltr", "ltrs", "lit", "litre", "litres", "liter", "liters", "लीटर"))
_units("%", "concentration", 1.0, ("%", "percent"))
_units("count", "count", 1.0, (
    "teaspoon", "teaspoons", "teaspoonful", "teaspoonfuls", "tsp",
    "tablespoon", "tablespoons", "tbsp", "spoon", "spoons",
))

_UNIT_ALTERNATION = "|".join(
    re.escape(name) for name in sorted(_UNIT_ALIASES, key=len, reverse=True)
)

_AREA_ALIASES = {
    "acre": "acre", "acres": "acre",
    "hectare": "hectare", "hectares": "hectare", "ha": "hectare",
    "bigha": "bigha", "bighas": "bigha",
    "kanal": "kanal", "kanals": "kanal",
    "guntha": "guntha", "gunthas": "guntha",
    "cent": "cent", "cents": "cent",
}
_VOLUME_BASES = {"litre": "litre", "litres": "litre", "liter": "litre", "liters": "litre", "l": "litre"}
_PLANT_BASES = {"plant": "plant", "plants": "plant", "tree": "tree", "trees": "tree"}

_BASIS_ALTERNATION = "|".join(
    re.escape(name)
    for name in sorted({*_AREA_ALIASES, *_VOLUME_BASES, *_PLANT_BASES}, key=len, reverse=True)
)

_NUMBER = r"\d+(?:\.\d+)?"
_QUANTITY_RE = re.compile(
    rf"(?<![A-Za-z0-9_.])"
    rf"(?P<num>{_NUMBER})(?:\s*(?:-|–|—|to)\s*(?P<num2>{_NUMBER}))?"
    rf"\s*(?P<unit>{_UNIT_ALTERNATION})(?![A-Za-z])"
    rf"(?:\s*(?:/|\bper\b|\bin\b)\s*(?P<basis_num>{_NUMBER})?\s*"
    rf"(?P<basis>{_BASIS_ALTERNATION})\b)?",
    re.IGNORECASE,
)

_BASIS_RE = re.compile(
    rf"(?:/|\bper\b|\bin\b)\s*(?P<num>{_NUMBER})?\s*(?P<basis>{_BASIS_ALTERNATION})\b",
    re.IGNORECASE,
)

_CURRENCY_BEFORE = re.compile(r"(?:rs\.?|inr|₹|rupees?)\s*$", re.IGNORECASE)
_NON_QUANTITY_BEFORE = re.compile(
    r"(?:plot|token|order|invoice|sample|batch|khasra|survey|serial|sr\.?|s\.?no\.?)\s*(?:no\.?|number|id)?\s*$",
    re.IGNORECASE,
)
_PACK_AFTER = re.compile(
    r"^\s*(?:pack|packs|packet|packets|bag|bags|bottle|bottles|box|boxes|tin|tins|sachet|sachets|"
    r"price|cost|mrp|rupees?|rs\.?)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Quantity:
    raw: str
    value_low: float
    value_high: float
    unit: str
    dimension: str
    basis: str | None
    basis_kind: str | None
    basis_raw: str | None
    per_amount: float
    start: int
    end: int

    @property
    def basis_key(self) -> str | None:
        if self.basis is None:
            return None
        return f"{self.basis_kind}:{self.basis}"

    def normalised_bounds(self) -> tuple[float, float]:
        """Bounds per one acre / one litre / one plant, in the base unit."""
        amount = self.per_amount if self.per_amount > 0 else 1.0
        return self.value_low / amount, self.value_high / amount

    def display(self) -> str:
        text = " ".join(self.raw.split())
        if self.basis_raw and self.basis_raw.lower() not in text.lower():
            text = f"{text} {self.basis_raw}"
        return text


@dataclass(frozen=True)
class Interval:
    low: float
    high: float
    raw: str
    start: int
    end: int


def normalise_digits(text: str) -> str:
    """Translate Devanagari digits 1:1 so offsets remain valid."""
    return str(text).translate(DEVANAGARI_DIGITS)


def _excluded(text: str, start: int, end: int) -> bool:
    before = text[max(0, start - 24):start]
    after = text[end:end + 16]
    if _CURRENCY_BEFORE.search(before):
        return True
    if _NON_QUANTITY_BEFORE.search(before):
        return True
    if _PACK_AFTER.search(after):
        return True
    return False


def _parse_basis(raw: str | None, amount: float) -> tuple[str | None, str | None, float]:
    if not raw:
        return None, None, 1.0
    name = raw.lower()
    if name in _AREA_ALIASES:
        basis = _AREA_ALIASES[name]
        if basis == "hectare":
            return "acre", "area", amount * HECTARE_TO_ACRE
        return basis, "area", amount
    if name in _VOLUME_BASES:
        return "litre", "volume", amount
    if name in _PLANT_BASES:
        return _PLANT_BASES[name], "plant", amount
    return None, None, 1.0


def _parse_quantity(text: str, match: re.Match) -> Quantity | None:
    start, end = match.start(), match.end()
    if _excluded(text, start, end):
        return None
    unit_raw = match.group("unit").lower().rstrip(".")
    if unit_raw not in _UNIT_ALIASES:
        return None
    base_unit, dimension, factor = _UNIT_ALIASES[unit_raw]
    low = float(match.group("num"))
    high = float(match.group("num2")) if match.group("num2") else low
    low, high = min(low, high) * factor, max(low, high) * factor
    if not math.isfinite(low) or not math.isfinite(high) or high <= 0:
        return None
    amount = float(match.group("basis_num")) if match.group("basis_num") else 1.0
    basis, basis_kind, per_amount = _parse_basis(match.group("basis"), amount)
    return Quantity(
        raw=text[start:end],
        value_low=low,
        value_high=high,
        unit=base_unit,
        dimension=dimension,
        basis=basis,
        basis_kind=basis_kind,
        basis_raw=match.group("basis") if match.group("basis") else None,
        per_amount=per_amount,
        start=start,
        end=end,
    )


def _basis_mentions(source: str) -> list[tuple[int, int, str, str, float, str]]:
    """Basis phrases such as ``per acre`` and ``in 10 litre`` with their spans."""
    mentions: list[tuple[int, int, str, str, float, str]] = []
    for match in _BASIS_RE.finditer(source):
        amount = float(match.group("num")) if match.group("num") else 1.0
        basis, basis_kind, per_amount = _parse_basis(match.group("basis"), amount)
        if basis is None:
            continue
        mentions.append((match.start(), match.end(), basis, basis_kind,
                         per_amount, match.group(0).strip()))
    return mentions


def _attach_basis(
    quantity: Quantity,
    mentions: list[tuple[int, int, str, str, float, str]],
    quantities: list[Quantity],
    source: str,
) -> Quantity:
    """Attach a nearby basis ("50 kg urea per acre") to a bare quantity."""
    if quantity.basis is not None:
        return quantity
    for start, end, basis, basis_kind, per_amount, raw in mentions:
        if start < quantity.end:
            continue
        gap = source[quantity.end:start]
        if len(gap) > 60 or re.search(r"[.;!?\n]", gap):
            continue
        # Do not jump over another quantity: "2 kg urea and spray 2 ml per litre".
        if any(other is not quantity and other.start >= quantity.end and other.end <= start
               for other in quantities):
            continue
        return replace(quantity, basis=basis, basis_kind=basis_kind,
                       basis_raw=raw, per_amount=per_amount)
    return quantity


def extract_quantities(text: str) -> tuple[Quantity, ...]:
    """All quantity mentions with a recognised unit, in text order."""
    source = normalise_digits(text)
    mentions = _basis_mentions(source)
    quantities: list[Quantity] = []
    for match in _QUANTITY_RE.finditer(source):
        quantity = _parse_quantity(source, match)
        if quantity is None:
            continue
        # A quantity sitting inside a basis phrase (the "10 litre" of
        # "in 10 litre water") describes the diluent, not a second dose.
        if quantity.basis is None and any(
            quantity.start < end and quantity.end > start for start, end, *_ in mentions
        ):
            continue
        quantities.append(quantity)
    return tuple(_attach_basis(quantity, mentions, quantities, source) for quantity in quantities)


def is_dose(quantity: Quantity) -> bool:
    """Mass, volume or concentration rather than a count/spoon estimate."""
    return quantity.dimension in {"mass", "volume", "concentration"}


_INTERVAL_PATTERNS = (
    re.compile(
        r"\b(?:after|interval\s+of|every)\s+(?:about\s+|around\s+)?"
        r"(?P<num>\d+(?:\.\d+)?)(?:\s*(?:-|–|to)\s*(?P<num2>\d+(?:\.\d+)?))?\s*(?:days?|din)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?P<num>\d+(?:\.\d+)?)\s*(?:days?|din)\s*(?:interval|gap)\b",
        re.IGNORECASE,
    ),
)
_NEGATION = re.compile(
    r"\b(?:do\s+not|don't|dont|does\s+not|never|avoid|not\s+to|should\s+not|no\s+need\s+to|mat|nahi)\b",
    re.IGNORECASE,
)


def extract_intervals(text: str) -> tuple[Interval, ...]:
    """Application intervals in days, e.g. "after 5 days", "at 10 day interval"."""
    source = normalise_digits(text)
    found: list[Interval] = []
    seen: set[tuple[int, int]] = set()
    for pattern in _INTERVAL_PATTERNS:
        for match in pattern.finditer(source):
            span = (match.start(), match.end())
            if span in seen:
                continue
            seen.add(span)
            groups = match.groupdict()
            low = float(groups["num"])
            high = float(groups["num2"]) if groups.get("num2") else low
            found.append(Interval(
                low=min(low, high),
                high=max(low, high),
                raw=text[match.start():match.end()],
                start=match.start(),
                end=match.end(),
            ))
    return tuple(sorted(found, key=lambda item: item.start))


def has_negation(text: str) -> bool:
    return bool(_NEGATION.search(normalise_digits(text)))
