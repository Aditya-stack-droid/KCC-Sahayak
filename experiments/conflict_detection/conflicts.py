"""Deterministic detection of contradictory advice across retrieved answers.

The module compares answers pairwise and reports:

* ``product``      — different products recommended for the same question;
* ``polarity``     — one answer advises against a product the other recommends;
* ``dosage``       — same product/unit/basis but substantially different amounts;
* ``basis``        — doses expressed on different, non-convertible bases
                     (e.g. per acre vs per bigha);
* ``dimension``    — same product dosed in incompatible units (kg vs litre);
* ``interval``     — substantially different application intervals.

Nothing is merged, averaged or resolved.  A non-empty conflict list means the
caller should escalate to a human advisor.  The dosage tolerance is explicit:
amounts are compared after unit/basis normalisation, and a conflict is raised
only when even the closest pair of amounts differs by at least 50% of the
larger value (so 50 vs 55 kg/acre is not a conflict; 50 vs 100 is).
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Iterable, Sequence

from .products import find_products, negated_products
from .units import Interval, Quantity, extract_intervals, extract_quantities, is_dose

DOSE_GAP_TOLERANCE = 0.5
MAX_ANSWERS = 5
SEVERITY_ORDER = {"high": 3, "medium": 2, "low": 1}


@dataclass(frozen=True)
class Conflict:
    kind: str
    severity: str
    explanation: str
    answer_a: int
    answer_b: int

    def as_dict(self) -> dict:
        return {
            "kind": self.kind,
            "severity": self.severity,
            "explanation": self.explanation,
            "answer_a": self.answer_a,
            "answer_b": self.answer_b,
        }


@dataclass(frozen=True)
class ConflictReport:
    n_answers: int
    conflicts: tuple[Conflict, ...]
    escalate: bool
    max_severity: str | None
    reasons: tuple[str, ...]

    def as_dict(self) -> dict:
        return {
            "n_answers": self.n_answers,
            "conflicts": [conflict.as_dict() for conflict in self.conflicts],
            "escalate": self.escalate,
            "max_severity": self.max_severity,
            "reasons": list(self.reasons),
        }


def _interval_gap(low_a: float, high_a: float, low_b: float, high_b: float) -> float:
    if low_a <= high_b and low_b <= high_a:
        return 0.0
    gap = low_b - high_a if low_b > high_a else low_a - high_b
    reference = max(high_a, high_b)
    return gap / reference if reference > 0 else 1.0


def _ratio(low_a: float, high_a: float, low_b: float, high_b: float) -> float:
    low = min(low_a, low_b)
    high = max(high_a, high_b)
    if low <= 0:
        return float("inf")
    return high / low


def _severity_for_ratio(ratio: float) -> str:
    return "high" if ratio >= 2.0 else "medium"


def _dose_conflicts(
    quantities_a: Sequence[Quantity],
    quantities_b: Sequence[Quantity],
    products_a: set[str],
    products_b: set[str],
    index_a: int,
    index_b: int,
) -> list[Conflict]:
    doses_a = [quantity for quantity in quantities_a if is_dose(quantity)]
    doses_b = [quantity for quantity in quantities_b if is_dose(quantity)]
    if not doses_a or not doses_b:
        return []

    def key(quantity: Quantity) -> tuple[str, str]:
        return quantity.dimension, quantity.basis_key or "none"

    keys_a = {key(quantity) for quantity in doses_a}
    keys_b = {key(quantity) for quantity in doses_b}
    shared = keys_a & keys_b
    conflicts: list[Conflict] = []

    if not shared:
        basis_a = {quantity.basis_key for quantity in doses_a if quantity.basis_key}
        basis_b = {quantity.basis_key for quantity in doses_b if quantity.basis_key}
        if basis_a and basis_b and not (basis_a & basis_b):
            return [Conflict(
                "basis", "medium",
                "doses use different, non-convertible bases "
                f"({', '.join(sorted(basis_a))} vs {', '.join(sorted(basis_b))}); "
                "the amounts cannot be compared directly",
                index_a, index_b,
            )]
        dimensions_a = {quantity.dimension for quantity in doses_a}
        dimensions_b = {quantity.dimension for quantity in doses_b}
        if not (dimensions_a & dimensions_b):
            return [Conflict(
                "dimension", "medium",
                "the same product is dosed in incompatible units "
                f"({', '.join(sorted(dimensions_a))} vs {', '.join(sorted(dimensions_b))})",
                index_a, index_b,
            )]
        return []

    shared_products = products_a & products_b
    for dimension, basis in sorted(shared):
        group_a = [quantity for quantity in doses_a if key(quantity) == (dimension, basis)]
        group_b = [quantity for quantity in doses_b if key(quantity) == (dimension, basis)]
        if not group_a or not group_b:
            continue
        best: tuple[float, Quantity, Quantity] | None = None
        for quantity_a in group_a:
            low_a, high_a = quantity_a.normalised_bounds()
            for quantity_b in group_b:
                low_b, high_b = quantity_b.normalised_bounds()
                gap = _interval_gap(low_a, high_a, low_b, high_b)
                if best is None or gap < best[0]:
                    best = (gap, quantity_a, quantity_b)
        if best is None or best[0] < DOSE_GAP_TOLERANCE:
            continue
        _, quantity_a, quantity_b = best
        low_a, high_a = quantity_a.normalised_bounds()
        low_b, high_b = quantity_b.normalised_bounds()
        ratio = _ratio(low_a, high_a, low_b, high_b)
        product_text = f" for {', '.join(sorted(shared_products))}" if shared_products else ""
        conflicts.append(Conflict(
            "dosage", _severity_for_ratio(ratio),
            f"dosage differs{product_text}: {quantity_a.display()} vs {quantity_b.display()}"
            f" (about {ratio:.1f}x apart)",
            index_a, index_b,
        ))
        break  # one dosage conflict per pair is enough for escalation
    return conflicts


def _detect_pair(a: str, b: str, index_a: int, index_b: int) -> list[Conflict]:
    products_a = find_products(a)
    products_b = find_products(b)
    negated_a = negated_products(a)
    negated_b = negated_products(b)

    conflicts: list[Conflict] = []

    # Polarity: a product rejected in one answer and recommended in the other.
    for product in sorted((negated_a & (products_b - negated_b)) | (negated_b & (products_a - negated_a))):
        conflicts.append(Conflict(
            "polarity", "high",
            f"one answer advises against {product} while the other recommends it",
            index_a, index_b,
        ))

    product_conflict = bool(products_a and products_b and not (products_a & products_b))
    if product_conflict:
        conflicts.append(Conflict(
            "product", "high",
            "different products recommended: "
            f"{', '.join(sorted(products_a))} vs {', '.join(sorted(products_b))}",
            index_a, index_b,
        ))

    if not product_conflict:
        conflicts.extend(_dose_conflicts(
            extract_quantities(a), extract_quantities(b), products_a, products_b, index_a, index_b,
        ))

        intervals_a = extract_intervals(a)
        intervals_b = extract_intervals(b)
        if intervals_a and intervals_b:
            candidates = [
                (_interval_gap(interval_a.low, interval_a.high, interval_b.low, interval_b.high),
                 interval_a, interval_b)
                for interval_a in intervals_a
                for interval_b in intervals_b
            ]
            gap, interval_a, interval_b = min(candidates, key=lambda item: item[0])
            if gap >= DOSE_GAP_TOLERANCE:
                ratio = _ratio(interval_a.low, interval_a.high, interval_b.low, interval_b.high)
                conflicts.append(Conflict(
                    "interval", _severity_for_ratio(ratio),
                    f"application interval differs: {interval_a.raw.strip()} vs {interval_b.raw.strip()}",
                    index_a, index_b,
                ))
    return conflicts


def detect_pair(answer_a: str, answer_b: str, index_a: int = 0, index_b: int = 1) -> tuple[Conflict, ...]:
    """Conflicts between two answers (empty tuple means no conflict detected)."""
    if not isinstance(answer_a, str) or not isinstance(answer_b, str):
        return ()
    if not answer_a.strip() or not answer_b.strip():
        return ()
    return tuple(_detect_pair(answer_a, answer_b, index_a, index_b))


def detect_many(answers: Iterable[str], max_answers: int = MAX_ANSWERS) -> ConflictReport:
    """Scan up to ``max_answers`` answers pairwise and summarise escalation.

    The returned report has no consensus answer and no merged text: conflict
    detection never resolves advice.
    """
    cleaned = [answer for answer in answers if isinstance(answer, str) and answer.strip()]
    selected = cleaned[:max_answers]
    found: list[Conflict] = []
    seen: set[tuple[str, str]] = set()
    for index_a, index_b in itertools.combinations(range(len(selected)), 2):
        for conflict in _detect_pair(selected[index_a], selected[index_b], index_a, index_b):
            key = (conflict.kind, conflict.explanation)
            if key in seen:
                continue
            seen.add(key)
            found.append(conflict)

    found.sort(key=lambda conflict: (-SEVERITY_ORDER.get(conflict.severity, 0), conflict.kind,
                                     conflict.answer_a, conflict.answer_b))
    max_severity = found[0].severity if found else None
    kinds: list[str] = []
    for conflict in found:
        if conflict.kind not in kinds:
            kinds.append(conflict.kind)
    reasons = ("ANSWER_CONFLICT",) + tuple(f"{kind.upper()}_CONFLICT" for kind in kinds) if found else ()
    return ConflictReport(
        n_answers=len(selected),
        conflicts=tuple(found),
        escalate=bool(found),
        max_severity=max_severity,
        reasons=reasons,
    )
