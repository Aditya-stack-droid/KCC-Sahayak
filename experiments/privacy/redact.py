"""Configurable PII redaction built on :mod:`experiments.privacy.recognizers`.

The module is standalone and has no side effects.  Raw values are never
logged by these helpers; callers receive spans and can decide what to store.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

from .recognizers import ENTITY_TYPES, Detection, detect

DEFAULT_PLACEHOLDERS: dict[str, str] = {
    "PHONE": "[PHONE]",
    "EMAIL": "[EMAIL]",
    "PERSON": "[PERSON]",
    "AADHAAR": "[AADHAAR]",
    "IFSC": "[IFSC]",
    "BANK_ACCOUNT": "[BANK_ACCOUNT]",
    "URL": "[URL]",
}


@dataclass(frozen=True)
class RedactionConfig:
    """Redaction settings.

    enabled_types:
        Subset of entity types to redact.  ``None`` means all supported types.
    placeholders:
        Optional overrides for the default ``[TYPE]`` placeholders.
    policy:
        Passed to :func:`recognizers.detect`; ``"balanced"`` or ``"strict"``.
    """

    enabled_types: Sequence[str] | None = None
    placeholders: Mapping[str, str] = field(default_factory=dict)
    policy: str = "balanced"

    def __post_init__(self) -> None:
        if self.enabled_types is not None:
            unknown = set(self.enabled_types) - set(ENTITY_TYPES)
            if unknown:
                raise ValueError(f"unknown entity types: {sorted(unknown)}")
        unknown = set(self.placeholders) - set(ENTITY_TYPES)
        if unknown:
            raise ValueError(f"unknown placeholder types: {sorted(unknown)}")
        if self.policy not in ("balanced", "strict"):
            raise ValueError("policy must be 'balanced' or 'strict'")


@dataclass(frozen=True)
class RedactionResult:
    original: str
    redacted: str
    detections: tuple[Detection, ...]

    @property
    def changed(self) -> bool:
        return self.original != self.redacted

    def counts(self) -> dict[str, int]:
        counts = {kind: 0 for kind in ENTITY_TYPES}
        for det in self.detections:
            counts[det.type] += 1
        return {kind: n for kind, n in counts.items() if n}


def redact(text: str, config: RedactionConfig | None = None) -> RedactionResult:
    """Return *text* with every detected PII span replaced by a placeholder.

    Replacement is performed from right to left so earlier spans keep their
    original offsets.  Text outside the spans (including whitespace and
    punctuation) is preserved.  Applying the function twice is idempotent
    because placeholders contain no detectable PII.
    """
    cfg = config or RedactionConfig()
    detections = detect(text, policy=cfg.policy, enabled_types=cfg.enabled_types)
    placeholders = {**DEFAULT_PLACEHOLDERS, **dict(cfg.placeholders)}

    redacted = text
    for det in sorted(detections, key=lambda d: d.start, reverse=True):
        placeholder = placeholders.get(det.type, f"[{det.type}]")
        redacted = redacted[: det.start] + placeholder + redacted[det.end:]

    return RedactionResult(original=text, redacted=redacted, detections=detections)


def redact_text(text: str, config: RedactionConfig | None = None) -> str:
    """Convenience wrapper returning only the redacted string."""
    return redact(text, config=config).redacted


def redact_spans(
    text: str,
    spans: Sequence[tuple[int, int, str]],
    placeholders: Mapping[str, str] | None = None,
) -> str:
    """Redact explicit ``(start, end, type)`` spans.

    This is used by the evaluation report to mask **gold** spans as well as
    detected spans.  It therefore keeps missed entities out of reports and logs.
    Overlapping spans are merged before replacement so offsets remain valid.
    """
    merged: list[tuple[int, int, str]] = []
    for start, end, kind in sorted((int(s), int(e), str(t)) for s, e, t in spans):
        if end <= start:
            continue
        if merged and start < merged[-1][1]:
            old_start, old_end, old_kind = merged[-1]
            # Keep the type of the longer original span; the merged region is
            # redacted either way.
            merged[-1] = (
                old_start,
                max(old_end, end),
                old_kind if old_end - old_start >= end - start else kind,
            )
        else:
            merged.append((start, end, kind))

    mapping = {**DEFAULT_PLACEHOLDERS, **dict(placeholders or {})}
    out = text
    for start, end, kind in reversed(merged):
        out = out[:start] + mapping.get(kind, f"[{kind}]") + out[end:]
    return out
