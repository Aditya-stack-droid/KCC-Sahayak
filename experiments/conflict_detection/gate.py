"""Optional enhanced gate wrapper: escalate when retrieved answers conflict.

The production gate (``src/confidence_gate.py``) is imported read-only and is
never modified.  This wrapper runs the original decision on the first answer
and can only make the outcome stricter: an AUTO_ANSWER becomes ESCALATE when a
conflict is found, and an existing ESCALATE keeps its original code.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from confidence_gate import GateThresholds, decide  # noqa: E402

from .conflicts import ConflictReport, detect_many  # noqa: E402


@dataclass(frozen=True)
class EnhancedDecision:
    action: str
    base_action: str
    base_code: str
    base_reason: str
    reasons: tuple[str, ...]
    conflict_report: ConflictReport

    def as_dict(self) -> dict:
        return {
            "decision": self.action,
            "base_decision": self.base_action,
            "base_code": self.base_code,
            "base_reason": self.base_reason,
            "reasons": list(self.reasons),
            "conflict_report": self.conflict_report.as_dict(),
        }


def decide_with_conflicts(
    answers: Sequence[str],
    query_type: str,
    intent_confidence: float,
    retrieval_similarity: float,
    thresholds: GateThresholds | None = None,
    retrieved_type: str | None = None,
) -> EnhancedDecision:
    """Run the production gate, then add the conflict rule.

    ``answers`` is the top-k retrieved text in rank order.  The first answer is
    what the production gate would have returned, so ``base_action`` is exactly
    that decision.  Conflicting answers can only add escalation.
    """
    th = thresholds or GateThresholds()
    answer_list = [answer for answer in answers if isinstance(answer, str)]
    base = decide(
        query_type=query_type,
        intent_confidence=intent_confidence,
        retrieval_similarity=retrieval_similarity,
        thresholds=th,
        answer=answer_list[0] if answer_list else None,
        retrieved_type=retrieved_type,
    )
    report = detect_many(answer_list)
    if report.conflicts:
        reasons = tuple(dict.fromkeys(["ANSWER_CONFLICT", *report.reasons]))
        return EnhancedDecision(
            action="ESCALATE",
            base_action=base.action,
            base_code=base.code,
            base_reason=base.reason,
            reasons=reasons,
            conflict_report=report,
        )
    return EnhancedDecision(
        action=base.action,
        base_action=base.action,
        base_code=base.code,
        base_reason=base.reason,
        reasons=(),
        conflict_report=report,
    )
