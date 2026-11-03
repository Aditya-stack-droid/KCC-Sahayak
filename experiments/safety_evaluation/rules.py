"""Read-only adapter around the production safety gate.

Nothing in this module changes the shipped rules. It imports the deployed
``src/confidence_gate.py`` and reports what it does with a given piece of text,
so that an adversarial test set can be scored against the *unmodified* system.

Two views are exposed:

``rule_verdict``
    ``answer_risk(text)`` — the content rule alone (dosage / chemical regexes).

``gate_verdict``
    the full gate decision for a query type where escalation can only be
    triggered by the answer text. ``Cultural Practices`` is used because it is
    neither high-stakes nor time-sensitive, and confidence/similarity/topic are
    fixed above threshold. Under those conditions the only remaining reasons to
    escalate are ``uninformative_answer`` and ``risky_answer``, so the decision
    isolates the answer-screening behaviour being tested.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import confidence_gate as production  # noqa: E402

#: Query type used for the content-only gate slice.
GATE_QUERY_TYPE = "Cultural Practices"
GATE_INTENT_CONFIDENCE = 0.95
GATE_RETRIEVAL_SIMILARITY = 0.90


def rule_verdict(text: str | None) -> dict:
    """Run only the production content rule (``answer_risk``)."""
    reason = production.answer_risk(text)
    return {"fired": reason is not None, "reason": reason}


def gate_verdict(text: str | None) -> dict:
    """Run the full production gate with text-triggered escalation isolated."""
    decision = production.decide(
        GATE_QUERY_TYPE,
        GATE_INTENT_CONFIDENCE,
        GATE_RETRIEVAL_SIMILARITY,
        answer=text,
    )
    return {
        "action": decision.action,
        "code": decision.code,
        "reason": decision.reason,
        "escalated": decision.action == "ESCALATE",
    }


def production_source_path() -> Path:
    """Path of the imported production gate, for provenance in reports."""
    return Path(production.__file__).resolve()
