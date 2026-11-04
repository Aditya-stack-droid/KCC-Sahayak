"""Read-only adapters from authored cases to existing project components."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
OOS = ROOT / "experiments" / "out_of_scope"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(OOS) not in sys.path:
    sys.path.insert(0, str(OOS))

from asr import normalize_transcript  # noqa: E402
from confidence_gate import GateThresholds, answer_risk, decide  # noqa: E402
from lang import detect_register  # noqa: E402
from ner import extract_entities  # noqa: E402
from weak_labels import label_query  # noqa: E402

from experiments.conflict_detection.conflicts import detect_many, detect_pair  # noqa: E402
from experiments.conflict_detection.gate import decide_with_conflicts  # noqa: E402
from experiments.out_of_scope.oos_detectors import agricultural_evidence, decide as oos_decide  # noqa: E402
from experiments.privacy.recognizers import detect  # noqa: E402
from experiments.privacy.redact import redact_text  # noqa: E402


def _load_clean_text():
    path = ROOT / "scripts" / "01_prepare_data.py"
    spec = importlib.util.spec_from_file_location("kcc_prepare_data", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.clean_text


clean_text = _load_clean_text()
THRESHOLDS = GateThresholds()


def _query_request_status(payload: str) -> str:
    from backend.app import QueryRequest
    try:
        QueryRequest.model_validate_json(payload)
        return "valid"
    except Exception:
        return "invalid"


def _malformed_call(fn_name: str, value) -> str:
    functions = {
        "extract_entities": extract_entities,
        "label_query": label_query,
        "detect_register": detect_register,
        "answer_risk": answer_risk,
    }
    try:
        functions[fn_name](value)
        return "returned"
    except (AttributeError, TypeError, ValueError):
        return "controlled_error"
    except Exception:
        return "uncontrolled_error"


def _asdict_decision(decision) -> dict:
    return {"action": decision.action, "code": decision.code}


def run_probe(case: dict):
    probe = case["probe"]
    text = case.get("text", "")

    if probe == "register":
        return detect_register(text)
    if probe == "entities":
        return extract_entities(text)
    if probe == "intent":
        return label_query(text)
    if probe == "risk":
        return answer_risk(text) is not None
    if probe == "conflict":
        return sorted({item.kind for item in detect_pair(text, case["text_b"])})
    if probe == "privacy_types":
        return [item.type for item in detect(text)]
    if probe == "redaction":
        return redact_text(text)
    if probe == "redaction_idempotent":
        once = redact_text(text)
        return redact_text(once) == once
    if probe == "evidence":
        return agricultural_evidence(text)
    if probe == "oos_decision":
        decision = oos_decide(case["score"], case["threshold"], case["margin"], text)
        return decision.decision
    if probe == "multi_intent":
        label_query(text)
        return "single_label_only"
    if probe == "gate":
        decision = decide(
            case["query_type"],
            case["confidence"],
            case["similarity"],
            thresholds=THRESHOLDS,
            answer=case.get("answer"),
            retrieved_type=case.get("retrieved_type"),
        )
        return _asdict_decision(decision)
    if probe == "conflict_many":
        return detect_many(case["answers"]).escalate
    if probe == "conflict_gate":
        decision = decide_with_conflicts(
            case["answers"],
            case["query_type"],
            case["confidence"],
            case["similarity"],
            thresholds=THRESHOLDS,
        )
        return decision.action
    if probe == "clean_then_intent":
        return label_query(clean_text(text))
    if probe == "clean_then_entities":
        return extract_entities(clean_text(text))
    if probe == "asr_then_intent":
        return label_query(normalize_transcript(text))
    if probe == "malformed_call":
        return _malformed_call(case["fn"], case.get("value"))
    if probe == "query_request":
        return _query_request_status(case["payload"])
    raise KeyError(f"unknown probe: {probe}")


def compare(case: dict, observed) -> bool:
    if "expected_nonempty" in case:
        return bool(observed) is bool(case["expected_nonempty"])
    return case["expected"] == observed


def classify_failure(case: dict, observed) -> str:
    if "failure_type" in case:
        return case["failure_type"]
    probe = case["probe"]
    expected = case.get("expected")

    if probe in {"risk", "conflict_many"}:
        if expected and not observed:
            return "false_negative"
        if not expected and observed:
            return "false_positive"
        return "degraded"

    if probe == "conflict":
        expected_kinds = set(expected or [])
        observed_kinds = set(observed or [])
        if expected_kinds - observed_kinds:
            return "false_negative"
        if observed_kinds - expected_kinds:
            return "false_positive"
        return "degraded"

    if probe == "privacy_types":
        expected_kinds = set(expected or [])
        observed_kinds = set(observed or [])
        if expected_kinds - observed_kinds:
            return "false_negative"
        if observed_kinds - expected_kinds:
            return "false_positive"
        return "degraded"

    if probe in {"entities", "clean_then_entities"}:
        for field in ("crop", "symptom"):
            expected_values = set((expected or {}).get(field, []))
            observed_values = set((observed or {}).get(field, []))
            if expected_values - observed_values:
                return "false_negative"
            if observed_values - expected_values:
                return "false_positive"
        return "degraded"

    if probe in {"gate", "conflict_gate"}:
        expected_action = expected["action"] if isinstance(expected, dict) else expected
        observed_action = observed["action"] if isinstance(observed, dict) else observed
        if expected_action == "ESCALATE" and observed_action == "AUTO_ANSWER":
            return "false_negative"
        if expected_action == "AUTO_ANSWER" and observed_action == "ESCALATE":
            return "false_positive"
        return "degraded"

    if probe == "multi_intent":
        return "missing_capability"
    if probe == "query_request":
        return "validation_gap"
    return "degraded"


def display_input(case: dict) -> str:
    if "text_b" in case:
        return f"{case.get('text', '')} || {case['text_b']}"
    if "payload" in case:
        return case["payload"]
    return case.get("text", "")


def evaluate_all(cases) -> list[dict]:
    records = []
    for case in cases:
        error = ""
        try:
            observed = run_probe(case)
        except Exception as exc:
            observed = f"raised:{type(exc).__name__}"
            error = f"{type(exc).__name__}: {exc}"
        passed = compare(case, observed)
        records.append({
            "case_id": case["id"],
            "category": case["category"],
            "probe": case["probe"],
            "input": display_input(case),
            "expected": case.get("expected", case.get("expected_nonempty")),
            "observed": observed,
            "status": "pass" if passed else "fail",
            "failure_type": "" if passed else classify_failure(case, observed),
            "note": case.get("note", ""),
            "synthetic": True,
            "error": error,
        })
    return records


def canonical_cases(cases) -> str:
    return json.dumps(cases, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
