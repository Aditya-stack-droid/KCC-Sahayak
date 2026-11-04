"""Deterministic metrics and Markdown rendering for behavioural probes."""

from __future__ import annotations

import hashlib
import json
from collections import Counter

CATEGORIES = {
    "A": "Typo and spelling robustness",
    "B": "Hindi / Roman-Hindi / Hinglish",
    "C": "Negation and prohibitions",
    "D": "Numbers, units, dosage and ambiguity",
    "E": "Ambiguous or underspecified queries",
    "F": "Multi-intent queries",
    "G": "Empty, malformed and oversized input",
    "H": "Cross-component interactions",
}


def canonical_hash(cases) -> str:
    payload = json.dumps(cases, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def summarize(records, cases) -> dict:
    total = len(records)
    passed = sum(record["status"] == "pass" for record in records)
    failed = total - passed
    by_category = {}
    for code, label in CATEGORIES.items():
        selected = [record for record in records if record["category"] == code]
        failures = Counter(record["failure_type"] for record in selected if record["status"] == "fail")
        by_category[code] = {
            "label": label,
            "total": len(selected),
            "passed": sum(record["status"] == "pass" for record in selected),
            "failed": sum(record["status"] == "fail" for record in selected),
            "failure_types": dict(sorted(failures.items())),
        }
    failure_types = Counter(record["failure_type"] for record in records if record["status"] == "fail")
    probes = Counter(record["probe"] for record in records)
    crashes = sum(bool(record["error"]) for record in records)
    return {
        "schema": 1,
        "experiment": "batch_b_improvement5_behavioural_regression",
        "evidence_class": "authored synthetic behavioural probes; no human labels or expert review",
        "synthetic": True,
        "n_cases": total,
        "passed": passed,
        "failed": failed,
        "pass_rate": round(passed / total, 4) if total else 0.0,
        "crashes": crashes,
        "case_sha256": canonical_hash(cases),
        "categories": [code for code in CATEGORIES],
        "by_category": by_category,
        "by_failure_type": dict(sorted(failure_types.items())),
        "by_probe": dict(sorted(probes.items())),
        "interpretation": (
            "Expected values encode the behaviour this study treats as correct; a failure "
            "is a documented capability gap, not a harness error. These are synthetic probes, "
            "not a representative sample and not a production safety or accuracy estimate."
        ),
    }


def _cell(value, limit: int = 150) -> str:
    if isinstance(value, str):
        text = value
    else:
        text = json.dumps(value, ensure_ascii=False, sort_keys=True)
    text = " ".join(text.split()).replace("|", "\\|")
    if len(text) > limit:
        return text[: limit - 1] + "…"
    return text


def render(metrics: dict, records) -> str:
    failures = [record for record in records if record["status"] == "fail"]
    lines = [
        "# Behavioural regression report (Improvement 5)",
        "",
        "**Evidence class:** authored synthetic probes, deterministic execution, no model",
        "training and no human labels or expert review. This is a behavioural audit,",
        "not a real-world accuracy or safety estimate.",
        "",
        f"**Cases:** {metrics['n_cases']}  ",
        f"**Passed:** {metrics['passed']}  ",
        f"**Failed:** {metrics['failed']}  ",
        f"**Pass rate:** {metrics['pass_rate']:.2%}  ",
        f"**Unhandled crashes:** {metrics['crashes']}  ",
        f"**Case-set SHA-256:** `{metrics['case_sha256']}`",
        "",
        "Expected values encode the behaviour the study treats as correct for robustness,",
        "safety or interface handling. A failed case records a capability gap; it is not",
        "an exception in the harness. Production and previous experiment outputs are not",
        "modified by this run.",
        "",
        "## Coverage",
        "",
        "| Category | Scope | Cases | Passed | Failed | Failure types |",
        "|---|---|---:|---:|---:|---|",
    ]
    for code in metrics["categories"]:
        block = metrics["by_category"][code]
        failure_text = ", ".join(f"{name} {count}" for name, count in block["failure_types"].items()) or "—"
        lines.append(
            f"| {code} | {block['label']} | {block['total']} | {block['passed']} | "
            f"{block['failed']} | {failure_text} |"
        )

    lines.extend([
        "",
        "## Failure types",
        "",
        "| Type | Cases |",
        "|---|---:|",
    ])
    for name, count in metrics["by_failure_type"].items():
        lines.append(f"| {name} | {count} |")

    lines.extend([
        "",
        "## Failures by category",
        "",
        "All failing cases are listed. Full machine-readable records are in `failures.json`;",
        "every case, including passing controls, is in `cases.csv`.",
        "",
    ])
    for code in metrics["categories"]:
        selected = [record for record in failures if record["category"] == code]
        if not selected:
            continue
        lines.extend([
            f"### {code} — {CATEGORIES[code]}",
            "",
            "| ID | Probe | Input | Expected | Observed | Type |",
            "|---|---|---|---|---|---|",
        ])
        for record in selected:
            lines.append(
                f"| {record['case_id']} | {record['probe']} | {_cell(record['input'])} | "
                f"{_cell(record['expected'])} | {_cell(record['observed'])} | "
                f"{record['failure_type']} |"
            )
        lines.append("")

    lines.extend([
        "## Passing controls",
        "",
        f"{metrics['passed']} cases passed. They cover clean English baselines, empty-input",
        "safety, API length/type validation, clause-local negation, unit conversion,",
        "conflict escalation, privacy redaction, gate routing and out-of-scope decisions.",
        "See `cases.csv` for the complete per-case record.",
        "",
        "## Protocol",
        "",
        "1. Cases are authored in `br_cases.py`; each is marked synthetic and has one",
        "   expected value fixed before the evaluator was run.",
        "2. `br_probes.py` calls existing production or Batch B components read-only.",
        "3. The evaluator writes `metrics.json`, `cases.csv`, `failures.json` and this",
        "   report under `results/behavioural_regression/` only.",
        "4. No threshold, rule, lexicon or model was tuned on these cases.",
        "",
        "## What this establishes",
        "",
        "- A reproducible behavioural baseline for the components named in `cases.csv`.",
        "- Separately counted false positives and false negatives, plus missing",
        "  capabilities and validation gaps.",
        "- Explicit evidence that multi-intent detection is absent: mixed queries still",
        "  receive one weak-label category and no multi-intent signal.",
        "- Explicit evidence of language, typo, unit, negation and PII edge cases.",
        "",
        "## What this does not establish",
        "",
        "- Real-world frequency of any failure. The cases are authored probes.",
        "- End-to-end retrieval or answer correctness: no MiniLM/FAISS model is loaded.",
        "- ASR accuracy; only transcript normalisation is exercised.",
        "- Human or agronomist agreement, severity, or production readiness.",
        "- That a passing case is agronomically correct; it only matches the expected",
        "  component behaviour.",
        "",
        "## Reproduce",
        "",
        "```bash",
        "OMP_NUM_THREADS=1 .venv312/bin/python experiments/behavioural_regression/run_evaluation.py",
        "OMP_NUM_THREADS=1 .venv312/bin/python -m pytest tests/unit/test_behavioural_regression.py -q",
        "```",
        "",
    ])
    return "\n".join(lines)
