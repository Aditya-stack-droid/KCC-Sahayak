"""Render the multi-intent v2 safety report."""
from __future__ import annotations


def _n(v, nd=4):
    return "n/a" if v is None else f"{v:.{nd}f}"


def _block(metrics: dict, key: str) -> list[str]:
    b = metrics[key]
    base, v1, v2 = b["baseline"], b["v1"], b["v2"]
    lines = [
        f"### {key.replace('_', ' ').title()} ({b['n']} cases)",
        "",
        "| Metric | baseline (single) | v1 (decompose) | v2 (risk-first) |",
        "|---|---:|---:|---:|",
        f"| Exact intent-set match | {_n(base['exact_set_match'])} | {_n(v1['exact_set_match'])} | {_n(v2['exact_set_match'])} |",
        f"| Micro precision | {_n(base['micro']['precision'])} | {_n(v1['micro']['precision'])} | {_n(v2['micro']['precision'])} |",
        f"| Micro recall | {_n(base['micro']['recall'])} | {_n(v1['micro']['recall'])} | {_n(v2['micro']['recall'])} |",
        f"| Micro F1 | {_n(base['micro']['f1'])} | {_n(v1['micro']['f1'])} | {_n(v2['micro']['f1'])} |",
        f"| Macro F1 | {_n(base['macro_f1'])} | {_n(v1['macro_f1'])} | {_n(v2['macro_f1'])} |",
        f"| Missed-intent rate | {_n(base['missed_intent_rate'])} | {_n(v1['missed_intent_rate'])} | {_n(v2['missed_intent_rate'])} |",
        f"| Extra-intent rate | {_n(base['extra_intent_rate'])} | {_n(v1['extra_intent_rate'])} | {_n(v2['extra_intent_rate'])} |",
        f"| Decomposition accuracy | {_n(base['decomposition_accuracy'])} | {_n(v1['decomposition_accuracy'])} | {_n(v2['decomposition_accuracy'])} |",
        f"| High-risk intent recall | {_n(base['high_risk_intent_recall'])} | {_n(v1['high_risk_intent_recall'])} | {_n(v2['high_risk_intent_recall'])} |",
        f"| Safety escalation precision | {_n(base['safety_escalation']['precision'])} | {_n(v1['safety_escalation']['precision'])} | {_n(v2['safety_escalation']['precision'])} |",
        f"| Safety escalation recall | {_n(base['safety_escalation']['recall'])} | {_n(v1['safety_escalation']['recall'])} | {_n(v2['safety_escalation']['recall'])} |",
        f"| **High-risk false negatives** | {base['high_risk_fn_count']} | {v1['high_risk_fn_count']} | **{v2['high_risk_fn_count']}** |",
        "",
    ]
    if v2["high_risk_false_negatives"]:
        lines.append("**v2 missed high-risk cases (individually):** " + ", ".join(v2["high_risk_false_negatives"]))
        lines.append("")
    return lines


def render(metrics: dict) -> str:
    lines = [
        "# Multi-intent safety study (v2, risk-first)",
        "",
        "**Evidence class:** authored synthetic cases; rule-based systems; no model training.",
        "The v2 detector is **standalone** and not connected to the production gate.",
        "",
        "Risk intents: " + ", ".join(metrics["protocol"]["risk_intents"]) + ".",
        "",
    ]
    for key in ("original_set", "adversarial_set", "combined"):
        lines += _block(metrics, key)
    lines += ["## Limitations", ""]
    lines += [f"- {x}" for x in metrics["limitations"]]
    lines.append("")
    return "\n".join(lines)
