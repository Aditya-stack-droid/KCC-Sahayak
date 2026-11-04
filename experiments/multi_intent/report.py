"""Render the multi-intent comparison report from measured metrics."""
from __future__ import annotations


def _n(v, nd=4):
    return "n/a" if v is None else f"{v:.{nd}f}"


def render(metrics: dict) -> str:
    s, m = metrics["single"], metrics["multi"]
    lines = [
        "# Multi-intent detection: single-label baseline vs rule-based decomposition",
        "",
        "**Evidence class:** authored synthetic cases; rule-vs-rule comparison; no model loaded,",
        "no human labels and no expert review. The decomposer is **standalone and escalate-only**",
        "and is not wired into the production gate.",
        "",
        f"**Cases:** {metrics['counts']['total']} — {metrics['counts']['multi']} multi-intent, "
        f"{metrics['counts']['single']} single-request controls, "
        f"{metrics['counts']['safety_positive']} with a high-stakes/time-sensitive intent.",
        "",
        "## 1. Headline comparison",
        "",
        "| Metric | single-label baseline | rule-based decomposition |",
        "|---|---:|---:|",
        f"| Exact intent-set match | {_n(s['exact_set_match'])} | {_n(m['exact_set_match'])} |",
        f"| Micro precision | {_n(s['micro']['precision'])} | {_n(m['micro']['precision'])} |",
        f"| Micro recall | {_n(s['micro']['recall'])} | {_n(m['micro']['recall'])} |",
        f"| Micro F1 | {_n(s['micro']['f1'])} | {_n(m['micro']['f1'])} |",
        f"| Macro F1 | {_n(s['macro']['macro_f1'])} | {_n(m['macro']['macro_f1'])} |",
        f"| Missed-intent rate | {_n(s['missed_intent_rate'])} | {_n(m['missed_intent_rate'])} |",
        f"| Extra-intent rate | {_n(s['extra_intent_rate'])} | {_n(m['extra_intent_rate'])} |",
        f"| Decomposition accuracy | {_n(s['decomposition_accuracy'])} | {_n(m['decomposition_accuracy'])} |",
        f"| Safety-escalation recall | {_n(s['safety']['recall'])} | {_n(m['safety']['recall'])} |",
        f"| Safety-escalation precision | {_n(s['safety']['precision'])} | {_n(m['safety']['precision'])} |",
        "",
        "## 2. By language",
        "",
        "| Language | n | baseline micro-F1 | decomposition micro-F1 | decomposition exact |",
        "|---|---:|---:|---:|---:|",
    ]
    for lang, block in metrics["by_language"].items():
        lines.append(f"| {lang} | {block['n']} | {_n(block['single']['micro']['f1'])} | "
                     f"{_n(block['multi']['micro']['f1'])} | {_n(block['multi']['exact_set_match'])} |")
    lines += [
        "",
        "## 3. Residual failures (not hidden)",
        "",
        f"- Cases where the decomposer **missed** a gold intent: {len(metrics['failures']['multi_missed_intents'])}",
        f"- Cases where the decomposer **added** an intent: {len(metrics['failures']['multi_extra_intents'])}",
        f"- High-risk queries not flagged for escalation: {metrics['failures']['safety_missed'] or 'none'}",
        "",
        "Missed-intent examples:",
        "",
        "| Case | Gold | Predicted | Missing |",
        "|---|---|---|---|",
    ]
    for row in metrics["failures"]["multi_missed_intents"][:20]:
        lines.append(f"| {row['case_id']} | {'; '.join(row['gold'])} | {'; '.join(row['predicted']) or '—'} | {'; '.join(row['missing'])} |")
    lines += ["", "## 4. Limitations", ""]
    lines += [f"- {item}" for item in metrics["limitations"]]
    lines.append("")
    return "\n".join(lines)
