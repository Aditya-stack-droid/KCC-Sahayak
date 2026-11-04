"""Render the answer-screening comparison report from measured metrics."""
from __future__ import annotations


def _pct(v):
    return "n/a" if v is None else f"{v * 100:.2f}%"


def _num(v, nd=4):
    return "n/a" if v is None else f"{v:.{nd}f}"


def _confusion(stats: dict) -> str:
    c = stats["confusion_matrix"]
    return (f"TP={c['tp']} FP={c['fp']} FN={c['fn']} TN={c['tn']}")


def render(metrics: dict) -> str:
    v1, v2 = metrics["v1"], metrics["v2"]
    o1, o2 = v1["overall"], v2["overall"]
    lines = [
        "# Answer-screening evaluation: shipped rule vs negation/measurement-aware screener",
        "",
        "**Evidence class:** authored synthetic cases, deterministic, no model training and no human",
        "or expert review. The shipped production gate is **unchanged**; this measures a candidate",
        "rule (`answer_screening.screening`) against the frozen rule (`confidence_gate.answer_risk`).",
        "",
        "**Cases:** {total} total — {unsafe} unsafe, {safe} safe, {amb} ambiguous (excluded from P/R/F1).".format(
            total=metrics["counts"]["total"], unsafe=metrics["counts"]["unsafe"],
            safe=metrics["counts"]["safe"], amb=metrics["counts"]["ambiguous_excluded"]),
        "",
        "## 1. Headline comparison",
        "",
        "| Metric | v1 (shipped) | v2 (candidate) | Change |",
        "|---|---:|---:|---:|",
        f"| True positives | {o1['tp']} | {o2['tp']} | {o2['tp'] - o1['tp']:+d} |",
        f"| False positives | {o1['fp']} | {o2['fp']} | {o2['fp'] - o1['fp']:+d} |",
        f"| False negatives | {o1['fn']} | {o2['fn']} | {o2['fn'] - o1['fn']:+d} |",
        f"| True negatives | {o1['tn']} | {o2['tn']} | {o2['tn'] - o1['tn']:+d} |",
        f"| Precision | {_num(o1['precision'])} | {_num(o2['precision'])} | |",
        f"| Recall | {_num(o1['recall'])} | {_num(o2['recall'])} | |",
        f"| F1 | {_num(o1['f1'])} | {_num(o2['f1'])} | |",
        f"| F1 95% CI | [{o1['f1_95ci']['low']}, {o1['f1_95ci']['high']}] | [{o2['f1_95ci']['low']}, {o2['f1_95ci']['high']}] | |",
        f"| **Unsafe-answer acceptance rate** | {_pct(o1['unsafe_acceptance_rate'])} | {_pct(o2['unsafe_acceptance_rate'])} | {metrics['comparison']['unsafe_acceptance_rate_change']:+} |",
        f"| **Negation false-positive rate** | {_pct(v1['negation_fp']['negation_false_positive_rate'])} | {_pct(v2['negation_fp']['negation_false_positive_rate'])} | {metrics['comparison']['negation_false_positive_rate_change']:+} |",
        "",
        f"Confusion matrices — v1: {_confusion(o1)}; v2: {_confusion(o2)}.",
        "",
        "## 2. Before / after by language",
        "",
        "| Language | n | v1 recall | v2 recall | v1 FP | v2 FP |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for lang in sorted(v1["languages"]):
        a, b = v1["languages"][lang], v2["languages"][lang]
        lines.append(f"| {lang} | {a['n']} | {_num(a['recall'])} | {_num(b['recall'])} | {a['fp']} | {b['fp']} |")

    lines += [
        "",
        "## 3. Negation situations (Priority 1)",
        "",
        f"Of {v1['negation_fp']['safe_negation_cases']} safe negation/warning cases, "
        f"v1 flagged {v1['negation_fp']['flagged']} and v2 flagged {v2['negation_fp']['flagged']}.",
        "",
        "| Situation | n | v1 flagged | v2 flagged |",
        "|---|---:|---:|---:|",
    ]
    situations = sorted(set(v1["situations"]) | set(v2["situations"]))
    for sit in situations:
        n = v1["situations"].get(sit, {}).get("n", 0)
        # flagged counts are not in the block; recompute from fp+tp is wrong for mixed cases,
        # so report FP/TN split through false_positive lists is impractical here; use block metrics.
        lines.append(f"| {sit} | {n} | FP={v1['situations'].get(sit, {}).get('fp', 'n/a')} | FP={v2['situations'].get(sit, {}).get('fp', 'n/a')} |")

    lines += [
        "",
        "## 4. Measurement classes (Priority 3)",
        "",
        "v2 escalates household/container/packaging measures it cannot interpret; v1 does not",
        "recognise most of them. Recall by situation is in `metrics.json`.",
        "",
        "## 5. Residual errors (not hidden)",
        "",
        f"- v2 false negatives: {', '.join(v2['false_negatives']) or 'none'}",
        f"- v2 false positives: {', '.join(v2['false_positives']) or 'none'}",
        f"- New false negatives vs v1: {', '.join(metrics['comparison']['new_false_negatives_vs_v1']) or 'none'}",
        f"- New false positives vs v1: {', '.join(metrics['comparison']['new_false_positives_vs_v1']) or 'none'}",
        "",
        "## 6. Limitations",
        "",
    ]
    lines += [f"- {item}" for item in metrics["limitations"]]
    lines.append("")
    return "\n".join(lines)
