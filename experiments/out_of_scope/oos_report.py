"""Render the Improvement 4 report from ``results/out_of_scope/metrics.json``.

Generated, never hand-edited: every number comes from the metrics file, and a
missing required block raises instead of printing a guess.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import oos_data  # noqa: E402

APPROACHES = (
    ("knn_distance", "A. kNN distance"),
    ("embedding_lr", "B. Embedding + logistic regression"),
)


def _f(x, nd: int = 4) -> str:
    return "—" if x is None else f"{x:.{nd}f}"


def _pct(x) -> str:
    return "—" if x is None else f"{100 * x:.1f}%"


def render(metrics: dict) -> str:
    for key in ("protocol", "seeds", "data", "thresholds", "benchmark",
                "full_in_scope_test", "by_register_out_of_scope", "ambiguous",
                "mixed_intent", "evidence", "threshold_sensitivity", "limitations"):
        if key not in metrics:
            raise KeyError(f"metrics.json is missing {key!r}")

    data = metrics["data"]
    synth = data["synthetic"]
    bench = metrics["benchmark"]
    lines = [
        "# Improvement 4 — out-of-scope detection",
        "",
        f"Generated from `results/out_of_scope/metrics.json` "
        f"({metrics.get('generated_utc', 'timestamp missing')}). Every value below is "
        "read from that file.",
        "",
        "> **Evidence class: synthetic negatives.** Out-of-scope queries were authored "
        "for this study by its authors. They are not real farmer traffic and are not a "
        "representative sample. In-scope queries are real logged KCC questions from "
        "Batch A's grouped test split.",
        "",
        "## 1. What this measures, and what it does not",
        "",
        "The task is binary: is a query agricultural (in scope for KCC-Sahayak) or not? "
        "Two standalone detectors are compared — a kNN distance rule over the existing "
        "grouped index, and a logistic-regression baseline over the existing MiniLM "
        "embeddings. Neither is wired into production.",
        "",
        "- It does **not** estimate real-world out-of-scope prevalence; the synthetic "
        "negatives are not a sample of what callers actually ask.",
        "- It does **not** measure multilingual in-scope performance: the corpus is "
        "English by the project register detector, so Hindi and Hinglish appear only "
        "as synthetic negatives.",
        "- It does **not** replace the production intent classifier, retriever or safety "
        "gate.",
        "",
        "## 2. Data and split integrity",
        "",
        "| Quantity | Value |",
        "|---|---:|",
        f"| Corpus rows (Batch A grouped split) | {data['corpus_rows']:,} |",
        f"| Corpus test rows (agricultural) | {data['corpus_split_counts'].get('test', 0):,} |",
        f"| Synthetic out-of-scope queries | {synth['n_queries']} |",
        f"| Synthetic question families | {synth['n_families']} |",
        f"| Categories | {synth['n_categories']} |",
        f"| Families per split (train / val / test) | "
        f"{synth['per_split_families'].get('train', 0)} / "
        f"{synth['per_split_families'].get('val', 0)} / "
        f"{synth['per_split_families'].get('test', 0)} |",
        f"| Queries per split (train / val / test) | "
        f"{synth['per_split_queries'].get('train', 0)} / "
        f"{synth['per_split_queries'].get('val', 0)} / "
        f"{synth['per_split_queries'].get('test', 0)} |",
        f"| Synthetic families spanning splits | "
        f"{data['split_checks']['synthetic_families_spanning_splits']} |",
        f"| Corpus question groups spanning splits | "
        f"{data['split_checks']['corpus_groups_spanning_splits']} |",
        "",
        "Each family holds the English, Hindi and Hinglish realisation of one question, "
        "and the family is the split unit, so no translation or paraphrase of a test "
        "question is visible during training or threshold selection.",
        "",
        f"Corpus test register mix (5,000-question sample): "
        f"{data['corpus_registers_test_sample']}.",
        "",
        "## 3. Threshold selection (validation only)",
        "",
        "| Approach | Threshold | Validation FRR | Validation recall | n (in / out) |",
        "|---|---:|---:|---:|---:|",
    ]
    for key, label in APPROACHES:
        t = metrics["thresholds"][key]
        lines.append(
            f"| {label} | {_f(t['threshold'])} | {_pct(t['val_false_rejection_rate'])} | "
            f"{_pct(t['val_recall'])} | {t['n_val_in_scope']:,} / "
            f"{t['n_val_out_of_scope']} |")
    lines += [
        "",
        metrics["thresholds"]["knn_distance"]["policy"] + ".",
        "",
        "## 4. Benchmark results",
        "",
        f"Benchmark: {bench['knn_distance']['n_in_scope']} agricultural grouped-test "
        f"queries + {bench['knn_distance']['n_out_of_scope']} synthetic out-of-scope "
        "queries. Positive class = out-of-scope.",
        "",
        "| Approach | Precision | Recall | F1 | F1 95% CI | Balanced acc. | AUROC |",
        "|---|---:|---:|---:|---|---:|---:|",
    ]
    for key, label in APPROACHES:
        b = bench[key]
        lines.append(
            f"| {label} | {_f(b['precision'])} | {_f(b['recall'])} | {_f(b['f1'])} | "
            f"{b['f1_ci95']} | {_f(b['balanced_accuracy'])} | {_f(b['auroc'])} |")
    lines += [
        "",
        "| Approach | False acceptance rate (out-of-scope let through) | "
        "False rejection rate (agricultural rejected) |",
        "|---|---:|---:|",
    ]
    for key, label in APPROACHES:
        b = bench[key]
        lines.append(
            f"| {label} | {_pct(b['false_acceptance_rate'])} | "
            f"{_pct(b['false_rejection_rate'])} |")
    lines += [
        "",
        "### Confusion matrices",
        "",
        "| Approach | TP | FN (out-of-scope accepted) | FP (agricultural rejected) | TN |",
        "|---|---:|---:|---:|---:|",
    ]
    for key, label in APPROACHES:
        c = bench[key]["confusion"]
        lines.append(f"| {label} | {c['tp']} | {c['fn']} | {c['fp']} | {c['tn']} |")
    lines += [
        "",
        "## 5. Full agricultural test set (not a sample)",
        "",
        "| Approach | Agricultural test rows | Rejected | False rejection rate |",
        "|---|---:|---:|---:|",
    ]
    for key, label in APPROACHES:
        f = metrics["full_in_scope_test"][key]
        lines.append(
            f"| {label} | {f['n']:,} | {f['n_rejected']:,} | "
            f"{_pct(f['false_rejection_rate'])} |")
    lines += [
        "",
        "## 6. By register (synthetic negatives only)",
        "",
        "| Register | n | A. kNN rejection | B. LR rejection |",
        "|---|---:|---:|---:|",
    ]
    for register in ("en", "hi", "hinglish"):
        entry = metrics["by_register_out_of_scope"].get(register)
        if not entry:
            continue
        lines.append(
            f"| {register} | {entry['n']} | "
            f"{_pct(entry['knn_distance']['rejection_rate'])} "
            f"({entry['knn_distance']['n_rejected']}/{entry['n']}) | "
            f"{_pct(entry['embedding_lr']['rejection_rate'])} "
            f"({entry['embedding_lr']['n_rejected']}/{entry['n']}) |")
    lines += [
        "",
        "No in-scope Hindi or Hinglish queries exist in the corpus, so this table says "
        "nothing about false rejections for those registers.",
        "",
        "## 7. Ambiguous and mixed-intent behaviour",
        "",
        "These sets are descriptive only: they are excluded from threshold selection and "
        "from the headline benchmark.",
        "",
        "| Set | Approach | n | in_scope | ambiguous | out_of_scope | Rejected |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for set_name, key in (("Ambiguous", "ambiguous"), ("Mixed intent", "mixed_intent")):
        for approach, label in APPROACHES:
            entry = metrics[key][approach]
            counts = entry["decision_counts"]
            lines.append(
                f"| {set_name} | {label} | {entry['n']} | "
                f"{counts.get('in_scope', 0)} | {counts.get('ambiguous', 0)} | "
                f"{counts.get('out_of_scope', 0)} | {_pct(entry['rejection_rate'])} |")
    ev = metrics["evidence"]
    lines += [
        "",
        "Agricultural-evidence override: a query scored below the rejection band but "
        "containing a crop/pest/agronomy term is flagged `ambiguous` (needs review) "
        "rather than silently rejected.",
        "",
        "| Set | n | With agricultural evidence |",
        "|---|---:|---:|",
    ]
    for name, label in (("corpus_val_sample", "Agricultural validation sample"),
                        ("synthetic_test_out_of_scope", "Synthetic test negatives"),
                        ("ambiguous", "Ambiguous set"),
                        ("mixed_intent", "Mixed-intent set")):
        block = ev[name]
        lines.append(f"| {label} | {block['n']} | {block['with_evidence']} |")
    lines += [
        "",
        "## 8. Threshold sensitivity (descriptive, not used for selection)",
        "",
        "| Approach | Threshold | FAR | FRR | F1 |",
        "|---|---:|---:|---:|---:|",
    ]
    for key, label in APPROACHES:
        for row in metrics["threshold_sensitivity"][key]:
            lines.append(
                f"| {label} | {row['threshold']} | {_pct(row['false_acceptance_rate'])} | "
                f"{_pct(row['false_rejection_rate'])} | {_f(row['f1'])} |")
    lines += [
        "",
        "## 9. Interpretation",
        "",
    ]
    for key, label in APPROACHES:
        b = bench[key]
        lines.append(
            f"- **{label}** rejects {_pct(b['recall'])} of synthetic negatives while "
            f"rejecting {_pct(b['false_rejection_rate'])} of agricultural benchmark "
            f"queries; AUROC {_f(b['auroc'])}.")
    lines += [
        "- Neither detector is integrated into production; both are standalone and "
        "evaluated offline.",
        "- A higher synthetic recall is not evidence of real-world safety or usefulness: "
        "the negatives share author, domain and vocabulary with each other.",
        "",
        "## 10. Limitations",
        "",
    ]
    lines += [f"- {x}" for x in metrics["limitations"]]
    env = metrics.get("environment", {})
    if env:
        versions = env.get("versions", {})
        lines += [
            "",
            f"Environment: Python {env.get('python')}, {env.get('platform')}; "
            f"numpy {versions.get('numpy')}, scikit-learn {versions.get('sklearn')}, "
            f"faiss {versions.get('faiss')}, sentence-transformers "
            f"{versions.get('sentence_transformers')}.",
        ]
    lines += [
        "",
        "## 11. Reproduce",
        "",
        "```bash",
        "OMP_NUM_THREADS=1 .venv312/bin/python experiments/out_of_scope/run_oos_evaluation.py",
        "OMP_NUM_THREADS=1 .venv312/bin/python experiments/out_of_scope/oos_report.py",
        "OMP_NUM_THREADS=1 .venv312/bin/python -m pytest tests/unit/test_out_of_scope.py -q",
        "```",
        "",
        f"Fixed seeds: dataset {metrics['seeds']['dataset']}, bootstrap "
        f"{metrics['seeds']['bootstrap']}. Thresholds are selected on validation data "
        "only; the test split is scored once.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    path = oos_data.RESULTS / "metrics.json"
    if not path.exists():
        raise SystemExit(f"missing {path}; run run_oos_evaluation.py first")
    metrics = json.loads(path.read_text(encoding="utf-8"))
    text = render(metrics)
    out = oos_data.RESULTS / "REPORT.md"
    out.write_text(text, encoding="utf-8")
    print(f"wrote {out} ({len(text.splitlines())} lines)")


if __name__ == "__main__":
    main()
