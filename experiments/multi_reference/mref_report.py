"""Render the Improvement 3 report from ``results/multi_reference/metrics.json``.

The report is generated, never hand-edited: every number is read from the
metrics file, and missing structure raises instead of printing a guess.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import mref_refdata as refdata  # noqa: E402

SYSTEMS = (
    ("P1_dense_all", "Dense, all test rows"),
    ("P1_dense_sample", "Dense, 3,000 sample"),
    ("P1_hybrid_sample", "Hybrid, 3,000 sample"),
)
REQUIRED_SUBSETS = ("all", "gold_informative", "pool_ge_2", "pool_eq_1", "pool_ge_20",
                    "novel", "similar", "near_duplicate")


def _f(x, nd: int = 4) -> str:
    return "—" if x is None else f"{x:.{nd}f}"


def _ci(d) -> str:
    lo, hi = d.get("ci95", [None, None])
    if lo is None or hi is None:
        return "—"
    return f"[{lo:.3f}, {hi:.3f}]"


def _mean(metrics: dict, system: str, subset: str, metric: str):
    try:
        return metrics["systems"][system]["subsets"][subset][metric]["mean"]
    except KeyError as exc:
        raise KeyError(f"metrics.json missing {system}/{subset}/{metric}") from exc


def _n(metrics: dict, system: str, subset: str) -> int:
    return int(metrics["systems"][system]["subsets"][subset]["n_rows"])


def render(metrics: dict) -> str:
    for key in ("protocol", "seeds", "reference_pools", "reproduction_gate", "systems",
                "paired_comparisons", "limitations"):
        if key not in metrics:
            raise KeyError(f"metrics.json is missing {key!r}")

    pools = metrics["reference_pools"]
    per_row = pools.get("references_per_test_row") or pools.get("pool_size") or {}
    per_group = pools.get("references_per_group") or {}
    skew = pools.get("skew") or {}
    gate = metrics["reproduction_gate"]
    lines = [
        "# Improvement 3 — multi-reference retrieval evaluation",
        "",
        f"Generated from `results/multi_reference/metrics.json` "
        f"({metrics.get('generated_utc', 'timestamp missing')}). Every value below is read "
        "from that file; nothing is estimated or typed by hand.",
        "",
        "## 1. What this measures, and what it does not",
        "",
        "A retrieved answer is scored against **every** distinct answer KCC advisors "
        "actually logged for the same question group, not only against the one answer "
        "attached to the evaluation row. That answers a narrow question: *does the "
        "retrieved answer agree with some historical advisor answer to this question?*",
        "",
        "- It does **not** measure agricultural correctness. Historical advisor answers "
        "are not verified gold, and some of them contradict each other.",
        "- Semantic similarity is reported separately and is **not** correctness.",
        "- No reference answer is generated, synthesised or inferred; references come "
        "from the logged corpus.",
        "- Human relevance and expert agronomic review remain **PENDING** "
        "(Batch B Phase 2 annotation infrastructure exists but has no human labels).",
        "",
        "## 2. Do the reference pools support multi-reference evaluation?",
        "",
        f"Verdict: **{'YES' if pools['supports_multi_reference'] else 'NO'}** "
        f"(pre-registered criteria: ≥{pools['criteria']['min_share_rows_with_2plus_references']:.0%} "
        "of test rows with ≥2 references and "
        f"≥{pools['criteria']['min_share_diverse_multi_ref_groups']:.0%} of sampled "
        "multi-reference groups containing genuinely different answers).",
        "",
        "| Quantity | Value |",
        "|---|---:|",
        f"| Grouped test rows | {pools['n_test_rows']:,} |",
        f"| Test question groups | {pools['n_test_groups']:,} |",
        f"| Groups with at least one reference | {pools['n_groups_with_references']:,} |",
        f"| Distinct reference answers | {pools['n_unique_references']:,} |",
        f"| Rows with ≥1 reference | {pools['n_rows_with_reference']:,} "
        f"({_f(pools['share_rows_with_reference'] * 100, 1)}%) |",
        f"| Rows with ≥2 references | {pools['n_rows_with_2plus_references']:,} "
        f"({_f(pools['share_rows_with_2plus_references'] * 100, 1)}%) |",
        f"| Rows with ≥5 references | {pools['n_rows_with_5plus_references']:,} "
        f"({_f(pools['share_rows_with_5plus_references'] * 100, 1)}%) |",
        f"| Rows with no usable reference (excluded) | {pools['n_rows_without_reference']:,} "
        f"({_f(pools['share_rows_without_reference'] * 100, 1)}%) |",
        f"| References per test row: mean / median / p90 / max | "
        f"{per_row['mean']} / {per_row['median']} / {per_row['p90']} / {per_row['max']:,} |",
        f"| References per group with ≥1 reference: mean / median / p90 / max | "
        f"{per_group['mean']} / {per_group['median']} / {per_group['p90']} / "
        f"{per_group['max']:,} |",
        f"| Test rows in groups with ≥100 references | "
        f"{skew['n_rows_in_groups_with_ge_100_references']:,} "
        f"({_f(skew['share_rows_in_groups_with_ge_100_references'] * 100, 1)}%) |",
        f"| Mean max pairwise token-F1 inside a group | "
        f"{pools['reference_diversity']['mean_max_pairwise_token_f1']} |",
        f"| Sampled multi-reference groups with max pairwise F1 < 0.80 | "
        f"{_f(pools['reference_diversity']['share_groups_below_0_80'] * 100, 1)}% |",
        "",
        "References are built only from the **test** side of the grouped split, so no "
        "pool can contain an answer that is also in the retrieval index.",
        f"Split check: groups spanning splits = {metrics['split_checks']['groups_spanning_splits']} "
        f"of {metrics['split_checks']['n_groups']:,}.",
        "",
        "## 3. Batch A reproduction gate",
        "",
        f"Result: **{'PASS' if gate['ok'] else 'FAIL'}** — the rebuilt rankings must "
        "reproduce the frozen Batch A numbers before any new metric is reported.",
        "",
        "| Check | Batch A | Recomputed | Abs. diff | OK |",
        "|---|---:|---:|---:|:---:|",
    ]
    for check in gate["checks"]:
        lines.append(
            f"| {check['check']} | {_f(check['batch_a_value'])} | "
            f"{_f(check['recomputed_value'])} | {_f(check['abs_diff'], 8)} | "
            f"{'yes' if check['ok'] else 'NO'} |")
    lines += [
        "",
        f"Sample seed for the hybrid comparison: {gate['sample_seed']}; "
        f"{gate['n_sample_queries']:,} queries in both systems.",
        "",
        "## 4. Primary results (P1: unscreened index, comparable with Batch A)",
        "",
        "`own` = the original single-reference metric; `max` = best over the pool "
        "including the row's own answer (≥ own by construction); `max-other` = best over "
        "the pool **excluding** the row's own answer; `random-other` = mean over five "
        "seeded random non-own references (chance control).",
        "",
        "| Metric | Dense, all test rows | Dense, 3,000 sample | Hybrid, 3,000 sample |",
        "|---|---:|---:|---:|",
        f"| Rows | {_n(metrics, 'P1_dense_all', 'all'):,} | "
        f"{_n(metrics, 'P1_dense_sample', 'all'):,} | "
        f"{_n(metrics, 'P1_hybrid_sample', 'all'):,} |",
    ]
    rows = (
        ("own_reference_f1", "Own-reference token-F1 (original metric)"),
        ("max_reference_f1", "Max-reference token-F1"),
    )
    for key, label in rows:
        vals = [_mean(metrics, s, "all", key) for s, _ in SYSTEMS]
        lines.append(f"| {label} | {_f(vals[0])} | {_f(vals[1])} | {_f(vals[2])} |")
    for key, label in (
        ("max_other_reference_f1", "Max-other-reference token-F1"),
        ("random_other_reference_f1", "Random-other-reference token-F1 (control)"),
        ("max_reference_semantic", "Max semantic similarity (not correctness)"),
    ):
        vals = [_mean(metrics, s, "pool_ge_2", key) for s, _ in SYSTEMS]
        lines.append(f"| {label} | {_f(vals[0])} | {_f(vals[1])} | {_f(vals[2])} |")
    lines += [
        "",
        f"The last three rows use the `pool_ge_2` subset "
        f"(n = {_n(metrics, 'P1_dense_all', 'pool_ge_2'):,} for dense-all), because "
        "excluding the query's own answer is only defined when another reference exists.",
        "",
        "Rows where the single reference scored below 0.5 but another advisor answer "
        "to the same question matched at ≥0.5 (dense, all test rows):",
        "",
    ]
    rec = metrics["systems"]["P1_dense_all"]["subsets"]["pool_ge_2"]["recovered_misses"]
    lines += [
        f"- Comparable rows: {rec['n_comparable']:,}",
        f"- Single-reference misses (own F1 < 0.5): {rec['n_single_reference_miss']:,}",
        f"- Recovered by another reference: {rec['n_recovered_by_another_reference']:,} "
        f"({_f((rec['share_of_misses_recovered'] or 0) * 100, 1)}% of misses)",
        "",
        "## 5. Paired statistical comparisons",
        "",
        "Paired bootstrap over identical queries (2,000 resamples, seed "
        f"{metrics['seeds']['bootstrap']}); NaNs are dropped pairwise and `n` reports the "
        "usable pairs.",
        "",
        "| Comparison | n | Mean delta | 95% CI | Improved | Worse | Cohen's d_z |",
        "|---|---:|---:|---|---:|---:|---:|",
    ]
    for label, cmp_ in metrics["paired_comparisons"].items():
        safe_label = label.replace("|", "\\|")
        lines.append(
            f"| {safe_label} | {cmp_['n']:,} | {_f(cmp_['mean_delta'])} | "
            f"{_ci(cmp_)} | {_f((cmp_['share_improved'] or 0) * 100, 1)}% | "
            f"{_f((cmp_['share_worse'] or 0) * 100, 1)}% | {_f(cmp_['cohens_dz'], 3)} |")
    lines += [
        "",
        "A CI that excludes zero indicates a difference on these queries under this "
        "protocol; it is not a claim about agricultural quality. Comparisons marked "
        "`informative own answer only` drop rows whose logged answer is a call-log note, "
        "so both sides compare answer against answer.",
        "",
        "## 6. Subsets: reference availability and novelty",
        "",
        "Dense retrieval on all test rows, by pool size:",
        "",
        "| Subset | n | Own F1 | Max-other F1 | Random-other F1 |",
        "|---|---:|---:|---:|---:|",
    ]
    for subset, label in (("pool_eq_1", "Exactly one reference"),
                          ("pool_ge_2", "Two or more references"),
                          ("pool_ge_20", "Twenty or more references")):
        lines.append(
            f"| {label} | {_n(metrics, 'P1_dense_all', subset):,} | "
            f"{_f(_mean(metrics, 'P1_dense_all', subset, 'own_reference_f1'))} | "
            f"{_f(_mean(metrics, 'P1_dense_all', subset, 'max_other_reference_f1'))} | "
            f"{_f(_mean(metrics, 'P1_dense_all', subset, 'random_other_reference_f1'))} |")
    diag = metrics.get("diagnostics", {}).get("non_informative_own_rows")
    if diag:
        lines += [
            "",
            "The single-reference row can show a higher own-reference F1 than the "
            "informative-reference maximum. That is expected: "
            f"{diag['n']:,} test rows ({_f(diag['share_of_test_rows'] * 100, 1)}%) have a "
            "call-log note as their own logged answer. The original metric still scores "
            f"the retrieved answer against that note (mean {_f(diag['mean_own_f1_against_note'])}), "
            "which is not an answer, while the reference maximum uses only informative "
            f"references (mean {_f(diag['mean_max_other_f1'])}). "
            "Comparisons restricted to informative own answers are reported in §5.",
        ]
    lines += [
        "",
        "By novelty bucket (dense similarity to the grouped index, Batch A thresholds):",
        "",
        "| Bucket | n | Own F1 | Max F1 | Max-other F1 |",
        "|---|---:|---:|---:|---:|",
    ]
    for subset, label in (("near_duplicate", "Near-duplicate (≥0.90)"),
                          ("similar", "Similar (0.80–0.90)"),
                          ("novel", "Novel (<0.80)")):
        lines.append(
            f"| {label} | {_n(metrics, 'P1_dense_all', subset):,} | "
            f"{_f(_mean(metrics, 'P1_dense_all', subset, 'own_reference_f1'))} | "
            f"{_f(_mean(metrics, 'P1_dense_all', subset, 'max_reference_f1'))} | "
            f"{_f(_mean(metrics, 'P1_dense_all', subset, 'max_other_reference_f1'))} |")
    lines += [
        "",
        "## 7. Screening variant (P2: informative-only candidates)",
        "",
        "The deployed retriever screens call-log notes out of its candidates. This "
        "variant applies the same screen to both systems on the same 3,000 queries.",
        "",
        "| System | n | Own F1 | Max-other F1 |",
        "|---|---:|---:|---:|",
    ]
    for system, label in (("P2_dense_sample", "Dense, screened"),
                          ("P2_hybrid_sample", "Hybrid, screened")):
        lines.append(
            f"| {label} | {_n(metrics, system, 'all'):,} | "
            f"{_f(_mean(metrics, system, 'all', 'own_reference_f1'))} | "
            f"{_f(_mean(metrics, system, 'pool_ge_2', 'max_other_reference_f1'))} |")
    lines += [
        "",
        "## 8. Interpretation",
        "",
    ]
    own_dense = _mean(metrics, "P1_dense_all", "pool_ge_2", "own_reference_f1")
    maxo_dense = _mean(metrics, "P1_dense_all", "pool_ge_2", "max_other_reference_f1")
    rand_dense = _mean(metrics, "P1_dense_all", "pool_ge_2", "random_other_reference_f1")
    sem_dense = _mean(metrics, "P1_dense_all", "pool_ge_2", "max_reference_semantic")
    lines += [
        f"- On rows with an alternative reference, dense own-reference F1 is "
        f"{_f(own_dense)} and max-other-reference F1 is {_f(maxo_dense)}: the same "
        "retrieved answer agrees substantially more with the range of advisor answers "
        "than with the single logged one.",
        f"- The random-reference control scores {_f(rand_dense)}, so part of any gap is "
        "mechanical (more chances to match). The max-other − own delta must be read "
        "against that control, not on its own.",
        f"- Maximum semantic similarity to any reference is {_f(sem_dense)}; it is a "
        "maximum over the pool, so it rises with pool size, and it says the retrieved "
        "wording is close to some historical answer, not that the advice is right.",
        "- Dense vs hybrid differences on identical queries are in the paired table; "
        "they are conditional on one grouped split, not a general ranking of the methods.",
        "",
        "## 9. Limitations",
        "",
    ]
    lines += [f"- {x}" for x in metrics["limitations"]]
    env = metrics.get("environment", {})
    if env:
        versions = env.get("versions", {})
        lines += [
            "",
            "Environment: Python "
            f"{env.get('python')}, {env.get('platform')}; "
            f"numpy {versions.get('numpy')}, scikit-learn {versions.get('sklearn')}, "
            f"faiss {versions.get('faiss')}, sentence-transformers "
            f"{versions.get('sentence_transformers')}.",
        ]
    lines += [
        "",
        "## 10. Reproduce",
        "",
        "```bash",
        "OMP_NUM_THREADS=1 .venv312/bin/python experiments/multi_reference/run_evaluation.py",
        "OMP_NUM_THREADS=1 .venv312/bin/python experiments/multi_reference/report.py",
        "OMP_NUM_THREADS=1 .venv312/bin/python -m pytest tests/unit/test_multi_reference.py -q",
        "```",
        "",
        "The evaluation refuses to report new metrics if the Batch A reproduction gate "
        "fails.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    path = refdata.RESULTS / "metrics.json"
    if not path.exists():
        raise SystemExit(f"missing {path}; run run_evaluation.py first")
    metrics = json.loads(path.read_text(encoding="utf-8"))
    report = render(metrics)
    out = refdata.RESULTS / "REPORT.md"
    out.write_text(report, encoding="utf-8")
    print(f"wrote {out} ({len(report.splitlines())} lines)")


if __name__ == "__main__":
    main()
