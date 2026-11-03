"""Write the Phase 2 documentation package.

The report separates four provenances explicitly:

1. LLM-generated labels that already exist in the project (used only for
   stratification),
2. human annotations (none exist yet unless annotator files are present),
3. expert verification (never claimed; no expert has reviewed this sample),
4. adjudicated ground truth (only where a named adjudicator resolved a row).

Every agreement number is read from ``agreement.json``; when fewer than two
independent annotators exist the report says PENDING and prints no kappa.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from statistics import median
from typing import Any, Iterable, Sequence

from . import agreement, quality, schema
from . import store as store_mod


def _redact(text: str) -> str:
    """Redact text with the Phase 1 module; never fall back to raw text."""
    try:
        from experiments.privacy.redact import redact_text

        return redact_text(str(text))
    except Exception:
        return "[redaction unavailable: text withheld]"


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _table(header: Sequence[str], rows: Iterable[Sequence[Any]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(_fmt(value) for value in row) + " |")
    return "\n".join(lines)


def _similarity_summary(values: Sequence[float]) -> dict:
    if not values:
        return {}
    ordered = sorted(values)
    return {
        "min": round(ordered[0], 6),
        "median": round(median(ordered), 6),
        "max": round(ordered[-1], 6),
    }


def sample_overview(store: store_mod.Store) -> dict:
    items = store.load_items()
    key = store.load_key()
    summary = store.load_summary()
    if not items:
        return {"status": "PENDING", "reason": "sample not built"}

    records = [key[item["item_id"]] for item in items if item["item_id"] in key]
    strata = Counter(record["stratum"] for record in records)
    llm_labels = Counter(record["llm_label"] for record in records)
    provenance = Counter(
        candidate["provenance"] for record in records for candidate in record["candidates"]
    )
    pools = Counter(record["source"]["pool"] for record in records)
    similarities = [record["max_corpus_similarity"] for record in records]
    group_sizes = [record["corpus_group_size"] for record in records]
    questions = [item["query"] for item in items]

    return {
        "status": "OK",
        "n_items": len(items),
        "n_candidates": sum(len(item["candidates"]) for item in items),
        "n_unique_source_qids": len({record["source"]["qid"] for record in records}),
        "language": Counter(item.get("language", "") for item in items),
        "strata": dict(strata),
        "existing_llm_label_counts": dict(llm_labels),
        "existing_llm_label_provenance": "LLM-generated (Claude); used for stratification only; never human gold",
        "candidate_provenance_counts": dict(provenance),
        "source_pool_counts": dict(pools),
        "max_corpus_similarity": _similarity_summary(similarities),
        "corpus_group_size": {
            "min": min(group_sizes),
            "median": median(group_sizes),
            "max": max(group_sizes),
            "share_in_95_group": round(sum(1 for size in group_sizes if size >= 2) / len(group_sizes), 4),
        },
        "query_length": {
            "min": min(len(text) for text in questions),
            "median": median(len(text) for text in questions),
            "max": max(len(text) for text in questions),
        },
        "redacted_examples": [
            {"item_id": item["item_id"], "query": _redact(item["query"])} for item in items[:5]
        ],
        "stored_summary": summary,
    }


def _agreement_markdown(report: dict) -> str:
    if report.get("status") != "OK":
        return (
            f"**Status: {report.get('status', 'PENDING')}.** {report.get('reason', '')}\n\n"
            "No Cohen's or Fleiss' kappa is reported, because there are not yet two independent "
            "human annotators with overlapping items. The implemented calculations are covered by "
            "unit tests with known values, but those test values are not study results.\n"
        )

    lines = ["| Task | Pair | n common | Cohen's kappa | Observed agreement |", "|---|---|---:|---:|---:|"]
    for task, info in report["tasks"].items():
        if task in schema.CANDIDATE_TASKS:
            info = info["aggregate"]
        for pair, stats in info.get("pairs", {}).items():
            lines.append(
                f"| {task} | {pair} | {stats['n_common']} | {_fmt(stats['kappa'])} | "
                f"{_fmt(stats['observed_agreement'])} |"
            )
    if report.get("human_vs_llm", {}).get("status") == "OK":
        lines.append("")
        lines.append("| Annotator | n | Accuracy vs LLM | Cohen's kappa vs LLM |")
        lines.append("|---|---:|---:|---:|")
        for annotator, stats in report["human_vs_llm"]["annotators"].items():
            lines.append(
                f"| {annotator} | {stats['n']} | {_fmt(stats['accuracy_vs_llm'])} | {_fmt(stats['kappa'])} |"
            )
    return "\n".join(lines) + "\n"


def _qc_markdown(qc: dict) -> str:
    if qc.get("status") == "PENDING":
        covered = qc.get("n_items", 0) - len(qc.get("items_uncovered_by_any_annotator", []))
        return (
            f"**Status: PENDING.** {qc.get('reason', '')}\n\n"
            f"- Sample items awaiting annotation: **{qc.get('n_items', 0)}**\n"
            f"- Annotator files found: **{qc.get('n_annotators', 0)}**\n"
            f"- Items covered by any annotator: **{covered}**\n\n"
            "No coverage, agreement or quality statistic is reported until real sheets exist.\n"
        )
    lines = [
        f"- Annotators with files: **{qc['n_annotators']}**",
        f"- Items covered by at least two annotators: **{qc['items_with_two_or_more_annotators']}**",
        f"- Items not covered by anyone: **{len(qc['items_uncovered_by_any_annotator'])}**",
        f"- Adjudication records: **{qc['adjudication']['records']}** "
        f"({qc['adjudication']['resolved']} resolved)",
        f"- Disagreements awaiting adjudication: **{qc['adjudication']['pending_count']}**",
        "",
        "| Annotator | Items | Coverage | Items missing tasks | Duplicates | Warnings |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for annotator, info in qc["per_annotator"].items():
        lines.append(
            f"| {annotator} | {info['items_annotated']} | {info['coverage']:.2%} | "
            f"{info['items_with_missing_tasks']} | {len(info['duplicate_items'])} | "
            f"{'; '.join(info['warnings']) or '—'} |"
        )
    if qc.get("consistency"):
        lines += ["", "| Pair | Overlapping items | Longest identical run | Copy flag |",
                  "|---|---:|---:|---|"]
        for pair, info in qc["consistency"].items():
            lines.append(
                f"| {pair} | {info['overlapping_items']} | {info['longest_identical_intent_run']} | "
                f"{'yes' if info['possible_copy_flag'] else 'no'} |"
            )
    return "\n".join(lines) + "\n"


def render_report(store: store_mod.Store) -> tuple[str, str, dict, dict, dict]:
    overview = sample_overview(store)
    agree = agreement.agreement_for(store)
    qc = quality.qc_report(store)
    revision_count = sum(1 for count in store.revision_counts().values() if count > 1)

    if overview.get("status") != "OK":
        sample_section = "The annotation sample has not been built yet."
        strata_rows: list[tuple] = []
    else:
        strata_rows = sorted(overview["strata"].items(), key=lambda item: (-item[1], item[0]))
        sample_section = (
            f"The sample contains **{overview['n_items']} English queries** and "
            f"**{overview['n_candidates']} candidate answers** ({overview['n_unique_source_qids']} unique source qids).\n\n"
            "| Stratum | Items |\n|---|---:|\n"
            + "\n".join(f"| {name} | {count} |" for name, count in strata_rows)
            + "\n\nThe strata are assigned by the eligibility rules in "
            "`experiments/annotation/sampling.py`; a query belongs to exactly one stratum. "
            "Realised counts differ from the quotas in the script when a stratum ran out of "
            "eligible rows, and that shortfall is recorded in `sample_summary.json`."
        )

    status = agree.get("status", "PENDING")
    contributing = ", ".join(agree.get("annotators", [])) or "none"
    small_classes = [
        f"`{name}` ({count})" for name, count in overview.get("existing_llm_label_counts", {}).items()
        if count < 5
    ]
    small_note = ""
    if small_classes:
        small_note = (
            "- Very small classes in the sample: " + ", ".join(sorted(small_classes)) +
            ". Per-class agreement for these will be unstable and is not a population estimate.\n"
        )

    report = f"""# Human annotation infrastructure — KCC-Sahayak (Batch B, Phase 2)

**Status of the study: {status}.** {agree.get('reason', '') or ''}

No human annotation exists in this repository unless annotator files are present under
`experiments/annotation/data/annotations/`. No expert has reviewed this sample. Nothing in
this report is expert-verified ground truth, and no kappa value is reported from
LLM-generated labels.

## 1. What this phase delivers

| Component | File |
|---|---|
| Label schema, normalisation and validation | `experiments/annotation/schema.py` |
| Stratified 300-query sample builder | `experiments/annotation/sampling.py` |
| Append-only per-annotator store, import/export, adjudication | `experiments/annotation/store.py` |
| Cohen's kappa, Fleiss' kappa, confusion matrices, disagreements | `experiments/annotation/agreement.py` |
| Coverage, duplicate, consistency and adjudication checks | `experiments/annotation/quality.py` |
| Offline single-file annotation interface | `experiments/annotation/ui.py` |
| Report writer | `experiments/annotation/report.py` |
| Command line entry point | `experiments/annotation/cli.py` |
| Focused tests | `tests/unit/test_annotation.py` |

## 2. Sample composition (measured)

{sample_section}

Existing LLM labels used for stratification (not shown to annotators, not treated as gold):

| LLM label | Items |
|---|---:|
""" + "\n".join(
        f"| {name} | {count} |"
        for name, count in sorted(overview.get("existing_llm_label_counts", {}).items())
    ) + f"""

Candidate composition: `{json.dumps(overview.get('candidate_provenance_counts', {}))}`.
Each item has one logged answer and two distractors (one same-topic, one different-topic).
Provenance is stored only in `key.jsonl`, never in the annotator interface.

Query length (characters): min {overview.get('query_length', {}).get('min', '—')}, median
{overview.get('query_length', {}).get('median', '—')}, max {overview.get('query_length', {}).get('max', '—')}.
Maximum corpus cosine similarity per query: `{json.dumps(overview.get('max_corpus_similarity', {}))}`;
share of queries in a Batch A 0.95 group: `{overview.get('corpus_group_size', {}).get('share_in_95_group', '—')}`.

Five redacted examples (PII masked using the Phase 1 redactor):

| Item | Redacted query |
|---|---|
""" + "\n".join(
        f"| {row['item_id']} | {row['query'].replace('|', '/')} |"
        for row in overview.get("redacted_examples", [])
    ) + f"""

## 3. Label set

| Task | Labels |
|---|---|
| Intent | {", ".join(f"`{code}` {name}" for code, name in schema.INTENT_CODES.items())} |
| Retrieval relevance | {", ".join(f"`{label}`" for label in schema.RELEVANCE_LABELS)} |
| Answer quality | {", ".join(f"`{label}`" for label in schema.QUALITY_LABELS)} |
| Safety (query and answer) | {", ".join(f"`{label}`" for label in schema.SAFETY_LABELS)} |
| Missing context | {", ".join(f"`{label}`" for label in schema.CONTEXT_LABELS)} |
| Certainty | 1 (guess), 2 (likely), 3 (clear) |
| Out of scope | true / false, structurally tied to `intent == OOS` |

`OT` ("Other") means an agriculture question outside the nine topics; `OOS` means the query is
not about agriculture at all. The schema enforces that distinction rather than leaving it to
interpretation.

## 4. Blinding and provenance separation

| Provenance | Where it lives | Status |
|---|---|---|
| LLM-generated intent labels | existing `data/annotation/annotated.csv`, copied to `key.jsonl` | **not human**, used for stratification only |
| Human annotations | `data/annotations/<annotator>.jsonl`, one shard per person | **{status}** — {agree.get('n_items_annotated', 0)} item(s) from {contributing} |
| Expert verification | nowhere | **PENDING** — no qualified agricultural expert has reviewed answers |
| Adjudicated ground truth | `data/adjudication.jsonl` | **PENDING** — {qc.get('adjudication', {}).get('resolved', 0)} resolved row(s) |

Blinding is structural: `make-ui --annotator NAME` embeds only that person's items and the
interface has no code path that reads another annotator's shard, the answer key, the strata or
the LLM labels. `key.jsonl` must be stored separately from the annotator files.

## 5. Agreement methodology and current result

Methods implemented (and unit-tested against known worked examples):

- **Cohen's kappa** for every annotator pair, per task, using complete cases for that pair;
  the number of items excluded for each side is reported (`n_missing_left` / `n_missing_right`).
- **Fleiss' kappa** when three or more annotators exist, on items rated by all of them.
- **Observed and expected agreement**, **confusion matrices**, and **per-class one-vs-rest
  kappa** with class support.
- **Set-valued agreement** (exact-set match and mean Jaccard) for multi-intent and
  missing-context, because kappa is not defined for label sets.
- **Human vs LLM** comparison, reported separately and never labelled human consensus.
- **Disagreement lists** prepared for human adjudication. No LLM resolves them.

{_agreement_markdown(agree)}

## 6. Quality control

Checks: missing items, blank tasks, duplicate/revision records, invalid labels, structural
contradictions, long identical label runs between annotators, and adjudication status.
Revisions found in the current store: **{revision_count}**.

{_qc_markdown(qc)}

## 7. How to run the study

```bash
# 1. build the frozen 300-query sample (already committed; do not rebuild silently)
.venv312/bin/python experiments/annotation/cli.py build-sample

# 2. generate one offline page per annotator
.venv312/bin/python experiments/annotation/cli.py make-ui --annotator A001 A002

# 3. each annotator fills their own page and exports JSON + CSV
# 4. import each exported file into that annotator's shard
.venv312/bin/python experiments/annotation/cli.py import --annotator A001 \\
    experiments/annotation/data/exports/annotations_A001.csv

# 5. check coverage and validity
.venv312/bin/python experiments/annotation/cli.py status
.venv312/bin/python experiments/annotation/cli.py validate

# 6. compute agreement only after two real sheets exist
.venv312/bin/python experiments/annotation/cli.py agreement

# 7. adjudicate a disagreement (a named human, never the tool)
.venv312/bin/python experiments/annotation/cli.py adjudicate --item kcc-0001 \\
    --task intent --label PP --adjudicator "A003" --note "pest control is primary"

# 8. regenerate this report
.venv312/bin/python experiments/annotation/cli.py report --overwrite
```

## 8. Limitations and what is still PENDING

- **No human labels, no human-human kappa and no expert review exist yet.** Every human
  result in this report is marked PENDING on purpose.
- The 300-query sample is drawn from the existing corpus, which is not a random sample of
  Indian farmers; it over-represents one call-centre workflow and the LLM annotation pool.
{small_note}- Only English is included in this pilot. Hindi and Hinglish support exists in the schema and
  interface but no Hindi/Hinglish items were drawn, so no cross-lingual claim is made.
- Retrieval relevance is judged on **candidate answers** (one logged answer plus two
  distractors), not on a live retrieval system's top-k. Phase 6 evaluates the retriever itself.
- The interface contains real corpus text. It is a local research tool with no authentication;
  files must stay on an authorised machine, and nothing from them may be pasted into public
  services.
- Inter-annotator agreement measures consistency, not correctness. Two annotators can agree and
  both be wrong; expert review is the missing step.
- The sample is frozen by hash; re-running `build-sample` without `--overwrite` fails, so a
  later report cannot silently describe a different sample.
"""

    qc_md = f"""# Annotation quality-control report (Batch B, Phase 2)

**Study status: {status}.** Generated from `experiments/annotation/data/` only.

## Current state

{_qc_markdown(qc)}

## What each check means

- **Coverage** — sampled items with no record from that annotator. Missing means not yet
  annotated, never "assumed safe".
- **Items missing tasks** — a record exists but some required label is blank. These are
  reported per task; imports accept partially complete records but QC counts them.
- **Duplicate items** — the same annotator saved the same item more than once. The store is
  append-only, the latest record wins, and every revision is kept.
- **Invalid labels** — records that no longer satisfy the schema. Imports reject an entire
  file if any row is invalid instead of silently dropping rows.
- **Consistency** — structural contradictions (OOS vs intent, multi-intent not containing the
  primary intent, `none` combined with another missing-context value) plus a long identical
  intent run between two annotators, which is a warning to inspect, not proof of copying.
- **Adjudication** — disagreements with no resolved adjudication record. Adjudication requires
  a named human; the tool cannot resolve a row on its own.

## PENDING

- Human annotation of the 300-query sample.
- A second independent annotator for every item.
- Expert review of answer correctness and potential harm.
- Adjudication of all disagreements by a qualified person.

No value in this report is an estimate of human agreement.
"""
    return report, qc_md, overview, agree, qc


def write_report(store: store_mod.Store, out_dir: str | Path, overwrite: bool = False) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    report_md, qc_md, overview, agree, qc = render_report(store)

    written: list[str] = []
    for name, text in (("REPORT.md", report_md), ("QC_REPORT.md", qc_md)):
        path = out / name
        if path.exists() and not overwrite:
            raise FileExistsError(f"refusing to overwrite {path} (pass --overwrite)")
        path.write_text(text, encoding="utf-8")
        written.append(str(path))

    for name, payload in (
        ("sample_overview.json", overview),
        ("agreement.json", agree),
        ("qc.json", qc),
    ):
        path = out / name
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        written.append(str(path))

    tables = out / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    if overview.get("status") == "OK":
        (tables / "strata.csv").write_text(
            "stratum,n\n" + "\n".join(f"{name},{count}" for name, count in overview["strata"].items()),
            encoding="utf-8",
        )
        (tables / "existing_llm_labels.csv").write_text(
            "llm_label,n\n" + "\n".join(
                f'"{name}",{count}' for name, count in overview["existing_llm_label_counts"].items()
            ),
            encoding="utf-8",
        )
        written.extend([str(tables / "strata.csv"), str(tables / "existing_llm_labels.csv")])

    if agree.get("status") == "OK":
        for task, info in agree["tasks"].items():
            if task in schema.CANDIDATE_TASKS:
                info = info["aggregate"]
            for pair, stats in info.get("pairs", {}).items():
                matrix = stats.get("confusion", {})
                if not matrix:
                    continue
                path = tables / f"confusion_{task}_{pair.replace('|', '_')}.csv"
                categories = matrix["categories"]
                lines = ["label/" + pair.split("|")[1] + "," + ",".join(categories)]
                for row in categories:
                    lines.append(row + "," + ",".join(str(matrix["counts"][row][col]) for col in categories))
                path.write_text("\n".join(lines), encoding="utf-8")
                written.append(str(path))

    return {"report": str(out / "REPORT.md"), "qc": str(out / "QC_REPORT.md"), "written": written}
