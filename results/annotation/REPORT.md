# Human annotation infrastructure — KCC-Sahayak (Batch B, Phase 2)

**Status of the study: PENDING.** fewer than two independent annotators have returned sheets; no human-human kappa exists yet

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

The sample contains **300 English queries** and **900 candidate answers** (300 unique source qids).

| Stratum | Items |
|---|---:|
| common | 80 |
| rare_intent | 55 |
| near_duplicate | 40 |
| ambiguous | 35 |
| insufficient_context | 35 |
| novel | 30 |
| potentially_unsafe | 25 |

The strata are assigned by the eligibility rules in `experiments/annotation/sampling.py`; a query belongs to exactly one stratum. Realised counts differ from the quotas in the script when a stratum ran out of eligible rows, and that shortfall is recorded in `sample_summary.json`.

Existing LLM labels used for stratification (not shown to annotators, not treated as gold):

| LLM label | Items |
|---|---:|
| Animal Husbandry | 41 |
| Cultural Practices | 26 |
| Fisheries | 21 |
| Government Scheme | 33 |
| Market Price | 14 |
| Nutrient Management | 54 |
| Other | 1 |
| Plant Protection | 73 |
| Varietal Information | 23 |
| Weather | 14 |

Candidate composition: `{"distractor_same_topic": 300, "logged": 300, "distractor_other_topic": 300}`.
Each item has one logged answer and two distractors (one same-topic, one different-topic).
Provenance is stored only in `key.jsonl`, never in the annotator interface.

Query length (characters): min 16, median
52.0, max 202.
Maximum corpus cosine similarity per query: `{"min": 0.570638, "median": 0.935238, "max": 1.0}`;
share of queries in a Batch A 0.95 group: `0.43`.

Five redacted examples (PII masked using the Phase 1 redactor):

| Item | Redacted query |
|---|---|
| kcc-0001 | query regarding cotrol measures of fungal diseases in tea |
| kcc-0002 | regarding nutrient application in black pepper |
| kcc-0003 | querry regarding coconut fertilizer dose |
| kcc-0004 | enquired about the loan facilities provided by the govt |
| kcc-0005 | asking for name of rice variety for late transplanting after flood |

## 3. Label set

| Task | Labels |
|---|---|
| Intent | `PP` Plant Protection, `NM` Nutrient Management, `WE` Weather, `GS` Government Scheme, `MP` Market Price, `VI` Varietal Information, `AH` Animal Husbandry, `FI` Fisheries, `CP` Cultural Practices, `OT` Other, `OOS` Out of scope |
| Retrieval relevance | `relevant`, `partially_relevant`, `irrelevant`, `cannot_judge` |
| Answer quality | `adequate`, `partially_adequate`, `inadequate`, `cannot_judge` |
| Safety (query and answer) | `safe`, `caution`, `unsafe`, `cannot_judge` |
| Missing context | `crop`, `crop_stage`, `season`, `location`, `pest_or_disease`, `soil`, `none` |
| Certainty | 1 (guess), 2 (likely), 3 (clear) |
| Out of scope | true / false, structurally tied to `intent == OOS` |

`OT` ("Other") means an agriculture question outside the nine topics; `OOS` means the query is
not about agriculture at all. The schema enforces that distinction rather than leaving it to
interpretation.

## 4. Blinding and provenance separation

| Provenance | Where it lives | Status |
|---|---|---|
| LLM-generated intent labels | existing `data/annotation/annotated.csv`, copied to `key.jsonl` | **not human**, used for stratification only |
| Human annotations | `data/annotations/<annotator>.jsonl`, one shard per person | **PENDING** — 0 item(s) from none |
| Expert verification | nowhere | **PENDING** — no qualified agricultural expert has reviewed answers |
| Adjudicated ground truth | `data/adjudication.jsonl` | **PENDING** — 0 resolved row(s) |

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

**Status: PENDING.** fewer than two independent annotators have returned sheets; no human-human kappa exists yet

No Cohen's or Fleiss' kappa is reported, because there are not yet two independent human annotators with overlapping items. The implemented calculations are covered by unit tests with known values, but those test values are not study results.


## 6. Quality control

Checks: missing items, blank tasks, duplicate/revision records, invalid labels, structural
contradictions, long identical label runs between annotators, and adjudication status.
Revisions found in the current store: **0**.

**Status: PENDING.** no annotator has submitted a sheet yet

- Sample items awaiting annotation: **300**
- Annotator files found: **0**
- Items covered by any annotator: **0**

No coverage, agreement or quality statistic is reported until real sheets exist.


## 7. How to run the study

```bash
# 1. build the frozen 300-query sample (already committed; do not rebuild silently)
.venv312/bin/python experiments/annotation/cli.py build-sample

# 2. generate one offline page per annotator
.venv312/bin/python experiments/annotation/cli.py make-ui --annotator A001 A002

# 3. each annotator fills their own page and exports JSON + CSV
# 4. import each exported file into that annotator's shard
.venv312/bin/python experiments/annotation/cli.py import --annotator A001 \
    experiments/annotation/data/exports/annotations_A001.csv

# 5. check coverage and validity
.venv312/bin/python experiments/annotation/cli.py status
.venv312/bin/python experiments/annotation/cli.py validate

# 6. compute agreement only after two real sheets exist
.venv312/bin/python experiments/annotation/cli.py agreement

# 7. adjudicate a disagreement (a named human, never the tool)
.venv312/bin/python experiments/annotation/cli.py adjudicate --item kcc-0001 \
    --task intent --label PP --adjudicator "A003" --note "pest control is primary"

# 8. regenerate this report
.venv312/bin/python experiments/annotation/cli.py report --overwrite
```

## 8. Limitations and what is still PENDING

- **No human labels, no human-human kappa and no expert review exist yet.** Every human
  result in this report is marked PENDING on purpose.
- The 300-query sample is drawn from the existing corpus, which is not a random sample of
  Indian farmers; it over-represents one call-centre workflow and the LLM annotation pool.
- Very small classes in the sample: `Other` (1). Per-class agreement for these will be unstable and is not a population estimate.
- Only English is included in this pilot. Hindi and Hinglish support exists in the schema and
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
