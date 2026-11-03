# Annotation quality-control report (Batch B, Phase 2)

**Study status: PENDING.** Generated from `experiments/annotation/data/` only.

## Current state

**Status: PENDING.** no annotator has submitted a sheet yet

- Sample items awaiting annotation: **300**
- Annotator files found: **0**
- Items covered by any annotator: **0**

No coverage, agreement or quality statistic is reported until real sheets exist.


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
