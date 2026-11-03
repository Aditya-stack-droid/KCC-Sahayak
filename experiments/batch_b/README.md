# Batch B — additive experiments for KCC-Sahayak

Batch B improves the research quality of KCC-Sahayak **without modifying the
original project or Batch A**. Every script in this directory is additive and
writes only to new paths.

## Hard rules

- No edits to `src/`, `backend/`, `frontend/`, `models/`, `data/`, `scripts/`,
  existing `tests/`, `paper/`, `docs/`, `README.md` or
  `research/EXPERIMENT_LOG.md`.
- No writes to `results/improved_evaluation/` (Batch A baseline).
- No original dataset, model or result is overwritten.
- No human labels, kappa values, expert reviews or performance improvements are
  fabricated. Human/expert work is marked **PENDING**.
- No Batch C work.

## Phase order (approved)

| Phase | Scope | Status |
|---:|---|---|
| 0 | Baseline verification and protection | complete |
| 1 | PII detection and redaction | complete (synthetic evaluation) |
| 2 | Human annotation infrastructure | complete (infrastructure only; human labels PENDING) |
| 3 | Independent safety evaluation | complete for Improvement 2 (synthetic adversarial set; no expert review) |
| 4 | Answer conflict detection | complete for Improvement 1 (synthetic pair set + descriptive corpus scan; no expert adjudication) |
| 5 | Out-of-scope and multi-intent detection | not started |
| 6 | Extended retrieval evaluation | not started |
| 7 | Behavioural and regression testing | not started |
| 8 | Documentation and final comparison | not started |

## Baseline manifest

`manifests/baseline_hashes.json` records SHA-256 hashes for Batch A outputs,
Batch A code/artefacts, original data, models, source, backend, scripts,
documents and existing tests.

```bash
# create once (already done in Phase 0)
.venv312/bin/python experiments/batch_b/verify_baseline.py --init

# verify at the start and end of every phase
.venv312/bin/python experiments/batch_b/verify_baseline.py --check
```

`--check` exits non-zero on any missing/changed file or unexpected addition in
an immutable directory. It never writes.

`manifests/batch_b_protection_hashes.json` (308 files: 220 original, 50 Batch A,
14 Phase 1, 24 Phase 2) is the protection baseline for Improvements 1 onwards.
It is never rebuilt, so it still describes the pre-Improvement-1 state. After an
approved improvement adds files, verify it with `--allow-additions`: new files
are listed, while changed or missing protected files still fail the check.
`manifests/batch_b_improvement1_hashes.json` is a post-Improvement-1 inventory
(325 files) used for a clean zero-addition check.

## Full protection inventory

`verify_phase_outputs.py` hard-codes its phases and is itself inside the frozen
308-file snapshot, so extending its scan list would make it report its own file
as changed. The scan definition therefore lives in data:

- `inventories/protection.json` — roots, groups, explicitly listed files
  (focused tests, the config itself), immutable prefixes and annotation
exemptions;
- `verify_inventory.py` — builds and checks a manifest from that config;
- `manifests/batch_b_protection_full.json` — **339 files**: 220 original, 50
  Batch A, 14 Phase 1, 24 Phase 2, 17 Improvement 1, 13 Improvement 2, 1 config.

```bash
.venv312/bin/python experiments/batch_b/verify_inventory.py --check
```

After an approved improvement adds files, add its roots to the config and
rebuild with `--snapshot --force`; `--allow-additions` verifies only that no
protected file changed while the new files are reviewed. The older 308-file and
325-file manifests are intentionally left untouched as historical baselines.

## Environment

- macOS Apple Silicon, CPU only; Python 3.12.13 (`.venv312`).
- No new dependency is required for Batch B.
- Pinned versions used: see `requirements-batch-b.txt`.
- Use `OMP_NUM_THREADS=1` for PyTorch/FAISS runs.
