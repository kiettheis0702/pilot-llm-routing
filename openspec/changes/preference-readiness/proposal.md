# Proposal

## Why

The current Data Gate reports broad preference counts but does not provide a strict reusable eligibility boundary or a held-out prompt-group split. P01 and feasible P02/P04 preparation can progress locally without inventing nine model equivalences, embeddings, or trained priors.

## What Changes

- Add strict preference parsing, exact-ID-only eligibility, exclusion counts, per-arm/pair coverage, and deterministic train/holdout preparation.
- Preserve source/config/prepared-input hashes and emit aggregate readiness evidence without storing conversation text in Git.
- Inventory possible local embedding/cache files only in the workspace and chosen data root; missing model identity, dimension, and provenance remain explicit blockers.
- Document unresolved paper choices and report all-arm readiness false when coverage or prerequisites fail. No training, external API calls, downloads, source rewrites, or alias promotion.

## Capabilities

### New Capabilities

- `preference-preparation`: validated exact-ID preference preparation, prompt-group split, provenance, and conservative readiness evidence.

### Modified Capabilities

None; existing audit behavior and raw/prepared RouterBench artifacts remain unchanged.

## Impact

New `scripts/preference_data.py`, CPU fixture tests, `docs/preference-readiness.md`, aggregate JSON reports, and this change's artifacts. Reuses existing pandas/NumPy dependencies and hashing conventions. Local derived pair data may be explicitly exported beneath the selected data root; raw inputs are read-only. The user has authorized feasible implementation in this session; the coordinating agent alone manages Git.
