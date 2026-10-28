# Proposal

## Why

G0 verified the stored RouterBench data and split, but the environment trusts upstream validation and there is no numeric policy boundary or evaluator. B01–B03 must make offline learning experiments reproducible before implementing research algorithms.

## What Changes

- Validate the simulator's fixed arm catalog, complete outcome grid, identifiers, rewards and costs; preserve existing valid smoke calls.
- Introduce numeric context and OnlinePolicy contracts with chosen-reward-only feedback.
- Verify and load the existing split manifest with pinned checksum and seed; select labeled deterministic subsets without rewriting data.
- Add learning-only evaluation with per-round traces, selected cost, action counts and empirical oracle regret.
- Add analytic and negative tests, plus a small local-data verification report.

## Capabilities

### New Capabilities

- `offline-bandit`: Validated simulator and policy information boundary.
- `frozen-splits`: Reproducible split loading, verification and subset provenance.
- `learning-evaluation`: Learning-only execution, offline metrics and run logging.

### Modified Capabilities

None; this is the repository's first OpenSpec change.

## Impact

Reuse `scripts/data_gate.py`; add focused modules and tests in the existing Python layout. No new dependencies, dataset downloads, embedding models, paid calls, training, deployment budget logic, baseline algorithms or Git publication. The supplied post-Data-Gate roadmap and implementation tickets are the upstream requirements; [G0 audit](../../../docs/data-gate-audit.md) narrows this change to B01–B03. The user explicitly approved repo-local OpenSpec initialization and implementation on 2026-10-28.
