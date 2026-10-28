# Proposal

## Why

B01–B03 now supplies validated learning and metrics, but no usable research baselines or experiment command. Add B04–B07 so algorithm behavior can be tested before preference data and embedding choices are resolved.

## What Changes

- Seeded random/fixed-arm policies, disjoint linear UCB, uncertainty-only exploration, and a documented fixed-window epoch-greedy interpretation.
- A reproducible toy/small RouterBench learning CLI with explicit synthetic features, configuration, code/data fingerprints, raw traces and a static regret plot.
- No claim that synthetic features reproduce the paper's embeddings; no paid APIs or full runs.

## Capabilities

### New Capabilities

- `bandit-policies`: Chosen-feedback baseline algorithms with deterministic reset and validation.
- `baseline-runs`: Bounded offline learning runs and reproducible artifact generation.

### Modified Capabilities

None. Uses the completed baseline-infrastructure change without changing its contract.

## Impact

New scripts/policies.py, scripts/run_baselines.py and tests; existing evaluator is reused. No new dependency. User authorized continued feasible implementation and commit/push on 2026-10-28. Source: published PILOT §3.1.3/§3.2 and Li et al. 2010 disjoint LinUCB.
