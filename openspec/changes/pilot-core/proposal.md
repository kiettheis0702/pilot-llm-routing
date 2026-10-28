# Proposal

## Why

The published PILOT online equations can be tested independently while real preference training and embeddings remain unavailable. A resumable synthetic math core provides that evidence without pretending to have a trained prior.

## What Changes

- Add a normalized-context, cosine-exploitation policy with explicit caller priors and positive per-arm regularization.
- Add strict JSON model checkpoint save/load and deterministic continuation.
- Test analytic updates, malformed states and evaluator integration on synthetic data only.

## Capabilities

### New Capabilities

- `pilot-learning`: Preference-prior initialized online cosine UCB and validated model persistence.

### Modified Capabilities

None.

## Impact

Only scripts/pilot.py, tests/test_pilot.py, docs/pilot-core.md and this change. Reuse LinearUCB and the learning evaluator; no dependencies, data rewriting, external training, embeddings or cost-policy changes. User authorization covers feasible implementation; parent coordinates Git publication.
