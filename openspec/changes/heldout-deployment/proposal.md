# Proposal

## Why

The frozen PILOT model and standalone budget policy need a held-out integration boundary before end-to-end experiments. This runner makes deployment isolation and actual-versus-estimated budget reporting testable on synthetic data.

## What Changes

- Add a deployment-only runner with matching, disjoint learning/deployment declarations and tuning-estimator validation.
- Route contexts to a fixed PILOT predictor and query-derived input token counts to a tuning-only estimator.
- Report served-only rewards/oracle metrics, terminal insufficient budget, and actual-cost violations without updating the router.

## Capabilities

### New Capabilities

- `heldout-deployment`: Fixed-router held-out deployment with budgeted decisions and honest accounting.

### Modified Capabilities

None.

## Impact

Only scripts/deployment.py, tests/test_deployment.py, docs/deployment.md and this change. Reuse existing environment, PILOT, estimator and cost ledger. No external APIs, tokenizers, embeddings, data modifications or shared-module changes.
