# Proposal

## Why

A05–A07 require a cost estimator and an independent deployment budget policy. The current contracts expose historical costs only after selection and correctly leave deployment estimates unset; using those actual costs before choice would leak response information.

## What Changes

- Add a pure estimator fitted from caller-supplied per-arm tuning output counts, explicit prices and verified split selections; accept query input counts at estimation time.
- Implement the published PILOT Algorithm 3 threshold, bin allocation, rollover, fallback and explicit insufficient-budget state with a hard estimated affordability guard.
- Add a separate selected-cost ledger that reports actual overspend without changing policy decisions.
- Document paper interpretations, blocked RouterBench integration inputs, and analytic/boundary/mutation test evidence. No full experiment, training, downloads or API calls.

## Capabilities

### New Capabilities

- `deployment-cost-control`: Tuning-only token estimation, independent binned budget decisions, and separate actual-cost accounting.

### Modified Capabilities

None. Learning remains unbudgeted and existing split/evaluator APIs are unchanged.

## Impact

New `scripts/cost_policy.py`, `tests/test_cost_policy.py`, and `docs/cost-policy.md`; no additional dependencies. Real-data token counting and historical prices are prerequisites for future integration, not inferred from stored actual deployment costs. Scope follows the authorized A05–A07 standalone work in the supplied handoff.
