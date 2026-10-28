# Design

## Context

Reuse stable PILOT.predict_rewards, TuningTokenCostEstimator, BinnedCostPolicy, ActualCostLedger and OfflineBandit. The learning runner intentionally resets/updates a policy and cannot be repurposed for deployment.

## Goals / Non-Goals

Integrate a frozen cosine predictor with an estimated-budget deployment stream. No policy training/reset, linear-score clipping, external tokenization/pricing, dataset changes or scientific result claims.

## Decisions

- Add run_deployment with both learning and deployment SplitSelections. Validate phases, manifest/settings equality, unique/nonempty IDs, known-query coverage and disjoint membership. Call estimator.validate_deployment and reject tuning/learning overlap. These declarations do not attest the model's actual training history.
- Accept only a PILOT instance and a tuning estimator. Compare catalog sets; decision ties follow the model catalog order. Construct a fresh internal budget policy and ledger to avoid accidental reuse.
- Preflight copies selected contexts and query-derived token counts, checks dimensions/nonzero vectors, and validates every estimate before decisions. During each round, predict cosine from a fresh context, estimate from query-only counts, select by budget, reveal selected reward/cost, then account for actual cost and offline oracle metrics. No outcome table crosses either policy API.
- Return dataclass trace and JSON-compatible summary. A failed budget choice is a terminal decision, not a zero-reward observation. Record requested/served/unserved counts and served-only means (null when none served). Actual overspend remains visible but does not feed back into estimated decisions, matching the existing cost policy.
- Fingerprint selected inputs and frozen learned A/b/configuration for reproducibility. Record caller provenance and estimator metadata. No filesystem output API is added; downstream CLI owns serialization.

## Risks / Trade-offs

Python APIs and provenance labels are trusted local boundaries, not a hostile-code sandbox. Estimated affordability cannot guarantee actual budget compliance; the ledger reports violations. Empirical oracle regret is unconstrained and covers served rows only, not a budget-aware optimal baseline. Preflight estimation repeats the stateless estimator calculation to fail early on malformed future inputs.
