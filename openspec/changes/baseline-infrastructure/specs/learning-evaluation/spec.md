# Spec Delta

## Purpose

Execute learning streams with selected-arm feedback and reproducible offline reward, cost and empirical oracle metrics.

## ADDED Requirements

### Requirement: Learning-only lifecycle
The runner SHALL reset the policy once with the run seed, select one arm and issue one selected-reward update per learning query in frozen order. Non-learning selections, missing contexts, duplicate or unknown query IDs and invalid actions MUST fail. No learning budget constraint SHALL be imposed.

#### Scenario: Held-out selection
- **WHEN** a tuning or deployment selection is passed to the learning runner
- **THEN** the runner fails before resetting or updating the policy

### Requirement: Offline metrics after selection
The evaluator SHALL compute per-round and cumulative reward, actual selected cost, action counts and empirical oracle regret equal to the maximum stored reward for the query minus selected reward. Full outcomes MUST NOT be passed to the policy. Empty runs MUST fail instead of reporting misleading means.

#### Scenario: Analytic two-arm example
- **WHEN** selected rewards are 0.25 and 1, oracle rewards are 1 and 1, and selected costs are 0.1 and 0.2
- **THEN** total reward is 1.25, mean reward is 0.625, cost is 0.3 and cumulative empirical oracle regret is 0.75

### Requirement: Reproducible run logs
Each saved run SHALL contain a per-round trace and JSON summary with run seed, policy identity, arm catalog, split/subset provenance, context label/dimension/checksum and data fingerprints. Nonfinite metrics MUST NOT serialize. Saving MUST refuse an existing output directory.

#### Scenario: Repeated seeded execution
- **WHEN** the same seeded policy, inputs and selection are run twice
- **THEN** traces and metric summaries match and serialization preserves the measured values
