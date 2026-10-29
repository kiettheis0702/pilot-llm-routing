# Spec Delta

## Purpose

Support traceable offline routing research from source decisions through held-out experiments and published aggregate evidence.

## ADDED Requirements

### Requirement: Explicit source and cost evidence
Every experiment SHALL pin data, model pool, embedding recipe, token-count method, price units/date and cost order. Unresolved model bridges SHALL remain excluded from evidence-approved pools. Substitutions and unresolved paper differences SHALL be stated in each report.

#### Scenario: Unresolved eleven-model identity
- **WHEN** a requested pool contains a model without approved identity evidence
- **THEN** the evidence-approved experiment fails and identifies the unresolved model

### Requirement: Separated training and selection
Model priors SHALL use preference training data and declared preference holdout validation. Hyperparameters and mean response-token estimates SHALL use tuning data only. Learning SHALL receive selected-arm feedback only and deployment SHALL freeze learned state.

#### Scenario: Deployment label mutation
- **WHEN** deployment labels change after parameters are frozen
- **THEN** the selected parameters, cost estimates and learned state remain unchanged

### Requirement: Fair deployment comparisons
Every compared method SHALL use the same deployment query order, model pool, cost estimates, budget and declared cost-policy adaptation. Reports SHALL distinguish realized recorded costs from estimates and disclose assumptions about ranking or unserved queries.

#### Scenario: Budget exhaustion
- **WHEN** no permitted arm fits the remaining estimated budget
- **THEN** every method follows the same declared exhaustion rule and records served coverage and realized spend

### Requirement: Reproducible full experiment evidence
Experiments SHALL persist configuration, seeds, input/checkpoint identities, per-method traces and aggregate metrics. Final figures and reports SHALL be derived from completed real runs and distinguish repeated-seed variation from uncertainty over new datasets.

#### Scenario: Missing run
- **WHEN** an expected seed or method trace is absent
- **THEN** the report marks the experiment incomplete rather than silently dropping it

### Requirement: Sequential free local execution
Heavy stages SHALL run one at a time with one worker and the resource preflight. No model-provider generation request SHALL be made without explicit approval for that paid run.

#### Scenario: Insufficient RAM
- **WHEN** available physical RAM is below the required threshold
- **THEN** no new heavy job starts and existing active work remains untouched
