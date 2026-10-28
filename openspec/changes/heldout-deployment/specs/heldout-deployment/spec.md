## Purpose

Run held-out queries through a fixed learned cosine router and estimated cost policy, preserving training isolation and reporting observed deployment accounting honestly.

## ADDED Requirements

### Requirement: Validated held-out inputs
The runner SHALL require disjoint learning/deployment selections from the same declared manifest/settings and a compatible tuning estimator. It SHALL validate query coverage, catalogs, nonzero finite contexts, and query-only token count coverage before any budget decision. Learning provenance SHALL be labeled caller-declared.

#### Scenario: Invalid lineage
- **WHEN** selections overlap, manifests disagree or estimator tuning IDs overlap deployment
- **THEN** the run fails before a budget decision or router mutation

### Requirement: Frozen selected-outcome deployment
The runner SHALL use PILOT cosine predictions without reset or learning updates. Only contexts SHALL reach the router. Only predictions and estimated costs SHALL reach budget selection. Selected actual outcomes and oracle metrics SHALL be evaluated after the choice and SHALL NOT affect future choices.

#### Scenario: Perturbed held-out outcomes
- **WHEN** held-out costs or unchosen rewards change with contexts and estimates fixed
- **THEN** choices remain unchanged and only observed accounting or oracle metrics change

### Requirement: Complete honest accounting
The report SHALL include per-served-query decisions, actual costs, rewards, empirical oracle regret, budget ledger, requested/served/unserved counts and input provenance. Insufficient budget SHALL terminate explicitly without imputed outcomes. Means SHALL be null when none are served; actual overspend SHALL remain visible.

#### Scenario: Early stop
- **WHEN** the budget cannot select an arm for a query
- **THEN** the failed decision is reported, remaining queries stay unserved and no reward is fabricated

#### Scenario: Final partial bin
- **WHEN** three queries use a bin size of two
- **THEN** decision bins are 1,1,2 with the last bin's remaining-query count equal to one
