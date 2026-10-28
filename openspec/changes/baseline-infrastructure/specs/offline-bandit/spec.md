# Spec Delta

## Purpose

Provide a validated offline bandit simulation boundary that prevents accidental outcome leakage into learning policies.

## ADDED Requirements

### Requirement: Complete fixed outcome catalog
The simulator SHALL reject empty or duplicate identifiers, missing or extra query/arm pairs, rewards outside finite [0,1], and nonfinite or negative actual costs. Every query MUST have the same declared arm catalog; an omitted catalog SHALL be inferred from the complete input.

#### Scenario: Corrupt outcome input
- **WHEN** input contains NaN, an unknown query, a missing pair or a catalog mismatch
- **THEN** construction fails before any feedback is returned

#### Scenario: Valid small catalog
- **WHEN** two queries have complete outcomes for a declared two-arm catalog
- **THEN** construction succeeds without requiring the RouterBench eleven-arm pool

### Requirement: Selected reward information boundary
Learning policies SHALL receive only a finite, nonempty numeric context vector at selection and the selected arm ID, original context and selected reward at update. Actual cost and oracle outcomes MUST remain evaluation-only. Context dimensions MUST agree across queries and policy mutation MUST NOT alter stored contexts.

#### Scenario: Unchosen outcomes change
- **WHEN** unchosen rewards or costs change while selected outcomes and contexts remain fixed
- **THEN** a deterministic policy receives identical inputs and produces identical actions

#### Scenario: Repeated feedback request
- **WHEN** feedback has already been returned for a query
- **THEN** a second request fails; an invalid initial arm request does not consume that query

#### Scenario: Invalid vectors
- **WHEN** contexts are empty, multidimensional, nonfinite or have inconsistent dimensions
- **THEN** validation fails before selection
