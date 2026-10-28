## Purpose

Validate preference-prior online cosine UCB mathematics independently of unavailable real preference training and preserve learned models for deterministic continuation.

## ADDED Requirements

### Requirement: Explicit valid initialization
The policy SHALL require finite dimension-matched priors, a fixed unique arm catalog, explicit positive finite per-arm lambdas and nonempty provenance. Reset SHALL restore A=lambda I and b=lambda prior. No fallback prior or inferred lambda SHALL be used.

#### Scenario: Analytic initialization
- **WHEN** prior a is [2,0] and lambda a is 2
- **THEN** initial A is 2I, b is [4,0], and theta is [2,0]

### Requirement: Normalized cosine learning
The policy SHALL normalize nonzero finite contexts, select maximum cosine(theta,x)+alpha*sqrt(x solve(A,x)), and update only selected A/b with normalized context and reward. Invalid updates SHALL leave state unchanged. Ties SHALL follow catalog order. Zero query vectors SHALL fail; zero arm vectors SHALL have cosine zero.

#### Scenario: Magnitude invariance
- **WHEN** the same nonzero context is multiplied by a positive scalar
- **THEN** predictions, scores and updates agree within numerical tolerance

### Requirement: Validated model checkpoints
JSON checkpoints SHALL preserve version, ordered catalog, dimension, configuration, prior/provenance, seed, round count and learned A/b. Loading SHALL reject checksum/schema errors, nonfinite or mismatched arrays and nonsymmetric/nonpositive-definite matrices before returning a model. Saving SHALL refuse existing files.

#### Scenario: Resume
- **WHEN** a partially learned model is saved, reloaded and given the same remaining feedback
- **THEN** its actions and state match uninterrupted learning

#### Scenario: Corruption
- **WHEN** an artifact contains changed bytes or a re-signed invalid state
- **THEN** loading fails without modifying any existing policy
