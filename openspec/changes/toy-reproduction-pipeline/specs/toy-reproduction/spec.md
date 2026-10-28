## Purpose

Exercise the complete synthetic preference-to-deployment path without requiring external models, private data or paid services.

## ADDED Requirements

### Requirement: One-command synthetic reproduction
The command SHALL train a small labeled synthetic prior, validate its positive supported inverse accuracies, learn PILOT on a disjoint learning bucket, reload the learned checkpoint, and deploy without router updates on a held-out bucket. Every artifact SHALL identify synthetic inputs and retain input/code provenance.

#### Scenario: Complete smoke
- **WHEN** the documented default command executes in a clean checkout
- **THEN** preference and learned checkpoints, learning/deployment traces and a summary exist, with frozen deployment state and no network calls

### Requirement: Fail closed and preserve outputs
Existing output directories SHALL be refused. Unsupported/zero accuracy SHALL prevent PILOT initialization rather than creating fallback priors or lambdas. Actual overspend SHALL remain visible independently from estimated feasibility.

#### Scenario: Repeat execution
- **WHEN** a fresh output path is used with the same configuration
- **THEN** parameters, action traces and metrics match; overwriting an existing path fails
