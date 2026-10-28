## Purpose

Run bounded toy and RouterBench baseline learning comparisons with explicit provenance and regenerable metrics artifacts.

## ADDED Requirements

### Requirement: Bounded reproducible comparison
The CLI SHALL run all configured baselines on identical selected query order and context data, record seed/hyperparameters/code fingerprints and deviations, and preserve source files. RouterBench SHALL require a data root and verify prepared pins. Runs SHALL default to toy data and refuse overwriting an output directory.

#### Scenario: Repeat toy run
- **WHEN** two runs use identical configuration
- **THEN** their action traces and metrics match, with JSON summaries, CSV traces and a regret plot generated from those traces

### Requirement: Honest feature labeling
Toy features and deterministic hashed query features SHALL be explicitly synthetic and unsuitable for paper-result claims. No embedding provider calls or downloads SHALL occur. Real-data smoke runs SHALL be capped at 1,000 rows.

#### Scenario: Unbounded request
- **WHEN** a RouterBench request exceeds the smoke cap
- **THEN** the CLI rejects it before running a model
