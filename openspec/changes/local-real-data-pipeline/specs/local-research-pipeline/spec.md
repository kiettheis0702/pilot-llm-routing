# Spec Delta

## Purpose

Run auditable local experiments on the pinned preference and routing datasets, while keeping source identity uncertainty separate from numerical execution success.

## ADDED Requirements

### Requirement: Evidence-aware model pools
Preparation SHALL retain exact-ID behavior by default and offer explicit verified-alias and identity-match policies. Nonliteral mappings MUST include pinned source evidence. Unresolved mappings MUST remain excluded. Reports SHALL identify the policy, admitted arms, exclusions and fidelity limitations.

#### Scenario: Unverified checkpoint in a requested pool
- **WHEN** an experiment requires all eleven arms but some mappings remain unresolved
- **THEN** readiness fails with the missing identities rather than assigning invented priors

### Requirement: Source and split integrity
Real preparation SHALL verify pinned source and processed-file checksums before parsing. Existing routing splits and source text MUST remain unchanged. Preference prompt groups SHALL remain disjoint between training and holdout. Reports MUST distinguish observed task labels and provisional source choices from paper facts.

#### Scenario: Modified input file
- **WHEN** a source or split hash differs from its pinned value
- **THEN** preparation fails before producing accepted training inputs

### Requirement: Reproducible local embeddings
Embedding generation SHALL use an immutable model revision and a recorded recipe covering prompt construction, truncation, pooling, dimension, normalization and numeric precision. It SHALL run locally with bounded serial batches, report truncation counts, and never fall back to a paid API.

#### Scenario: Recipe changes on resume
- **WHEN** existing progress was created using different input hashes or recipe
- **THEN** resume is rejected without overwriting completed output

### Requirement: Validated scalable storage
Caches SHALL support finite float32 vectors larger than 16 MiB with aligned unique query IDs and prompt hashes, checksums and read-only memory-mapped loading. Partial caches MUST NOT be accepted as complete. Malformed dimensions, nonfinite values, path escapes and corrupted artifacts MUST fail validation.

#### Scenario: Cache corruption or unfinished generation
- **WHEN** a vector file is changed or completion metadata is absent
- **THEN** downstream training refuses the artifact

### Requirement: Bounded valid triplet sampling
Sampling SHALL preserve train-only, distinct-prompt and strictly-cheaper-negative eligibility without materializing the positive-negative Cartesian product. Sampling SHALL be seed-reproducible under input row reordering and reject pools without valid triplets.

#### Scenario: Large preference pool
- **WHEN** an anchor has many positive and negative candidates
- **THEN** one valid pair is sampled without allocating all candidate pairs

### Requirement: Real training and learning evaluation
The pipeline SHALL train only from validated preference vectors and comparisons with per-arm train/holdout support and a supplied cost-size order with provenance. Routing comparisons SHALL share the same selected stream, arm pool and projected contexts. Validation failures MUST block PILOT execution without fabricated lambda values.

#### Scenario: Unsupported held-out accuracy
- **WHEN** an arm has no held-out support or zero measured accuracy
- **THEN** the report exposes the blocker and no fallback prior is substituted

### Requirement: Research claims remain gated
Every real-data result SHALL identify embedding/model-pool substitutions and unresolved paper-equivalence questions. Budgeted research results MUST require justified token counts, historical prices, cost bounds and a declared common deployment protocol. Missing inputs SHALL be reported as blockers, not replaced with deployment outcomes.

#### Scenario: Missing historical cost inputs
- **WHEN** real embeddings exist but deployment cost provenance is incomplete
- **THEN** learning-only evaluation may run with that scope explicitly labeled and paper budget figures remain unavailable

### Requirement: Resource-aware execution
Heavy commands SHALL check available physical RAM and conflicting jobs before starting, use one worker and bounded batches, and refuse to start below 8 GiB without a run-specific explicit override. They MUST NOT terminate unrelated processes or increase parallelism to recover performance.

#### Scenario: Insufficient RAM
- **WHEN** available physical RAM is below the configured minimum
- **THEN** the command stops before loading a model and reports the measured available memory
