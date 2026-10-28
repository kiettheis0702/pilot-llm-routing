# Spec Delta

## Purpose

Provide a verifiable local preference-training core whose mathematical behavior and artifact contracts can be tested without generating embeddings or training on real benchmark data.

## ADDED Requirements

### Requirement: Explicit validated inputs
The core SHALL require finite supplied embeddings with model/version/source provenance, unique query IDs with associated prompt hashes, an exact arm catalog, valid decisive comparisons, and disjoint train/holdout prompt groups. Fitting SHALL use train comparisons only and require every arm to have train and holdout support.

#### Scenario: Invalid provenance or leaking split
- **WHEN** embeddings lack provenance or a prompt group appears in both partitions
- **THEN** the core rejects the inputs before fitting

### Requirement: Preference-conditioned triplets
Triplets SHALL use a positive with the anchor's preferred winner and a negative where that winner loses to a strictly lower supplied token-cost size. All members MUST have different prompt groups. Missing or incomplete size ordering/provenance MUST block sampling.

#### Scenario: Unsupported negative
- **WHEN** an anchor winner loses only to an equal or larger size model
- **THEN** those rows cannot provide its hard negatives and no valid-triplet claim is fabricated

### Requirement: Verified two-phase mathematics
The core SHALL fit a linear query projection with cosine triplet loss, then fit per-arm cosine softmax/BCE vectors while keeping the projection unchanged. Analytic gradients MUST agree with finite differences away from nondifferentiable points; seeded tiny fixture fits MUST be reproducible.

#### Scenario: Frozen projection
- **WHEN** phase two fits model vectors
- **THEN** the supplied projection parameters remain bitwise unchanged and held-out values cannot affect learned parameters

### Requirement: Supported accuracy diagnostics
Per-arm validation accuracy SHALL use only held-out comparisons involving that arm. Lambda SHALL equal inverse accuracy only when support exists and accuracy is positive. Zero or absent accuracy MUST produce an explicit blocker without invented priors or floors.

#### Scenario: No correct held-out comparisons
- **WHEN** a supported arm has zero measured accuracy
- **THEN** its lambda is unavailable and the diagnostic identifies the blocker

### Requirement: Safe reproducible artifacts
Embedding caches and checkpoints SHALL use validated JSON with checksums, schema versions, dimensions, model/version provenance, explicit IDs and fingerprints. Checkpoints SHALL retain projection, arm vectors, metrics and protocol settings. Reloading MUST preserve numerical output; malformed, nonfinite, inconsistent or corrupted artifacts MUST be rejected.

#### Scenario: Checkpoint readback and tampering
- **WHEN** an artifact is saved then reloaded without changes
- **THEN** vectors and projected output remain equal
- **WHEN** its payload is modified or dimensions contradict its catalog
- **THEN** loading fails
