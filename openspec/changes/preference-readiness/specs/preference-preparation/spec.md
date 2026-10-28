# Spec Delta

## Purpose

Prepare locally available preference comparisons with explicit eligibility, reproducible held-out queries, and evidence-backed readiness for later prior training.

## ADDED Requirements

### Requirement: Exact identities and explicit exclusions
Preparation SHALL admit only distinct arms with literal exact-ID mappings, one decisive winner, and one nonempty string turn in the prompt and each response. It SHALL report exclusion reasons and per-arm/per-pair counts. Invalid schemas, duplicate battle identities, and ambiguous mappings MUST fail closed.

#### Scenario: Unverified alias and incomplete content
- **WHEN** rows contain proposed aliases, ties, multi-turn conversations, or nested null responses
- **THEN** those rows are excluded from strict preparation with counted reasons and candidate mappings remain unapproved

### Requirement: Prompt groups stay together
Preparation SHALL assign normalized prompt groups to train or holdout deterministically for a seed, independently of incoming row order. Both partitions MUST contain groups; fewer than two groups or an invalid holdout fraction MUST fail. Battle IDs and prompt groups SHALL be disjoint across partitions.

#### Scenario: Duplicate prompts and reordered rows
- **WHEN** rows repeat a prompt with whitespace differences or arrive in another order
- **THEN** duplicates remain in one partition and assignment and serialized fingerprints remain stable

### Requirement: Read-only source provenance and readiness
The audit SHALL verify pinned preference input and prepared query hashes, record code/config/source fingerprints, and emit aggregate evidence without conversation text. All-arm prior readiness MUST remain false when any arm lacks train/holdout coverage or required embedding/prior provenance is absent.

#### Scenario: Two exact arms in an eleven-arm pool
- **WHEN** eligible preferences cover only two of eleven requested arms
- **THEN** the audit identifies the nine missing arms and does not claim full prior readiness

### Requirement: Scoped embedding availability evidence
The audit SHALL inventory candidate embedding/cache paths only under explicitly supplied roots without following directory links. Candidate files alone MUST NOT establish a validated embedding model, dimension, provenance, or usable triplets.

#### Scenario: No validated cache
- **WHEN** no reviewed embedding manifest and vectors are available
- **THEN** model identity and dimension are reported unknown, readiness remains blocked, and no embeddings are downloaded or generated
