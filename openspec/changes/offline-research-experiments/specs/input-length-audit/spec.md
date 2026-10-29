# Spec Delta

## Purpose

Measure complete prompt token lengths before freezing local embedding settings, with traceable input identity and bounded memory.

## ADDED Requirements

### Requirement: Reproducible untruncated counts
The audit SHALL count each verbatim input once with a pinned tokenizer, including special tokens and without truncation or padding. It SHALL preserve input order and record both normalized prompt hashes and verbatim identity.

#### Scenario: Text at the cap
- **WHEN** a prompt has exactly the candidate cap including special tokens
- **THEN** it is counted as fitting and a prompt one token longer is counted as truncated

#### Scenario: Corrupted prompt identity
- **WHEN** supplied text differs from its prompt hash or query IDs are duplicated
- **THEN** the audit fails before publishing a completed artifact

### Requirement: Bounded and separated reporting
The audit SHALL tokenize bounded batches, persist per-input lengths separately from aggregate reports, and report counts, quantiles, maxima and truncation at each candidate cap by phase. Candidate selection SHALL use only designated learning, preference-training or tuning inputs, never deployment or preference-holdout observations.

#### Scenario: Deployment outlier
- **WHEN** a deployment query is longer than every training query
- **THEN** deployment diagnostics record it without changing the selected cap

### Requirement: Immutable audit lineage
A completed audit SHALL identify source hashes, tokenizer files and revision, batching configuration and output checksums. Existing output artifacts SHALL NOT be silently overwritten. Public aggregate reports SHALL omit raw prompts and responses.

#### Scenario: Repeat output path
- **WHEN** an audit targets an existing output directory
- **THEN** it fails without changing the existing artifact
