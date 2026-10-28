# Spec Delta

## Purpose

Load the frozen routing split with verifiable provenance and deterministic learning subsets without changing prepared data.

## ADDED Requirements

### Requirement: Verified immutable manifest
The loader SHALL verify a caller-supplied SHA-256, query coverage, unique IDs, query text hashes, task metadata, allowed split names, disjoint prompt groups and deterministic stream positions. Assignments MUST match the declared seed and split settings, with 1,000 tuning rows and the remaining 10:1 learning/deployment target by default.

#### Scenario: Current RouterBench artifact
- **WHEN** the pinned seed-42 36,497-query manifest is loaded
- **THEN** it yields 1,000 tuning, 32,270 learning and 3,227 deployment rows without writing data

#### Scenario: Tampering or wrong seed
- **WHEN** checksum, assignments, metadata, coverage or declared seed disagree
- **THEN** loading fails before a learning run

### Requirement: Explicit subset provenance
Selections SHALL retain the frozen stream order and record source checksum, split seed, split name, full split size, selected IDs and subset settings. A requested subset MUST be deterministic for its seed and size, clearly distinguished from the full split, and must not change any split membership.

#### Scenario: Repeat subset
- **WHEN** the same manifest, subset seed and size are selected twice
- **THEN** selected IDs and their order are identical and the original manifest bytes are unchanged

#### Scenario: Invalid subset
- **WHEN** an unknown split, zero/negative/noninteger size or size above the split population is requested
- **THEN** selection fails instead of silently truncating or choosing another split
