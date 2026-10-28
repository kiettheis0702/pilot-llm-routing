# Tasks

## 1. Strict preference preparation

- [x] 1.1 Implement exact-ID eligibility, strict content/label validation and aggregate exclusions; verify CPU fixtures for ties, aliases, missing/nested values, multi-turn and duplicate identities; document the rules.
- [x] 1.2 Implement deterministic prompt-group train/holdout splitting and stable table fingerprints; verify row-order invariance, group disjointness and invalid split boundaries; document default seed/fraction.

## 2. Readiness evidence

- [x] 2.1 Implement source/prepared fingerprint verification, per-split arm/pair coverage, aggregate CLI output and scoped embedding candidate inventory; verify fail-closed inputs, honest missing-model/dimension reporting and report privacy with CPU fixtures.
- [x] 2.2 Run the read-only audit against pinned local Arena and prepared RouterBench queries; save aggregate JSON and docs with measured readiness, exclusions, hashes, command results and triplet/embedding blockers.

## 3. Integrated verification

- [x] 3.1 Run the complete current pytest suite, strict OpenSpec validation and actual diff review; record exact outcomes and preserve raw/config/legacy audit files.

## Workflow follow-up

- Parent agent reviews and performs separately authorized Git operations. No training or embedding API work belongs to this change.
