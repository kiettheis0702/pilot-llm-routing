# Design

## Context

See proposal.md and docs/data-gate-audit.md. Existing validation is only upstream of OfflineBandit, and the saved split already matches make_splits. Python 3.11+, NumPy, pandas and pytest are available. No embeddings or research policies exist yet.

## Goals / Non-Goals

Keep the existing smoke API compatible for valid inputs and build a separate numeric learning boundary. Reuse split generation solely for in-memory verification. Do not change normalization, regenerate the stored split, introduce a package restructure, or implement B04+.

## Decisions

1. Strengthen OfflineBandit with an optional explicit catalog (otherwise infer sorted IDs), full grid/numeric validation and defensive frame copies. Reject variable query pools. A single-arm declared pool remains valid. This extends existing code rather than maintaining two simulators.
2. Add scripts/bandit_contracts.py with OnlinePolicy protocol and ContextTable. ContextTable copies finite 1D vectors with a consistent dimension and returns copies. Policies receive no query metadata, costs or evaluator reference. select and observe receive separate copies so selection cannot corrupt the update context. Do not normalize synthetic vectors or imply learned projections.
3. Add scripts/splits.py. load_split_manifest requires the expected checksum, parses IDs as strings, verifies metadata and compares assignments/order with make_splits using the declared seed. FrozenSplits.select returns an immutable SplitSelection with ordered IDs and provenance. A subset ranks IDs by seed hash, then restores the frozen stream order. Exact subset size wins over prompt-group preservation within one bucket; prompt groups never cross bucket boundaries.
4. Add scripts/evaluation.py. run_learning constructs its environment internally from the selected query IDs and validates all selected contexts before policy.reset. The evaluator accesses rewards only after a selected feedback exists, and never goes to the policy. Report empirical_oracle_regret, not expected pseudo-regret. Log cost but do not pass it to observe or constrain learning.
5. RunResult contains trace rows and a JSON-compatible manifest. write_run creates a new directory exclusively and writes trace.csv and summary.json. Include selected data/context fingerprints, split checksum/settings and policy-qualified name; research policy hyperparameters remain a B07 configuration concern. Refuse overwrites. No CLI baseline is introduced.

## Risks / Trade-offs

- Python isolation is an API boundary, not a sandbox against hostile policy code; verify supplied call paths with spies and perturbation tests.
- Strict construction rejects previously accepted malformed data; valid old smoke calls remain supported.
- Full manifest verification recomputes hashes/splits in memory; acceptable at 36,497 queries, measured with the local-data check.
- Raw task labels, mapping uncertainty and context semantics remain research ambiguities; record synthetic-context status explicitly and do not train preferences.
- FrozenSplit objects convey local trusted provenance, not cryptographic attestation against a caller constructing arbitrary objects.

## Migration Plan

Add modules and tests on the current codex/data-gate branch; retain all data bytes and existing smoke entry points. No dependency or schema migration. Removal of these new APIs and reversal of constructor validation restores the old code if needed. No commit, push or deployment is authorized in this change.
