# Proposal

## Why

The real preference dataset is blocked on model coverage, embedding provenance, and cost-size ordering, but the numerical training and artifact contracts can be verified on small synthetic fixtures. A tested reusable core makes those later research inputs explicit instead of fabricating them.

## What Changes

- Add a NumPy-only two-phase mathematical core: cosine triplet projection followed by frozen-projection pairwise model-vector learning.
- Require explicit supplied embeddings, unique query/prompt identities, disjoint train/holdout groups, complete arm coverage, and provenance-backed cost-size values for hard negatives.
- Validate analytic gradients with finite differences and tiny deterministic loss-decrease fixtures.
- Add safe JSON cache/checkpoint serialization with dimensions, IDs, model/version, fingerprints, metrics, and exact reload behavior.
- Diagnose inverse-accuracy lambda only when each arm has positive measured supported accuracy; absent/zero support blocks lambda export.

## Capabilities

### New Capabilities

- `preference-training-core`: numerical preference fitting, explicit provenance/split contracts, and validated cache/checkpoint interchange.

### Modified Capabilities

None. The existing preference audit and full-pool blocked status remain unchanged.

## Impact

New `scripts/pretraining.py`, `tests/test_pretraining.py`, `docs/pretraining-core.md`, optional aggregate toy report, and this OpenSpec change. NumPy is already installed. No Torch, downloads, embedding generation, external API calls, real-data training, shared-file edits, or Git mutations are included. Parent review and separately authorized Git operations follow completion.
