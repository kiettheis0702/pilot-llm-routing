# Design

## Context

See proposal.md. Verified ACL paper section 2.2.2 equations 1/2 and Algorithm 1 at https://aclanthology.org/2025.findings-emnlp.1301.pdf. Existing LinearUCB owns validated chosen-arm updates and defensive A/b access.

## Goals / Non-Goals

Implement only online mathematics and a model checkpoint. No preference training, embedding substitution claims, accuracy-to-lambda estimator, cost policy or runner-resume protocol.

## Decisions

- Subclass LinearUCB rather than duplicate its chosen-feedback validation/update. Override reset, normalized context and cosine estimates. Preserve catalog order ties.
- Caller must provide a complete arm-to-vector prior, arm-to-positive-lambda mapping and nonempty provenance. Initial A=lambda I and b=lambda prior; never invent a prior. Scaling norms before normalization avoids overflow/underflow. Reject zero query contexts; define zero arm-vector cosine as zero (local convention, paper silent).
- Model JSON v1 includes configuration, original priors/provenance, seed, rounds, A/b and SHA-256 over canonical payload. No pickle or runtime imports from artifact. Check exact schema, finite shapes, symmetry/SPD and checksum before returning a new policy. Checksum detects accidental changes, not malicious re-signing. No RNG state is needed because PILOT action selection is deterministic.
- Save with exclusive creation after full in-memory serialization; refuse overwrites. A checkpoint restores a model for manual continuation; run_learning always resets by contract and is not a resume API.

## Risks / Trade-offs

Finite precision can destroy SPD even with positive mathematical ridge: inherited update rejects it before mutation. JSON state is quadratic in dimension; only toy verification here. Caller provenance is a label, not proof of real training. Positive lambdas are explicit because the paper's inverse-accuracy prescription does not define zero accuracy handling.

## Migration Plan

Add isolated module and tests, without modifying existing APIs/data. Parent reviews and publishes; reverting these additions removes the new capability.
