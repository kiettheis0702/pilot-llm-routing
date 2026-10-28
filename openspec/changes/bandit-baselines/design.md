# Design

## Context

Reuse OnlinePolicy/ContextTable/run_learning/write_run. No embedding cache or confirmed prior exists. NumPy ridge solves suffice without adding ML packages.

## Goals / Non-Goals

Test real algorithm state on small deterministic contexts and validate the CLI. No hyperparameter search on held-out data, performance-budget experiment, full training or embedding substitution claim.

## Decisions

- Each policy owns an ordered catalog and dimension; validate inputs before mutation. All RNGs are local. LinearUCB keeps disjoint A,b arrays with finite-update checks and stable solves. Raw vectors are used (normalization is the caller's feature contract).
- ExploreOnly chooses maximum uncertainty rather than uniform random (already provided separately). EpochGreedy uses one uniform exploratory round per fixed window and updates on every chosen feedback. Both choices are PAPER_INFERRED because the PILOT description does not fully specify them.
- CLI accepts toy or RouterBench source, seed, row cap, dimension, alpha, ridge and window. Synthetic hash features depend only on query text, never outcome labels. Every run marks DEVIATION: synthetic contexts.
- Use the existing evaluator for all methods. Artifact provenance captures Git HEAD plus hashes of implementation files so uncommitted execution remains identifiable. Per-policy configuration and selected data fingerprints are saved. Standard Matplotlib produces PNG/SVG from saved traces, explicitly marked smoke. This adds a locked plotting dependency instead of a custom scientific plotting implementation.

## Risks / Trade-offs

Quadratic state per arm and cubic solves are fine for toy dimensions but not a full high-dimensional plan. Do not extrapolate smoke timings to full experiments. The CLI caps rows/dimension. SVG is for diagnostics; publication figures require later plot tooling and a locked protocol.
