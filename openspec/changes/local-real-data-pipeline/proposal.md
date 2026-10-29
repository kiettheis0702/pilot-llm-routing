# Proposal

## Why

The offline numerical core passes 310 tests, but real execution still uses synthetic contexts. Evidence-backed aliases, reproducible local embeddings, scalable storage and explicit research gates are needed before actual preference training and routing evaluation.

## What Changes

- Preserve the existing exact-ID preparation default; add opt-in evidence-backed alias and identity-match policies with explicit exclusions and per-pool coverage.
- Pin BGE-M3 for local dense embeddings, with no paid provider path. Audit lengths, batch serially, fingerprint the complete recipe, and distinguish this experiment from the paper's OpenAI/Instructor embeddings.
- Add checked float32 NPY/memory-mapped caches and bounded triplet sampling without changing mathematical eligibility.
- Add real-data preparation, embedding, training and learning-comparison commands with source/split hashes, immutable outputs and resumable preparation.
- Record unresolved dataset, checkpoint, token-price and deployment-baseline questions. Full eleven-arm and paper-figure claims must fail closed until their prerequisites exist.
- Run serial tests and a real local embedding smoke; advance to actual training/evaluation only when its data/resource gates are satisfied.

## Capabilities

### New Capabilities

- `local-research-pipeline`: Evidence-aware model pools, local embedding artifacts and staged real-data execution with explicit fidelity and readiness reporting.

### Modified Capabilities

None in the main spec inventory (currently empty). The completed preference-readiness and pretraining-core change contracts remain the compatibility baseline: exact-ID defaults and small JSON artifact APIs are retained.

## Impact

`configs/model_mapping.json`, new embedding and research configuration; `scripts/preference_data.py`, `scripts/pretraining.py`, new local artifact/embedding/runner modules, tests and research documentation. Optional pinned PyTorch/Transformers dependencies are isolated from the existing CPU environment. Data and generated vectors/checkpoints remain ignored. No model API calls, automatic credit consumption, source-data rewrites, changes to saved splits, unrelated processes, commits or pushes are included.
