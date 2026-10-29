# Local real-data pipeline Implementation Plan

**Goal:** Run real local semantic inputs through the existing numerical pipeline with explicit evidence and readiness gates.
**Architecture:** Evidence-aware preparation feeds immutable disk caches, then existing preference training and learning policies. Resource and scientific gates precede heavy execution.
**Tech Stack:** Python 3.11, NumPy/pandas/Arrow; isolated pinned PyTorch/Transformers for local BGE-M3.
**Spec:** `specs/local-research-pipeline/spec.md`; implementation choices in `design.md`.
**Execution:** Current agent, serial TDD using writing-plans/executing-plans principles. User explicitly requested analysis followed by implementation in this turn; no duplicate approval or separate plan document. No commit/push.

## Global Constraints

Exact-ID default and legacy JSON API remain compatible. No paid API; no raw/split rewrites. One heavy command at a time, one worker, 8 GiB free physical RAM before starting. Report every substitution and unresolved scientific prerequisite. Freeze the encoder recipe across preference/routing inputs.

## Review Focus

Policy elevation without evidence (task 1.1); partial/corrupt or mismatched-resume artifacts (2.1/3.1); hidden Cartesian memory growth or prompt leakage (2.2); inadequate holdout/cost provenance (4.1); silent truncation or unsupported GPU (3.1/5.1).

## 1. Model evidence and preparation

- [x] 1.1 Add `resolve_mappings(mappings, policy)` in `scripts/model_mapping.py`, extend `prepare_preferences(..., mapping_policy='exact_id')`, record pinned mapping evidence and documentation. Verify policy isolation, absent evidence/ambiguous mappings and unchanged exact-ID results with targeted tests.
- [x] 1.2 Add the prepare stage in `scripts/research_pipeline.py`: verified source/split loading, aggregate 2/5/8-arm coverage, immutable private preference/query inputs and full-pool blockers. Verify altered-source rejection, all-arm rejection, repeated execution and actual-data counts.

## 2. Scalable numeric artifacts

- [x] 2.1 Add `DiskEmbeddingCache.create/load` in `scripts/embedding_store.py`, compatible `ids`, `prompt_hashes`, `dimension`, `provenance`, `get`, `fingerprint` interface with chunked reads. Test >16 MiB readback, nonfinite/shape/hash/path errors and refusal to overwrite or read partial artifacts.
- [x] 2.2 Integrate disk caches into pretraining, replace Cartesian triplet enumeration with equivalent count-weighted sampling and chunk projection/read vectors per triplet. Test training-only/distinct-prompt eligibility, reorder determinism, large candidate pools and legacy roundtrips.

## 3. Local encoder and resource control

- [x] 3.1 Add `scripts/resources.py`, `scripts/local_embeddings.py`, pinned `configs/local_embedding.json`, optional requirements and local setup instructions. Test insufficient resources, recipe validation, interrupted generation, matching resume and tokenizer truncation diagnostics; confirm CLI help has no paid-provider path.

## 4. Actual training and comparison interfaces

- [x] 4.1 Add fit/learn stages with cache/prepared lineage validation, explicit cost-size provenance, held-out readiness and identical contexts/stream for PILOT/baselines. Verify end-to-end fixture, wrong recipe/pool/support rejection and retained full-research blockers; document exact commands.

## 5. Integrated verification and research handoff

- [x] 5.1 Install isolated pinned embedding dependencies; verify RTX 5070 compatibility; download only the pinned public encoder and run a 128-query local embedding smoke, read back and resume-check it. Record actual memory, truncation, speed and checksums. Evidence: `outputs/local-pipeline-verification-20261029/embedding-verification.json`; CUDA FP16 passed, 128 × 1,024 vectors, zero truncation, readback/resume passed after the 8 GiB preflight passed.
- [x] 5.2 Run the full CPU tests, legacy toy smoke and new staged CLI probes sequentially; inspect generated artifacts, validate OpenSpec and Git diff. Update research readiness with completed work, observed evidence, unresolved mappings/dataset questions and cost/deployment prerequisites. Do not claim full experiments when those gates remain open. Evidence: 338 passed / one OS symlink skip, eight CPU/data probes and five GPU/cache probes; both engineering verification tasks complete.

## Workflow follow-up

- Full preference/routing encoding and real learned priors require the chosen final recipe and an evidenced token-cost order. Proceed when those gates are satisfied; record actual blockers otherwise.
- Budget sweeps, multiple seeds, paper figures and poster require historical tokenizer/pricing inputs and the common baseline deployment protocol, plus resolved or explicitly labeled dataset/model deviations.
