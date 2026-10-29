# Design

## Context

See proposal.md. Python 3.11 and the CPU numerical environment are working. The baseline is 310 passing tests. Existing preparation admits two literal IDs (374 pairs); evidence gathered in this chat identifies three direct aliases and three further model-identity matches. Three RouterBench provider aliases remain unresolved. No learned real checkpoint or semantic vector cache exists.

## Goals / Non-Goals

Goals: preserve old APIs, make real local experiments executable and reproducible, and distinguish software readiness from scientific fidelity. Establish a staged path through preparation, encoding, fitting and learning evaluation.

Non-goals: no paid requests, proxying subscription credentials, current-model generation, relabeling tasks to force 64, invented historical prices, automatically accepting the three unresolved mappings, or claiming Figure 3/4 reproduction from a development run. Full experiments remain gated on documented scientific inputs; their pending requirements will be retained in the research report.

## Decisions

### Approaches and selected default

1. OpenAI embeddings would match the main paper more closely but require a separately approved credit-consuming call. Excluded for this implementation.
2. Instructor-XL matches the paper's embedding sensitivity variant but is heavier and needs its instruction recipe resolved. Keep as a documented later option.
3. BGE-M3 local is selected as the development default: pinned revision `5617a9f61b028005a4858fdac845db406aefb181`, dense 1024 dimensions, CLS pooling, L2 normalization, no task instruction, FP16 GPU inference, float32 persisted vectors. This is an explicitly labeled embedding substitution, not paper equivalence.

Initial probe: batch size 2, max length 512, one numerical/loader worker, 128 RouterBench learning queries. The max-length cap is an engineering choice, and the exact tokenizer truncation count is reported. Full runs require checking the length distribution and selecting a fixed recipe; no silent cap change during resume. Routing and preference vectors must use the same recipe.

### Evidence policies

`exact_id` stays the default in `prepare_preferences`. Add keyword `mapping_policy` with `exact_id`, `verified_alias`, `identity_match` as increasing inclusion levels. Evidence entries record URL, revision, observed relationship and limitations. Two Claude renames and WizardLM version are `verified_alias`; Llama/CodeLlama/Yi are `identity_match`. Mistral/Mixtral/Claude-v2 remain unresolved. A policy is an experimental interpretation, not a proof that serving parameters matched. `--require-all-arms` fails if any requested arm is excluded.

### Modules and artifacts

- `scripts/model_mapping.py`: mapping validation, allowed catalogs and evidence-policy resolution.
- `scripts/embedding_store.py`: `DiskEmbeddingCache` compatible with the existing cache's read interface; `manifest.json`, `vectors.npy`, recipe and prompt/ID metadata. Create-only artifacts; manifest published last. Read with `allow_pickle=False`, mmap, SHA-256 and chunked finite checks. Legacy JSON remains readable.
- `scripts/local_embeddings.py`: pinned local model download/encode entry point, single process, tokenizer length audit, batch checkpoint progress, matching-recipe resume and explicit resource preflight. No inference provider dependencies or API credentials.
- `scripts/research_pipeline.py`: staged prepare/fit/learn CLI. Private prompt tables go only into ignored `outputs/`; aggregate reports and evidence go into tracked docs/reports. Source hashes checked before reads. Source raw files and saved split remain untouched.
- `scripts/resources.py`: Windows physical memory and active heavy-job inventory, with a clear fail-closed preflight.
- `scripts/pretraining.py`: accept the disk cache through a declared compatible cache type; sample one uniformly selected valid pair by candidate counts instead of constructing Cartesian products. Project in chunks and read training vectors per triplet rather than copying all triplets at once.
- `configs/local_embedding.json`, `configs/research_protocol.json`: explicit defaults, deviations, unresolved research prerequisites, optional token-cost size input. No synthetic cost values may be used in real fits.
- Optional embedding environment lives in ignored `.venv-embedding/`; pinned requirements retain the CPU lock unchanged. CUDA wheel must support RTX 5070; validate a small actual operation before embedding.

### Training and evaluation gates

Preparation can enumerate the 2/5/8-arm policies without fitting. The identity-match development pool is eight arms and must always be labeled reduced-pool/identity-level evidence. Support and prompt-overlap diagnostics accompany the prepared rows. Full 11-arm remains false.

`fit` requires a complete validated preference cache and a JSON cost-size input covering exactly the admitted arms with a nonempty provenance. No dataset realized-cost average will masquerade as per-token prices. Epoch/dimension settings are explicit and bounded for the first run; checkpoints use the existing validated JSON (small projection/prior, not bulk vectors).

`learn` consumes a trained, validated checkpoint, matching routing cache and saved learning selection. PILOT and existing baselines use identical projected vectors; outcomes and realized cost are used only through the current feedback/evaluation interface. Budget deployment and paper-like figures remain blocked until the report's source/cost/baseline gates are closed.

### Compatibility and rollback

No existing defaults, data files, split manifests or legacy reports are overwritten. Each run creates a unique directory; resume applies only to unfinished encoding with exact matching identity. New configuration and scripts can be removed without migrating old artifacts. No commit/push is part of this request.

## Risks / Trade-offs

- Misidentified checkpoint -> separate evidence strength, three unresolved arms excluded, full-pool guard.
- Limited RAM / GPU memory -> check before each heavy stage, serial tests, batch 2, release encoder before CPU training, no OOM fallback to paid inference.
- 512-token truncation -> exact tokenizer audit and recorded cap; probe does not establish final recipe suitability.
- Corrupted/resumed cache -> hashes, complete-manifest gate, deterministic batch boundaries, no object arrays or arbitrary paths.
- Slow NumPy fitting -> bounded probe first; sampler avoids Cartesian expansion; no full-study timing claims.
- Historical data ambiguity -> report 0-shot and 86 task labels as observed; preserve unresolved authors' exact release/task grouping.

## Validation

Test identity-policy isolation; stale/tampered cache, partial resume, shape/finite failures; stable valid triplets; training recipe and support gates; resource denial; and real CLI preparation/encoding/readback. Run the full CPU suite and existing toy smoke sequentially after targeted tests. Real model smoke is distinct from unit fixtures and records revision, device, length audit, memory, elapsed time and hashes. Retain every blocked research prerequisite in the final handoff.
