# Design

## Context

See proposal.md. Existing code supplies frozen source checksums/splits, evidence-aware preference preparation, immutable disk caches, numerical prior training, learning policies and a standalone deployment cost policy. It has only small real embedding coverage; learned prior and budget results are still synthetic. The current Windows GPU environment is installed and verified.

## Goals / Non-Goals

**Goals:** Connect these components through immutable manifests, tuning-only choices, bounded local jobs and clearly scoped research claims. Preserve full research scope while delivering each stage independently.

**Non-Goals:** Paid generations, silently substituting unresolved model versions, altering raw data/splits, or claiming a local embedding substitution is the original paper result. Commit/push and contacting paper authors require separate user direction.

## Decisions

1. **Audit before full embedding.** Use the locally cached pinned fast tokenizer, disable tokenizer parallelism, count special tokens without truncation and process at most 32 texts per batch. Store ordered row counts privately and only aggregate statistics in tracked reports. Hash verbatim text as well as normalized prompt identity. Tokenizer-only auditing avoids loading the encoder just to measure text.
2. **Separate selection from diagnostics.** Choose the smallest cap in 512/1024/2048/4096/8192 covering at least 99% of routing learning+tuning and preference-training inputs, subject to measured GPU feasibility. Deployment and preference holdout lengths are diagnostics only. Record any resource-driven deviation before outcome evaluation. Keep the 512 smoke recipe and its caches intact.
3. **Freeze full inputs in separate manifests.** Export all routing phases in their frozen stream order, plus preference inputs from the explicitly selected evidence policy. Persist mapping/source hashes and cross-corpus prompt overlap. Reuse an embedding only when its full input/recipe identity matches; do not assume normalized text equivalence implies identical vectors.
4. **Historical cost evidence, not current quotes.** Inspect the pinned RouterBench converter and original paper for prices and token methods. Store observed outcome costs separately from estimated deployment costs. Tuning response lengths alone inform estimates; unresolved tokenizer equivalence is an explicit limitation. Never call paid endpoints to obtain missing metadata.
5. **Common evaluation adapter.** Add a documented score/ranking interface for all learning policies and apply the existing deployment cost policy identically, including ties, exhausted budget, realized spend and unserved queries. Verify with analytic cases and state hashes. Existing PILOT-only deployment remains compatible.
6. **Phased experiments.** First validate a small real prior-learning-deployment path, then freeze the complete protocol and run complete learning/deployment datasets and MMLU subset, multiple seeds, budget sweeps and ablations sequentially. Keep verified 5/8-arm development runs distinct from the full 11-arm reproduction target. Missing historical evidence is investigated and documented, never silently accepted as equivalence.
7. **Reviewable outputs.** New commands refuse existing output paths; completion manifests are published last. Record command/config/input hashes and versions. Aggregate figures, report and poster link the exact run manifests. Pending work stays unchecked.

## Risks / Trade-offs

- Long prompts increase GPU time and activation memory -> length audit, batch 1/2 feasibility probe, single worker and RAM gate.
- Historical provider aliases may have no public proof -> continue evidence collection and independent stages; preserve unresolved status and distinguish development results.
- Paper omits reproducibility settings -> declare local choices before evaluating outcomes and label sensitivity analyses.
- Many sequential experiments take time -> resumable stage artifacts and explicit run inventory; no reduced-data result is labeled full.
- Arena holdout arm coverage can be sparse -> report per-arm counts/accuracy and refuse invalid inverse-accuracy initialization rather than injecting a fallback.

## Migration Plan

Add scripts/configs and tests without rewriting completed preparation/cache artifacts. Introduce the audited recipe under a new path. Existing smoke commands remain valid. Interrupted stages can be retried into new paths or resumed only under exact identity; rollback consists of retaining old recipe/outputs and not selecting incomplete artifacts.
