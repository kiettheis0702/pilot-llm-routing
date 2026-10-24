# Reproduction log

## 2026-10-24 — Repository setup

Connected the local checkout to `kiettheis0702/pilot-llm-routing`, initialized `main`, and added project context and Git hygiene. No dataset or result was assumed complete.

## 2026-10-24 — First Data Gate audit

Evidence: [audit report](data-gate.md), [machine-readable measurements](../reports/data-gate.json), [source pins](../configs/data_sources.json).

- Downloaded and hash-verified both processed RouterBench variants and the public Arena 55K candidate.
- Selected 0-shot provisionally because it has unique IDs, complete reward/cost data, and 36,497 rows. The paper's exact file selection remains unknown.
- Found 86 raw task labels versus the paper's 64; retained the labels without inventing a mapping.
- Rejected 5-shot for strict preparation because of duplicate IDs and missing rewards. No silent imputation or merge.
- The older Arena 33K file was gated (HTTP 401); recorded metadata only. Arena 55K is a separately identified candidate, not silently substituted as the confirmed original.
- Counted exact-ID and proposed-alias preference coverage separately. No preference priors were trained.
- Preserved fractional scores; grouped duplicate prompts across a seed-42 split; recorded deterministic output hashes.
- Exported query and outcome tables separately. Kept estimated cost unknown and response-dependent cost out of preselection observations.
- Passed a one-step selected-arm feedback/update smoke check and the focused data-contract tests. These are infrastructure checks, not research results.

Status: **conditional Data Gate**. Offline baseline development can proceed under the provisional protocol; exact dataset provenance, alias versions, preference preprocessing, and deployment cost estimation still need resolution.
