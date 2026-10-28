# Design

## Context

See proposal.md. The current aggregate audit allows multi-turn rows in descriptive counts and does not validate response content for training. Existing exact mappings cover two GPT snapshots. The Data Gate already provides pinned sources, prepared query hashes, and a whitespace-normalized SHA-256 grouping convention.

## Goals / Non-Goals

**Goals:** implement separately testable local eligibility, grouped split, aggregate report, and scoped availability discovery; distinguish executable preparation from scientific readiness.

**Non-Goals:** train a prior, introduce a local substitute encoder, approve aliases, rewrite inputs, or claim the source/split matches the paper.

## Decisions

- Return an in-memory eligible pair table containing battle ID, original single prompt, prompt hash, mapped arms, and winner/loser. Do not export conversation-bearing tables in this task. Report canonical JSON hashes of rows and assignments instead.
- Require literal equality for `exact_id`; aliases cannot become exact merely by changing a status string. Mapping names and target arms must be unique. Preserve existing candidate scenarios in a separate descriptive profile, never as eligibility.
- Parse prompt and both responses as JSON lists of strings; retain explicit failure reasons for missing/empty/nonstring/malformed content. Enforce exactly one turn in all three columns, excluding whole-conversation labels rather than silently taking a first turn. Validate one-hot labels as numeric 0/1 values.
- Use the existing `query_hash` whitespace grouping. Order groups by SHA-256 of seed and group hash, choose `ceil(group_count * holdout_fraction)` groups (bounded to leave at least one train group), and order pairs by battle ID. Default fraction 0.2 and seed 42 are local decisions. This targets group fraction, not exact row fraction; do not stratify on labels or split a repeated query to meet an exact row target.
- Report per-split per-arm wins/losses and unordered pair counts, no-group overlap, prepared RouterBench prompt/segment matches, exclusion reason incidence and mutually exclusive precedence counts.
- Reuse pinned source verification and compare prepared queries against the existing Data Gate fingerprint before reading. Record selected inputs, config hashes, implementation hash and runtime. The CLI writes only its aggregate report.
- Inventory model/cache candidates by filename/extension in workspace and selected data root, skipping tool environments, VCS, and directory links. Inventory is not cache validation: model/version/dimension stay unknown until explicit provenance is reviewed.
- Paper Section 2.2.1 hard negatives depend on a token-cost-based size order. No such pinned order exists here; usable paper triplet count is unknown, not zero or a fabricated count. Model alias equivalence, exact source release, and model ordering are outside this preparation change.

## Risks / Trade-offs

- Conservative response/turn filtering loses examples → report losses and preserve raw inputs.
- Exact-text overlap checks miss paraphrases → explicitly limit the contamination claim.
- Small two-arm holdout is not eleven-arm validation → always report full-pool gaps; no fabricated priors.
- Filename inventory can miss unconventional caches → label scope/heuristic and keep readiness blocked, even when candidates exist.

## Migration Plan

Add standalone module/tests and a new aggregate report. Existing audit, data, configuration and baseline interfaces remain usable. Rollback removes only these added files; no source migration occurs.
