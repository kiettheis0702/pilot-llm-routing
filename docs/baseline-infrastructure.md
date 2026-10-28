# B01–B03 offline learning infrastructure

Source of truth: [OpenSpec change](../openspec/changes/baseline-infrastructure/tasks.md).

`OfflineBandit(queries, outcomes, arm_catalog=None)` now rejects corrupt numeric data,
duplicate/empty IDs and incomplete fixed query/arm grids at construction. An omitted
catalog is inferred; explicit one-arm and binary pools are supported. Existing
`observe/step/run_bandit_round` smoke calls retain their valid behavior.

New `OnlinePolicy` methods are `reset(seed)`, `select(context)`,
`observe(arm_id, context, reward)` and `predict_rewards(context)`. Policies own their
arm catalog. A `ContextTable(vectors, label=...)` validates finite, consistent 1D
vectors and copies inputs and outputs. No automatic normalization or embeddings are
claimed. The learning runner passes only vectors and selected rewards; actual costs
and oracle metrics stay in evaluation. Legacy smoke `Feedback` still includes cost
for compatibility and is not the new policy update interface.

These interfaces prevent accidental leakage in the supplied runner; Python objects
are not a security sandbox for untrusted policy code.

## Frozen splits

`load_split_manifest(path, queries, expected_sha256=..., seed=42)` checks the
supplied pin before parsing. It verifies query hashes and metadata against the
query table and compares the manifest with the existing grouped split algorithm
in memory. Defaults remain 1,000 tuning and a 10:1 target; rounding gives 32,270
learning and 3,227 deployment rows for the stored RouterBench dataset.

`splits.select("learning")` uses the full frozen order.
`splits.select("learning", limit=10, subset_seed=7)` chooses a deterministic subset
and restores that same order. The returned immutable `SplitSelection` records the
source checksum, split seed/settings, selected IDs, full size and subset settings.
It does not regroup duplicate prompts within a subset; all selected rows remain
in the original bucket. A caller can construct this DTO directly for toy fixtures,
but such a DTO is not evidence that a real manifest passed verification.

## Evaluation and logs

`run_learning(policy, queries, outcomes, contexts, selection, seed=42)` resets the
policy once, then calls select and observe once per query in the frozen order.
Tuning/deployment selections and missing contexts fail before reset. Selection
and update receive separate context copies. Invalid actions cannot trigger an
update. The evaluator computes the oracle only after selected feedback exists.

`RunResult.trace` contains per-round and cumulative selected reward/actual cost,
oracle reward and `empirical_oracle_regret = max_arm(reward) - selected_reward`.
`RunResult.summary` records totals, mean reward, all-arm action counts, seed,
policy qualified name, selected IDs, split/subset provenance, context label and
fingerprint, and selected query/outcome fingerprints. This is empirical regret
on stored outcomes, not expected-reward pseudo-regret. Actual cost retains the
historical RouterBench meaning. Estimated cost is neither filled nor used.

`write_run(result, new_directory)` writes `trace.csv` and `summary.json`; it rejects
nonfinite metrics and an existing directory. A failed filesystem write can leave
a partial directory; inspect it and use a new output directory on retry.
Custom policy configuration/checkpoints and a general experiment CLI belong to
B07 and later work; the current manifest is not a complete training checkpoint.

Run the small local-data verification (from the repository root):

```powershell
uv run --locked --offline --no-sync python -B -m reports.audits.baseline_checks_2026_10_28 --data-dir 'D:\DOWNLOAD GG\PILOT_DATA' --output-dir 'reports/runs/b01-b03-2026-10-28'
```

The output directory must not already exist. The probe verifies the full frozen
manifest, replays ten learning rows twice, and checks source/prepared hashes before
and after. Its fixed-arm fixture and synthetic two-dimensional contexts exist
only to validate plumbing; the numbers are not paper reproduction results.

Run unit/regression checks:

```powershell
uv run --locked --offline --no-sync python -B -m pytest -q -p no:cacheprovider
```
