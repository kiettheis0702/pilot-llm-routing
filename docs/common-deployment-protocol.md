# Common deployment protocol (local interpretation)

This supplies task 4.1 of `offline-research-experiments`. The paper requires one
cost policy for all baselines and distinguishes linear LinUCB rewards from PILOT
cosine rewards, but does not specify every baseline adaptation. Source:
[Sections 3.1.3–3.2](https://arxiv.org/html/2508.21141v2#S3.SS1.SSS3).
The following are declared implementation choices made before budget experiments;
they are not attributed to the authors or described as user-selected settings.

1. Freeze the learned model into a separate validated snapshot. Deployment never
   calls its learning/reset/UCB selection methods or changes its RNG. Snapshot
   checksums, configuration, learned matrices and routing seed are recorded.
2. PILOT keeps cosine predictions. LinUCB, ExploreOnly and EpochGreedy use their
   frozen linear ridge means, with no uncertainty bonus and no clipping or
   sigmoid. Extending the cost policy with an explicit `linear` score domain
   preserves their ordering. The default `cosine` domain remains range-checked.
   Linear values outside [0,1] are diagnostics, not calibrated probabilities.
3. Random keeps its constant 1/K utility and uniformly breaks ties among eligible
   arms. A permutation derived from the deployment seed and query ID fixes its
   random ranking across budget sweeps, independently of learned-model RNG and
   outcome labels. This is uniform among cost-eligible arms, not among all arms
   regardless of budget. Constant utilities are not fitted reward predictions.
4. All-to-one permits only its declared fixed arm, including in fallback. It
   terminates if that arm cannot pass the common rule; it never becomes a cheaper
   multi-model router. Full catalog, outcomes, estimates and oracle diagnostics
   still match other methods. Its preferred-arm indicator is not a fitted reward.
5. Every method shares query order, catalog, costs, budget, bins, ratio bounds,
   threshold-first selection, average-allowance fallback, estimated affordability
   and terminal exhaustion. Deterministic ties retain catalog order. A supplied
   tie permutation changes ties only, never a strict score preference.
6. Actual outcomes are revealed only after selection. Recorded costs do not alter
   future estimates or reservations. Report actual overspend, service coverage,
   served-only reward/regret and total reward divided by requested query count
   (zero service utility for unserved queries, not imputed observed labels).
7. Preflight validates all contexts, estimates and frozen-model predictions before
   the first reservation. Invalid later inputs cannot leave a partially executed
   budget run. Checkpoint loading refuses malformed/nonfinite/non-SPD states.
8. Existing PILOT-only calls stay compatible. Bare baseline objects are still
   rejected by that entry point; `run_policy_deployment` explicitly freezes and
   adapts supported policies. No deployment outcome selects parameters or bounds.

Validation sequence: analytic cost/domain/allowed-arm/tie tests; snapshot and
serialization tests; common-runner outcome-mutation, coverage and frozen-state
tests; full regression; real learned eight-arm integration with persisted traces.
Full tuning bounds, seeds and budget grids belong to task 2.3/3.2 and are not
considered frozen merely because this adapter exists. Positive ratio bounds remain
assumptions; negative predictions, zero costs or ratios outside those bounds do
not acquire the paper's theoretical guarantee.
