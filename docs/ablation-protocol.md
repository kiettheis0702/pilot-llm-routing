# Controlled ablations and reporting protocol

Declared on 2026-10-29 before the full tuning/deployment experiments. This extends
the existing OpenSpec research plan; it does not claim new runtime evidence.
The machine-readable inventory is [ablation_plan.json](../configs/ablation_plan.json).
The primary plan remains [main_experiment_plan.json](../configs/main_experiment_plan.json).

## Comparisons

All five variants compare with the matched **full-learning PILOT** result, on both
RouterBench and the existing MMLU subset, for seeds 7/42/2026 and all eight declared
budget fractions. Freeze the main per-domain tuning alpha, cost estimator and
ratio bounds before these runs. Do not retune a variant or inspect deployment
labels to choose parameters. This measures controlled sensitivity at the declared
settings, not each variant's best achievable performance.

| Variant | Single intervention | What remains shared |
| --- | --- | --- |
| `zero-prior-direction` | Initial arm theta is zero | Learned projection and per-arm lambda |
| `unit-initialization-strength` | Every initial lambda is 1 | Learned projection and arm theta |
| `prior-only` | No routing-learning updates | The matched initial projection, theta and lambda |
| `single-budget-bin` | One bin covers all requested queries | Exact main learned snapshot and all other cost-policy rules |
| `projection-dimension-8` | Retrain projection/prior at dimension 8 | BGE-M3, data, seed, 30/40 epochs, learning rate and margin |

Zero arm directions do **not** remove all preference information: the projection
and initialization strengths still depend on preferences. Equal lambda does not
remove learned directions. A single budget bin does not disable the threshold or
cost estimation. Dimension 8 is not the old 2/5-epoch smoke checkpoint and does not
test an alternative embedding provider. These interpretations must accompany the
figures, not appear only in an appendix.

For `prior-only`, retain a distinct `training_mode=none`, record zero updates and
require snapshot rounds=0. If the existing deployment API requires a learning
selection for disjointness, label it `unused_reference`; never describe it as
observed training data. The other newly trained variants consume selected-arm
feedback through the same full learning order. The single-bin variant reuses the
main snapshot byte-for-byte rather than retraining it.

Changing projection dimension requires three additional preference fits, one per
seed. Their train/holdout split, recipe and cost order stay fixed. If any required
arm has no valid holdout coefficient, stop that condition and report incomplete
coverage; never pick another seed or insert a fallback coefficient.

## Inventory and paired analysis

Primary experiments contain 858 deployment runs. The five additional variants
produce 5 x 2 domains x 3 seeds x 8 budgets = **240** deployment runs, for **1,098**
in total. They require 18 new full-learning streams (three trained variants),
18 learned snapshots, six initial snapshots and six reused main snapshots.
The existing primary learning-size sweep supplies the data-size comparison;
do not rerun it under a new name.

Pair results by dataset, seed and budget fraction. Report all three seed values
and paired differences, with arithmetic mean and min/max. With three fixed seeds,
these describe algorithm/initialization variation on this dataset; they are not
confidence intervals for unseen datasets. RouterBench contains the MMLU subset,
so do not treat the two datasets as independent replicates or pool them into an
overall significance test. Preserve all declared unfavorable and tied results.

Headline deployment metrics are total reward divided by **requested** queries,
served fraction, total recorded cost and estimated cost. Also report mean reward
among served queries, with `null` when none are served. Budget and prices are
historical benchmark accounting units. Estimated affordability does not guarantee
that recorded actual cost stays within budget; report actual excess and its
frequency separately. Never rank a method solely by accuracy among the small
subset it serves, or call a point feasible under actual cost without checking it.

Learning cumulative empirical-oracle regret uses the best stored arm reward for
each query in the same eight-arm pool, only in evaluation after the chosen action.
Oracle rewards never enter selection or updates. They are retrospective dataset
comparisons, not a deployed oracle or a regret bound. The paper's eleven-arm pool,
source-release ambiguity and local embedding substitution remain limitations.

## Completion checks and artifact boundaries

The runner must expand the full declared run inventory before execution. Each
completed run must pin source, split, context, prior, selected tuning configuration,
estimator, code/config and snapshot hashes. Save a completion manifest last, and
only include fully verified runs in a complete aggregate. Missing conditions are
reported explicitly; never average over the surviving seeds as if all ran.

Reuse is allowed only for exact matching identities. Learning prefixes are frozen
snapshots, never independent training reruns. No tuning-learned state is reused for
learning; no deployment result chooses a snapshot. Separate decision traces from
evaluation labels/costs so mutation probes can verify the decisions stay unchanged.
Report all requested queries, including the terminal unserved query and remaining
unserved count. Prior-only results must carry the zero-training annotation through
the aggregate and figures.

## Resource planning

Analytic workload counts, not measured runtimes:

- Primary learning: (32,270 + 12,369) x 3 x 13 = 1,740,921 rounds.
- Additional ablation learning: (32,270 + 12,369) x 3 x 3 = 401,751 rounds.
- Primary deployment requests: (3,227 + 1,275) x 3 x 13 x (8 + 3) = 1,931,358.
- Additional deployment requests: (3,227 + 1,275) x 3 x 5 x 8 = 540,240.

At 64 dimensions, one eight-arm float64 ridge state is about 260 KiB of numeric
arrays; all 36,497 projected float64 contexts are about 17.82 MiB. The raw routing
vector file is about 142.57 MiB and can be memory-mapped. These figures exclude
Python/JSON/dataframe copies and are **not** a measured process-memory bound.

A planning allowance of 3 KiB per trace row for 4,614,270 combined learning and
requested deployment rows is about 13.20 GiB, before checkpoints/manifests and
tuning traces. This is an explicit sizing assumption; sample actual serialized
rows and measure process private working set during the verified tuning run,
then update the estimate before opening full experiment jobs. Available C: disk
was 50.39 GiB during this planning pass; recheck at execution time. Reuse shared
contexts and checkpoint references rather than copying them into every run.

The 8 GiB available-physical-RAM gate and single-worker rule remain unchanged.
Runtime extrapolation needs measurements at dimension 64; the earlier 8D smoke
duration cannot establish it. No paid API is needed for these offline runs.

Task 2.3 remains open until numeric tuning/calibration and measured resource
estimates are available. Task 5.1 remains open until all declared comparisons
have executed and their artifacts have been independently checked.
