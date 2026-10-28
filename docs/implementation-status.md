# Implementation and research readiness — 2026-10-28

The current branch implements a runnable offline numerical pipeline. No Figure 3/4
reproduction or eleven-arm preference training is claimed. OpenSpec change files
are the source of truth for implemented units; this table maps them to the supplied
handoff's broader tickets.

| Tickets | Status | Evidence / remaining prerequisite |
| --- | --- | --- |
| G0 | Conditional | Pinned RouterBench 0-shot prepared; release/task/model ambiguities remain. [Audit](data-gate-audit.md). |
| B01–B07 | Implemented | Contracts, splits, metrics, five policy families and comparison CLI; toy and 64-row real-data plumbing runs with explicitly synthetic contexts. [Baselines](baselines.md). |
| P01/P04 | Preparation complete, real readiness blocked | 374 eligible exact-ID pairs, 299/75 grouped split, nine arms uncovered. [Readiness](preference-readiness.md). |
| P02 | Partial | Validated supplied-vector cache and provenance exist; real embedding generation/model choice and scalable storage are unresolved. |
| P03/P05–P07 | Numerical core implemented | Triplets, gradients, two-stage SGD, validation/lambda and checkpoints verified on tiny synthetic data. Actual model-pool training blocked. [Pretraining](pretraining-core.md). |
| A01–A07 | Implemented mathematical contracts | Prior-informed cosine UCB, selected-only updates, checkpointing and standalone cost policy with analytic/negative tests. [PILOT](pilot-core.md), [cost](cost-policy.md). |
| A08 | Synthetic integration complete; RouterBench acceptance blocked | Frozen deployment runner plus full synthetic CLI work. No real learned prior, embedding/token-price inputs, or baseline deployment score protocol. [Deployment](deployment.md), [toy](toy-pipeline.md). |
| E01–E06 / V01–V06 | Not run | A defensible Figure 3/4 protocol requires settled source/model/embedding/cost inputs and resource estimates. No hyperparameter or seed is selected on deployment data. |
| C01–C03 | Not research experiments | Unit/toy cases verify sensitivity and cost-error reporting; controlled hypotheses, replicates and statistics remain future work. |
| F01/F02 | Engineering package available | Locked environment, executable toy/baseline commands and plots derived from saved traces. Fresh-checkout evidence is recorded in the verification log. |
| F03 | Blocked | A results poster needs real experiments; synthetic evidence cannot stand in for them. |
| F04 | Scoped repository hygiene | Raw datasets stay external; explicit toy JSON and aggregate evidence are tracked. File-size/type checks are recorded in the log; no dedicated security audit implied. |

## Inputs needed to advance research

1. Resolve Arena release equivalence and the nine remaining model identity/version
   mappings, or explicitly approve a reduced arm pool/substitution as a deviation.
2. Supply a versioned embedding cache, or choose and authorize an embedding source
   and its resource/cost budget. Pin prompt construction, dimensionality and normalization.
3. Pin the per-arm token-cost order for negatives, actual query/tokenizer counts,
   tuning-only response-token counts, historical input/output prices and cost bounds.
4. Resolve 0-shot/5-shot selection and 86 raw task labels versus the paper's 64,
   then lock fair baseline deployment behavior and a bounded experiment protocol.

Until those inputs exist, additional benchmark-looking numbers would measure
unverified substitutions. The current synthetic run exercises the implementation
and records that limitation in every report. No paid calls, large model downloads
or full-scale training were performed.
