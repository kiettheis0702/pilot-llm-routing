# Frozen local experiment protocol

The final machine-readable declaration is
`configs/frozen_experiment_protocol.json`. It freezes the previously declared
primary and ablation choices together with the verified tuning artifacts and
numerical budget calibration. Full deployment experiments have not run yet.
This is the eight-arm **local experiment**, not a completed eleven-arm reproduction.

## Sources and model pool

Use the unchanged pinned `routerbench_0shot` prepared queries/outcomes/splits, the
complete BGE-M3 cache at the selected 2,048-token cap, and the `identity_match`
preference preparation. The frozen configuration records their hashes and paths.
The eight arms are exactly those in the prepared prior manifest; no model is
silently added, removed or renamed. Three unresolved provider bridges remain
excluded: Claude-v2, Mistral-7B-chat and Mixtral-8x7B-chat. Exact serving equivalence
is not established merely by a canonical name match for the identity-match group.

Every replicate retrains the preference projection/prior using seeds 7, 42 and
2026, dimension 64, 30 projection epochs, 40 prior epochs, learning rate .03 and
margin .2. Reuse the verified seed-42 fit only if all identities match. Preference
training/holdout remains 1,245/313 pairs, with grouped prompt separation. Holdout
accuracy determines initialization coefficients; it does not select dimensions,
epochs, seeds or early stopping. If a required coefficient is undefined, the
condition is incomplete; no fallback or replacement seed is allowed.

## Selection and budget inputs

Tuning used 1,000 RouterBench rows and 398 MMLU rows from the original frozen
tuning phase. MMLU filters the `mmlu-` task prefix within every phase, preserving
order, rather than resplitting or drawing replacement rows. Twenty-four candidates
per domain all retain their traces and final states. Independent replay verified
33,552 selected-arm updates and all pre-feedback predictions.

Both domains select PILOT alpha=1, LinUCB alpha=5 and EpochGreedy window=10 by
maximum total tuning reward with the first declared grid entry winning exact
ties. The other methods have no searched parameter. Main learning starts fresh;
never carry a tuning-learned state into learning.

| Domain | Positive ratio LB | Positive ratio UB | Reference estimated cost/query |
| --- | ---: | ---: | ---: |
| RouterBench | 0.039392639811913426 | 108195.24914660997 | 0.00317064 |
| MMLU | 0.1433755409743282 | 113008.78799494584 | 0.0014071105527638191 |

Use the exact JSON values, not rounded display values. LB is half the minimum
positive prequential predicted-score/estimated-cost ratio and UB twice the maximum
across the selected tuning candidates. These are empirical tuning assumptions,
not guaranteed bounds on deployment. All methods in a domain share them.

The reference arm is `gpt-4-1106-preview`. Each domain's absolute reference budget
is its mean estimated tuning reference-arm cost times the number of requested
deployment queries. Budget fractions are .05/.1/.2/.3/.5/.75/1/1.25. Domain-specific
output-length means use only its tuning subset. Input counts, historical prices
and the explicit cost-scale calibration are pinned; they are benchmark accounting
units and do not incur API charges.

## Complete inventory and reporting

The [primary plan](../configs/main_experiment_plan.json) declares 78 full learning
streams, 312 nested-prefix snapshots and 858 deployment runs across the two
datasets, three seeds and 13 methods. Learning fractions .1/.25/.5/1 use ceiling
prefix lengths; the first three prefixes use budget fraction .5. The full-learning
snapshot uses all eight budget fractions. Bin size is 100.

The [ablation plan](ablation-protocol.md) adds five controlled variants and 240
deployment runs, for 1,098 runs total. Preserve all paired controls and explicit
zero-training annotations for the prior-only variant. Use the common frozen
[deployment adapter](common-deployment-protocol.md) for every method. Freeze the
main parameters for the ablations; do not retune them on deployment outcomes.

Report reward per requested query, served fraction, mean reward among served
queries, estimates and recorded costs, actual budget excess, and retrospective
learning oracle regret. Show each seed and paired mean/min/max; do not claim a
population confidence interval from three fixed seeds or treat nested MMLU and
RouterBench as independent datasets. Missing conditions keep reports incomplete.
Fixed-arm scores are indicators and random scores are constant tie utilities;
linear ridge scores are unclipped. Shared cost bounds do not make these different
score scales calibrated probabilities. This is a declared comparison adaptation.

## Measured resources and execution

On this machine, the 64D real tuning commands completed in 13.696 seconds for
24,000 RouterBench rounds and 9.887 seconds for 9,552 MMLU rounds. Seven MMLU
process samples observed at most 306.094 MiB combined private physical working set
for the launcher/interpreter; this is a sampled maximum, not a proven peak or a
future memory cap. Available physical RAM during those samples stayed above
9.37 GiB. The resource sample file is pinned in the frozen configuration.

The 2,142,672 planned learning rounds would take about 20.4 minutes if they ran at
the observed RouterBench tuning average. This is a planning extrapolation only:
method mix, matrix evolution, serialization and IO differ. It excludes remaining
preference fits, deployment and independent verification. Measure the first full
stream and refine the estimate without changing the declared experiment choices.

The conservative trace allowance remains 3 KiB per row, about 13.20 GiB for
4,614,270 planned learning/deployment rows, plus checkpoint/tuning overhead. Inspect
free disk again before full runs and retain shared artifacts by reference. Neither
this storage assumption nor the sampled working set replaces the mandatory
8 GiB available-RAM preflight. Run one heavy job and one worker at a time; never
interrupt other active work or change thread counts to compensate for pressure.

## Paper differences and completion boundary

- The authors' exact RouterBench release/shot selection and the relationship
  between 86 local raw task labels and the paper's 64 tasks remain unconfirmed.
  Preserve the frozen local split; do not manufacture a task grouping.
- The local Arena release/preparation, 313-pair holdout rather than the reported
  500-example evaluation, eight-arm evidence pool, BGE-M3 embedding substitution,
  declared training settings and derived MMLU split limit reproduction claims.
- Triplet cost order uses a declared 1:1 input/output reference mix. Historical
  token counting approximates non-OpenAI tokenizers. Estimated affordability may
  permit recorded actual cost to exceed a budget; report the observed excess.
- EpochGreedy/ExploreOnly interpretations and common baseline deployment scoring
  are documented local adaptations. All tuning diagnostics are selection evidence,
  not held-out performance or proof that PILOT outperforms another method.

These differences remain part of every final report. The full eleven-arm source
equivalence target stays open. Freezing this local protocol completes preparation;
it does not complete experiment execution, ablations, figures, report, poster or
final reproduction checks.
