# Proposal

## Why

The local pipeline passes CPU and real GPU smoke checks, but it has not trained or evaluated a router on complete real inputs. The next stage must connect verified data, local embeddings, documented costs, held-out training and fair budget evaluation to an auditable research report.

## What Changes

- Audit complete input lengths using the pinned local tokenizer, then freeze a resource-bounded embedding recipe and generate full immutable caches.
- Investigate source-release, task-count and model-identity discrepancies without replacing missing evidence with name guesses.
- Record historical pricing, tokenizer assumptions, tuning-only output estimates and negative-sampling cost order.
- Train and validate real model priors, tune exclusively on tuning data and evaluate the same deployment cost policy across methods.
- Run sequential learning/deployment experiments with multiple seeds, controlled ablations and reproducible aggregate reports and figures.
- Produce a final report/poster and reproduction package, distinguishing substitutions from verified paper reproduction.

## Capabilities

### New Capabilities

- `input-length-audit`: Bounded, reproducible token-length statistics and a training-only recipe decision.
- `research-experiment-protocol`: Evidence-backed source/cost decisions, separated tuning and deployment, fair comparisons and traceable experiment artifacts.

### Modified Capabilities

None. Existing in-flight numerical and local-pipeline contracts remain in force.

## Impact

Extends scripts, tests, configs, documentation and aggregate reports. Existing pinned raw data, splits and smoke caches remain immutable. Uses the installed local GPU environment and public research sources; paid API requests require separate run-specific approval. Heavy work is sequential with one worker and at least 8 GiB available physical RAM.
