# Tasks

## 1. Tuning-only estimator (A05)

- [x] 1.1 Implement explicit-price token estimator and split/provenance validation; verify hand-calculated estimates, invalid/missing tuning inputs and mutation isolation; document API and unavailable RouterBench inputs.

## 2. Independent bin policy (A06)

- [x] 2.1 Implement threshold-first selection, equal bin allocation, rollover, base-bin utilization, fallback and termination; verify analytic and boundary cases including final partial bin, nonfinite values and strict estimated affordability; document paper interpretations.

## 3. Independent actual ledger (A07)

- [x] 3.1 Implement selected-cost accounting and truthful incomplete/overspend reports; verify actual-cost changes do not alter policy decisions, duplicates/forged decisions fail, and invalid observations do not mutate totals; document usage.

## 4. Integration verification

- [x] 4.1 Run complete pytest suite, focused mutation checks and strict OpenSpec validation; review actual diff and record exact commands/results and remaining A08 research-input blockers in docs/cost-policy.md.

## Workflow follow-up

- Parent reviews and owns authorized Git operations; do not stage, commit or push from the subtask.
- Future A08 integration needs verified token/pricing provenance and fixed-router deployment evaluation; no claim of end-to-end paper reproduction follows from these standalone checks.
