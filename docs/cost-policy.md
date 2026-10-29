# Standalone deployment cost control (A05–A07)

Implementation source of truth: [deployment-cost-policy](../openspec/changes/deployment-cost-policy/tasks.md). This module supplies reusable numeric APIs; it is not an A08 deployment runner or evidence of RouterBench performance.

## Token estimates

`TuningTokenCostEstimator.fit(tuning_selection, output_counts_by_query, prices, token_provenance=..., price_provenance=...)` requires exact query and arm coverage. Counts are nonnegative integers, prices are finite nonnegative `TokenPrices(input_per_million, output_per_million)` in one declared currency. Each output mean comes from that arm's tuning responses. Query input counts are supplied separately to `estimate(input_tokens)`; a scalar applies to all arms, or a mapping supplies one tokenizer's count per arm. The module does not tokenize or read responses.

For each arm, `estimated_cost = (input_count * input_price + tuning_mean_output_count * output_price) / 1_000_000`. This follows the estimator described in [the published paper, Section 3.2](https://aclanthology.org/2025.findings-emnlp.1301.pdf#page=6). The test's synthetic counts 10/30, prices 2/4 and input count 100 yield 0.00028; these are test values, not provider prices.

Call `validate_deployment(deployment_selection)` before using the fitted estimator on a deployment stream. It rejects a different split manifest, overlap with tuning, duplicates, emptiness or a non-deployment phase. Use `SplitSelection` from the verified loader; provenance strings and split objects supplied by untrusted code are not authentication. Returned estimates/metadata are copies and cannot change fitted statistics. Token/pricing provenance is mandatory and must describe tokenizer/version/prompt construction and currency/source/vintage when using real data.

## Real-data readiness

Prepared RouterBench outcomes contain historical `actual_cost` and null `estimated_cost`; query schemas do not supply model-specific input/output token counts. The [offline cost export](cost-inputs.md) now provides verified historical token counts and calibrated benchmark prices. The first real learned-router integration has passed with declared smoke bounds. Final tuned bounds and the full experiment protocol remain open. Never infer them from actual deployment costs or outputs. Historical reports about earlier missing inputs remain stage-specific; no paid API was called.

## Independent budget decisions

Construct `BinnedCostPolicy(arm_catalog=..., budget=B, query_count=Q, bin_size=S, lower_bound=LB, upper_bound=UB)`. Call `select(predicted_rewards, estimated_costs)` once per query. A successful immutable decision reserves the estimated cost immediately. A returned `insufficient_budget` decision terminates the stream; selection after failure or Q successes raises. Use a fresh instance for another run.

The policy accepts no outcomes, actual cost, context provider, `observe` function or router object. The caller supplies fixed-router cosine predictions in [-1,1] by default, separately from any UCB exploration scores. An explicit `score_domain="linear"` preserves finite raw ridge means for baseline comparisons; it does not make them calibrated probabilities. `report()` and `decisions` expose copies/immutable records for logging; the module does not write files.

Optional `allowed_arms` restricts both threshold and fallback candidates, while
`tie_break_order` must be a full catalog permutation and changes only exact ties.
Both validate before any budget reservation. Omitted options preserve catalog
ties and full eligibility. See [common deployment](common-deployment-protocol.md)
for the random/fixed baseline interpretation and its limitations.

The primary source is [Section 2.2.3 and Appendix C.2, Algorithm 3](https://aclanthology.org/2025.findings-emnlp.1301.pdf#page=13). For `N=ceil(Q/S)`, each bin gets `B_bin=B/N`; unused funds carry forward. Within a bin, the threshold denominator is

```text
phi(z) = (UB * e / LB)^z * (LB / e)
threshold[arm] = predicted_reward[arm] / phi(z)
```

The eligible set is checked before any average-budget fallback, and its maximum predicted reward wins. Bounds use reward per currency unit and must satisfy `0 < LB <= UB`. Bounds are explicit caller assumptions; the API does not infer or validate them against hidden deployment outcomes.

| Choice | Status and reason |
| --- | --- |
| Equal base bin budgets, carryover, threshold-first selection, highest reward, fallback to remaining budget divided by remaining queries | `PAPER_EXACT` formulas and ordering. |
| Utilization increments by `selected_estimated_cost / B_bin` and resets per bin | `PAPER_EXACT` Algorithm 3. Carryover can yield z>1 despite the main text's z in [0,1]; the implementation reports it and does not clamp or substitute the carryover-augmented denominator. |
| Remaining queries include the current one; final bin uses its actual length | `PAPER_INFERRED`, needed for a defined final-query fallback. |
| Threshold eligibility also requires cost <= currently available allocation | `DEVIATION`, explicit hard feasibility guard omitted by the pseudocode. It may reject a threshold-qualified arm; fallback then uses the paper's average rule. |
| Empty fallback returns terminal failure without spending or advancing | `PAPER_INFERRED` API representation of the paper's termination message. It does not skip the query, borrow future bin allocations, choose an over-average arm, or claim all Q queries can be served. |
| Equal reward ties follow declared catalog order | `PAPER_INFERRED` deterministic tie rule. |
| Negative predictions retain their sign; positive-cost arms fail their negative threshold but can be fallback eligible | `PAPER_INFERRED` literal inequality; no score clipping. |
| Zero budget permits zero-cost arms; utilization remains zero when B_bin=0 | `EXTENSION` for the undefined 0/0 boundary. Free arms with nonnegative reward pass the threshold; negative rewards require fallback. |
| Exact rational arithmetic over canonical decimal input representations, logarithmic threshold comparisons | Engineering `EXTENSION`: avoids accumulating float allocation error and overflow from exponentiation. Reports are finite floats; unrepresentable values fail before mutation. |

The positive-ratio and small-cost assumptions in the source need experimental validation. Zero/negative predictions and free arms fall outside those assumptions; defined code behavior is not a competitive-ratio guarantee. Strict estimated feasibility is guaranteed for the supplied numbers; actual output lengths may violate the real budget.

## Actual cost accounting

Bind `ledger = ActualCostLedger(policy)`. After obtaining only the chosen arm's actual cost, call `ledger.record(decision, actual_cost)`. Records must reference an actual successful decision object from that policy; duplicate, foreign, forged and failed decisions are rejected. Invalid/nonfinite costs or overflowing totals leave the ledger unchanged. Reporting includes a selected-cost trace, observed actual total, estimated total, actual excess above B, and actual-minus-estimate error for recorded decisions only.

`actual_total` is null while any successful decision lacks an actual observation. `actual_total_observed` always reports what has been recorded, `missing_actual_queries` identifies gaps, and `actual_costs_complete` refers to issued successful decisions, not all requested Q queries. The policy's `status` and `successful_queries` show premature termination separately. An observed actual violation is already real even if some costs are still missing. Mutating a returned report cannot change the ledger or the policy.

Minimal synthetic example (one currency unit throughout):

```python
from scripts.cost_policy import ActualCostLedger, BinnedCostPolicy

policy = BinnedCostPolicy(arm_catalog=("a", "b"), budget=3,
                         query_count=2, bin_size=2,
                         lower_bound=1, upper_bound=1)
ledger = ActualCostLedger(policy)
for actual in (4, 1):
    decision = policy.select({"a": 1, "b": 0}, {"a": 1, "b": 1})
    if decision.status != "selected":
        break
    ledger.record(decision, actual)
assert ledger.report()["estimated_total"] == 2
assert ledger.report()["actual_total"] == 5
assert ledger.report()["actual_budget_excess_observed"] == 2
```

Actual cost is reporting data, not a policy input. In real use, only retrieve that cost after selecting its arm. A future runner must separately enforce immutable router state and held-out query selection; this module does not claim to authenticate external callbacks or police caller code.

## Verification

Measured on 2026-10-28 with the existing local environment:

```powershell
.venv/Scripts/python.exe -B -m pytest tests/test_cost_policy.py -q -p no:cacheprovider
.venv/Scripts/python.exe -B -m pytest -q -p no:cacheprovider --tb=short
& 'C:/Users/ASUS/AppData/Roaming/npm/openspec.cmd' validate deployment-cost-policy --strict
git diff --check -- scripts/cost_policy.py tests/test_cost_policy.py docs/cost-policy.md openspec/changes/deployment-cost-policy
```

- Focused: exit 0, **83 passed in 1.05s**. Full suite: exit 0, **242 passed in 6.44s**. Strict OpenSpec validation: exit 0, valid. Diff whitespace check: exit 0.
- Red/green: estimator 32 expected missing-module failures then 32 passes; policy 40 expected missing-class failures then 72 passes; ledger 11 expected missing-class failures then 83 passes.
- An earlier full-suite run exited 1 with **205 passed and 37 failed** in `tests/test_pilot.py`, all `ModuleNotFoundError: scripts.pilot` while that parallel change was in its test-first phase. After its implementation appeared, the fresh complete suite passed as above. This failure was reported rather than hidden or skipped.
- Six mutation probes executed in fresh Python processes with a modified module compiled in memory and the named pytest test selected; each exited 1 with exactly one failed regression. The source file was never altered for mutation runs.

| Deliberate mutation | Regression that detected it |
| --- | --- |
| Substitute input price for output price | `test_tuning_means_and_prices_match_hand_calculation` |
| Remove the hard affordability intersection | `test_hard_affordability_intersects_threshold` |
| Divide fallback allowance by full Q | `test_fallback_uses_remaining_queries_including_current_and_max_reward` |
| Divide rollover utilization by total allocated funds | `test_bin_rollover_last_partial_bin_and_utilization_reset` |
| Preserve utilization when opening a new bin | `test_bin_rollover_last_partial_bin_and_utilization_reset` |
| Discard actual selected costs when updating totals | `test_actual_overspend_is_reported_and_never_changes_estimated_state` |

The example above was executed and its measured trace/report saved to [cost-policy-toy.json](../reports/cost-policy-toy.json), with module SHA-256 and explicitly synthetic input values. It records estimated total 2, actual total 5 and excess 2. This is an accounting fixture, not a RouterBench experiment or trained-router evaluation.

Inspection covered the standalone diff and relevant input-validation/leakage paths. No dedicated security audit was invoked. No lint/typecheck/build/E2E commands are configured for this Python module; these were not represented as passing. Full experiments, billable evaluation and production access were not run. Parent review owns Git operations.
