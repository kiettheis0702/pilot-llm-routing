# Spec Delta

## Purpose

Control estimated deployment spending independently of the learned router, without leaking observed output lengths into estimates or concealing actual spending.

## ADDED Requirements

### Requirement: Tuning-only token estimates
The estimator SHALL fit per-arm mean output tokens only from the declared tuning selection, require explicit input/output prices and token provenance, and estimate cost from caller-supplied query input counts. It MUST reject incomplete or extra tuning IDs, inconsistent arm catalogs, invalid counts/prices, held-out fitting selections and mismatched or overlapping deployment selections.

#### Scenario: Analytic estimate
- **WHEN** tuning outputs for one arm are 10 and 30 tokens, input tokens are 100, and prices per million are 2 input and 4 output
- **THEN** its estimated cost is 0.00028 independent of any actual deployment output

### Requirement: Threshold selection
The policy SHALL first consider arms with estimated cost at most available bin budget and reward divided by `(UB*e/LB)^z*(LB/e)`, then select maximum predicted reward. Bounds MUST be finite and satisfy `0 < LB <= UB`; predictions MUST be finite cosine values in [-1,1]. Ties SHALL follow declared catalog order.

#### Scenario: Budget beats a permissive threshold
- **WHEN** a threshold admits a cost exceeding available budget
- **THEN** that arm cannot be selected

### Requirement: Bins and rollover
For Q positive queries and bin size S, the policy SHALL allocate B/ceil(Q/S) per bin, carry unused funds forward, reset z each bin, and update z by estimated cost divided by the base bin allocation. It SHALL handle a final partial bin using its actual query count. Numeric accounting MUST preserve estimated spend at most allocated and total budget.

#### Scenario: Final partial bin
- **WHEN** Q=3, S=2, B=4 and the first two costs are 0.5 each
- **THEN** the final one-query bin starts with 3 available and z=0

### Requirement: Fallback and termination
Only if the threshold-eligible affordable set is empty, the policy SHALL consider costs at most available budget divided by queries remaining in the current bin, including the current query. An empty fallback SHALL produce terminal insufficient-budget status without spending or advancing. Further selection after failure or Q successful decisions MUST fail.

#### Scenario: Affordable but above average fallback
- **WHEN** no arm passes the threshold and the cheapest arm costs 3 with 4 available for 2 queries
- **THEN** the policy terminates as insufficient budget even though 3 is currently affordable

### Requirement: Actual-cost separation
A separate ledger SHALL record one actual selected-arm cost per successful decision, reject duplicates and invalid costs, and report estimated/actual totals and actual budget excess separately. Actual costs MUST NOT alter policy budgets or choices. Reports MUST identify missing actual observations instead of treating them as zero.

#### Scenario: Estimated feasibility with actual overspend
- **WHEN** selected estimated costs total 2 under budget 3 and actual costs total 5
- **THEN** reports show estimated spend 2, actual spend 5 and actual excess 2

### Requirement: Explicit numerical behavior
The implementation SHALL reject invalid/nonfinite settings and arithmetic failures before mutating state. Zero budget and zero costs SHALL be supported without dividing by zero, while negative predictions SHALL not be silently clipped. Caller-owned mappings and returned reports MUST NOT mutate future estimates or policy decisions.

#### Scenario: Negative cosine fallback
- **WHEN** every prediction is negative, positive cost fits the remaining average budget, and no threshold arm is eligible
- **THEN** fallback selects the highest predicted reward without changing its sign
