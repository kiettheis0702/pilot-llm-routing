## Purpose

Provide comparable offline learning baselines that train only on chosen-arm feedback and reproduce seeded action traces.

## ADDED Requirements

### Requirement: Baseline lifecycle and validity
Policies SHALL validate fixed arm catalogs, context dimension/finite values and rewards in [0,1]. Reset SHALL restore learned state and seeded random state. Ties SHALL use catalog order. Random and fixed policies SHALL not learn from unchosen outcomes.

#### Scenario: Replay
- **WHEN** a policy is reset with the same seed and given identical feedback
- **THEN** its action and prediction sequence is identical

### Requirement: Disjoint ridge learning
Linear UCB SHALL use ridge A=lambda I, b=0, linear prediction x dot solve(A,b), and exploration alpha sqrt(x dot solve(A,x)). Only the selected arm SHALL update A by xx-transpose and b by reward*x. Invalid or nonfinite updates SHALL leave state unchanged.

#### Scenario: Hand-calculated update
- **WHEN** lambda=1, alpha=1, x=[1,0] and arm a observes reward 1
- **THEN** a's prediction is 0.5, a's UCB is 0.5+sqrt(0.5), and unchosen b's UCB remains 1

### Requirement: Explicit exploration schedules
Explore Only SHALL choose maximum ridge uncertainty. Epoch Greedy SHALL choose uniformly at rounds 0,window,2*window and otherwise choose maximum linear predicted reward. Both SHALL learn only from observed feedback and identify these underspecified paper details as local interpretations.

#### Scenario: Window boundary
- **WHEN** window=3 and six chosen-feedback rounds run
- **THEN** exploration occurs at rounds 0 and 3; remaining rounds exploit
