# Held-out deployment integration

`scripts/deployment.py` connects a fixed `PILOT` model or an explicitly frozen
baseline adapter, query-only token inputs,
`TuningTokenCostEstimator`, and fresh `BinnedCostPolicy`/`ActualCostLedger` instances.
No model reset, UCB selection or feedback update runs during deployment.

```python
from dataclasses import asdict
import json
from scripts.deployment import run_deployment

result = run_deployment(
    policy=learned_pilot,
    queries=queries, outcomes=outcomes, contexts=deployment_contexts,
    learning_selection=learning_selection,
    deployment_selection=deployment_selection,
    estimator=tuning_estimator,
    input_tokens=query_input_counts,
    input_token_provenance="describe query-only tokenizer/count procedure",
    budget=6.0, bin_size=2, lower_bound=0.01, upper_bound=1.0,
)
encoded = json.dumps(asdict(result), allow_nan=False)
```

The example names are caller-supplied objects. An executable, fully synthetic
fixture is `tests/test_deployment.py::test_analytic_deployment_and_partial_bin`:

```powershell
.\.venv\Scripts\python.exe -B -m pytest tests/test_deployment.py -q -p no:cacheprovider
```

`input_tokens` must map exactly the deployment IDs to nonnegative integer input
counts, or to arm-to-count mappings covering the full catalog for different
tokenizers. They must derive from query inputs, not stored model responses or
actual costs. The runner validates shape/counts and records a fingerprint plus
provenance label; it cannot attest how a caller obtained those numbers. The fitted
estimator contributes its tuning-only output means and explicit token prices.

Preflight validates full outcome-grid construction, known and disjoint learning
and deployment IDs, matching manifests/split settings, estimator deployment
provenance, tuning/learning disjointness, catalogs, context dimensions/nonzero
vectors, token coverage and all estimated costs. Contexts, counts and outcomes
are copied before router calls. Estimation is stateless and repeated in preflight
to reject bad future inputs before any reservation.

Every served round follows predict cosine → estimate query cost → budget select
→ reveal selected actual reward/cost → record actual cost and offline oracle
metrics. Oracle values never enter the router or budget policy. The router gets
only a fresh numeric context. Bare baseline objects are rejected by this entry
point. Use `run_policy_deployment(policy, ..., deployment_seed=42)` to explicitly
freeze a supported baseline; its remaining arguments match `run_deployment`.
Linear means are not clipped or converted into cosine predictions. Full semantics
and source distinctions are in [common-deployment-protocol.md](common-deployment-protocol.md).

`DeploymentResult.trace` is a tuple of `DeploymentRound` objects containing
`query_id`, a nested `BudgetDecision`, actual reward/cost, empirical oracle reward
and regret, and cumulative reward/regret. `summary` includes requested, served and
unserved counts; learning/deployment declarations; context/data/model fingerprints;
token and price provenance; the full actual-cost ledger; and served-only metrics.
`served_fraction` and `reward_per_requested_query` also report coverage and total
reward divided by requested queries. The latter assigns zero service utility to
unserved requests, without inventing observed outcome rows.
Use `dataclasses.asdict(result)` for JSON serialization; this API does not write files.

Insufficient budget terminates the stream. `terminal_decision` records the first
unserved query and its failed decision; no fabricated zero-reward trace row is
added. `mean_reward_served` is null when no query is served. Unconstrained empirical
oracle regret covers only served stored outcomes; it is not a budget-aware optimal
comparison and cannot rank policies with different service counts on its own.

The existing cost-policy affordability rule and conventions remain unchanged.
Estimated budget compliance does not guarantee actual budget compliance. Actual
overspend is reported by `summary["budget"]` and does not influence subsequent
estimated-cost choices. See [cost-policy.md](cost-policy.md) for its interpretation
and numerical accounting rules.

Learning split lineage is explicitly `caller_declared_not_attested`: supplying a
learning selection does not prove that a loaded model was trained on those rows.
The caller must use verified frozen selections and maintain true model lineage.
The model fingerprint identifies its current learned matrices/configuration, not
the history that created them. This trusted local boundary is not a sandbox for
malicious Python subclasses or concurrent mutation by unrelated threads.

Synthetic evidence: three rows route a,b,a across bins 1,1,2, with reward 1.75,
estimated cost 3, actual cost 1.1 and served oracle regret 1.25. Other tests verify
bitwise-equal before/after model checkpoints, exact call order, selected feedback,
outcome perturbation invariance, invalid future input rejection before decisions,
actual overspend, and zero/later-service termination. No real tokenization, price
API, embedding provider, dataset rewrite or paper-result claim is involved.

Verification on 2026-10-28: 27 initial tests failed before the module existed.
The implemented runner passed those tests; an additional call-order test then
passed, bringing the targeted suite to **28 passed in 13.66s** (exit 0).
Full workspace `python -B -m pytest -q -p no:cacheprovider` completed with
**299 passed in 58.88s** (exit 0), including concurrently developed units.
`openspec validate heldout-deployment --strict` reported the change valid.

## Frozen baseline integration — 2026-10-29

`FrozenDeploymentRouter` validates and copies learned matrices, parameters and RNG
state, solves its frozen coefficients once, and supports checksummed save/load.
No source-model method or RNG is called during adapted deployment. Random uses
query/seed-derived permutations among eligible tied utilities. Fixed policies
permit only their named arm, even in fallback. The default cost policy remains
cosine-bounded; the adapter explicitly selects the linear domain for linear ridge
baselines. Predictions are preflighted for all requested queries, and reports
include out-of-range linear-score diagnostics and the declared adaptation.

The complete regression suite passed 390 tests (one Windows symlink skip). Real
integration used the verified BGE-M3/prior/cost artifacts, 128 learning queries,
64 held-out deployment queries, 13 methods and three predeclared smoke budgets.
Independent replay checked all 1,664 learning decisions and 1,360 served
deployment decisions across 39 completed runs. This is integration evidence,
not final tuned benchmark results. [Report](../reports/common-deployment-2026-10-29.json).
