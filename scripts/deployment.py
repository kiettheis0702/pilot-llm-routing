"""Held-out fixed-PILOT deployment; outcomes remain evaluation-only."""
from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from numbers import Integral

import numpy as np
import pandas as pd

from scripts.bandit_contracts import ContextTable
from scripts.cost_policy import (ActualCostLedger, BinnedCostPolicy, BudgetDecision,
                                 TuningTokenCostEstimator)
from scripts.data_gate import OfflineBandit
from scripts.pilot import PILOT
from scripts.deployment_adapter import FrozenDeploymentRouter
from scripts.splits import SplitSelection


@dataclass(frozen=True)
class DeploymentRound:
    query_id: str
    decision: BudgetDecision
    reward: float
    actual_cost: float
    oracle_reward: float
    empirical_oracle_regret: float
    cumulative_reward: float
    cumulative_empirical_oracle_regret: float


@dataclass(frozen=True)
class DeploymentResult:
    trace: tuple[DeploymentRound, ...]
    summary: dict


def _fingerprint(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        allow_nan=False, separators=(",", ":")).encode("utf-8")).hexdigest()


def _ids(selection: SplitSelection, phase: str) -> tuple[str, ...]:
    if not isinstance(selection, SplitSelection) or selection.split != phase:
        raise ValueError(f"Expected {phase} selection")
    if not isinstance(selection.manifest_sha256, str) or not selection.manifest_sha256.strip():
        raise ValueError("Manifest fingerprint is required")
    ids = selection.query_ids
    if (isinstance(ids, (str, bytes)) or not ids
            or any(not isinstance(q, str) or not q.strip() for q in ids)
            or len(set(ids)) != len(ids)):
        raise ValueError("Selection IDs must be nonempty unique strings")
    return tuple(ids)


def _tokens(value, arms):
    if isinstance(value, Mapping):
        if set(value) != set(arms):
            raise ValueError("Input token catalog mismatch")
        return {a: _tokens(value[a], ()) for a in arms}
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError("Input token counts must be nonnegative integers")
    return int(value)


def run_deployment(policy: PILOT | FrozenDeploymentRouter, queries: pd.DataFrame, outcomes: pd.DataFrame,
                   contexts: ContextTable, learning_selection: SplitSelection,
                   deployment_selection: SplitSelection,
                   estimator: TuningTokenCostEstimator, input_tokens: Mapping, *,
                   input_token_provenance: str, budget: float, bin_size: int,
                   lower_bound: float, upper_bound: float) -> DeploymentResult:
    """Run a fresh budget stream without reset or observe on the learned router.

    Learning split lineage and query-only token derivation are caller declarations,
    not attestations. No outcomes are supplied to the router, estimator or selector.
    """
    if not isinstance(policy, (PILOT, FrozenDeploymentRouter)):
        raise ValueError("Deployment requires PILOT or an explicit frozen baseline adapter")
    adapted = isinstance(policy,FrozenDeploymentRouter)
    if not isinstance(estimator, TuningTokenCostEstimator):
        raise ValueError("Deployment requires a tuning-only cost estimator")
    learned = _ids(learning_selection, "learning")
    ids = _ids(deployment_selection, "deployment")
    for key in ("manifest_sha256", "seed", "tuning_size", "learning_ratio"):
        if getattr(learning_selection, key) != getattr(deployment_selection, key):
            raise ValueError("Learning/deployment split manifests or settings differ")
    if set(learned).intersection(ids):
        raise ValueError("Learning and deployment overlap")
    estimator.validate_deployment(deployment_selection)
    estimator_metadata = estimator.metadata()
    if set(learned).intersection(estimator_metadata["tuning_query_ids"]):
        raise ValueError("Learning and tuning overlap")
    if (not isinstance(input_token_provenance, str) or not input_token_provenance.strip()):
        raise ValueError("Input token provenance is required")
    if not isinstance(input_tokens, Mapping) or set(input_tokens) != set(ids):
        raise ValueError("Input token counts must cover exactly deployment IDs")

    environment = OfflineBandit(queries, outcomes, arm_catalog=policy.arms)
    if not set(learned + ids) <= set(queries.query_id):
        raise ValueError("Unknown selection query IDs")
    if set(estimator_metadata["prices"]) != set(policy.arms):
        raise ValueError("Estimator and model catalogs differ")
    try:
        selected_contexts = ContextTable({q: contexts.get(q) for q in ids}, label=contexts.label)
    except KeyError as exc:
        raise ValueError("Missing deployment context") from exc
    if selected_contexts.dimension != policy.dimension:
        raise ValueError("Model/context dimensions differ")
    if any(not np.any(selected_contexts.get(q)) for q in ids):
        raise ValueError("Zero deployment context")
    counts = {q: _tokens(input_tokens[q], policy.arms) for q in ids}
    # Stateless preflight catches invalid future estimates before any reservation.
    for q in ids:
        estimator.estimate(counts[q])
    predictions_by_query = ({q:policy.predict_rewards(selected_contexts.get(q)) for q in ids}
                            if adapted else None)
    cost_policy = BinnedCostPolicy(arm_catalog=policy.arms, budget=budget,
        query_count=len(ids), bin_size=bin_size, lower_bound=lower_bound, upper_bound=upper_bound,
        score_domain=policy.score_domain if adapted else 'cosine')
    ledger = ActualCostLedger(cost_policy)
    selected_outcomes = outcomes.loc[outcomes.query_id.isin(ids),
        ["query_id", "arm_id", "reward", "actual_cost"]].copy()
    selected_outcomes = selected_outcomes.astype({"reward": float, "actual_cost": float})
    evaluation = selected_outcomes.set_index(["query_id", "arm_id"])
    selected_queries = queries.set_index("query_id").loc[list(ids), ["query", "task"]].reset_index()
    model_state = policy.source_state() if adapted else {"arms": list(policy.arms), "dimension": policy.dimension,
                   "A": policy.A.tolist(), "b": policy.b.tolist(), "rounds": policy.rounds,
                   "alpha": policy.alpha, "prior_provenance": policy.provenance}
    provenance = {"queries_sha256": _fingerprint(selected_queries.to_dict("records")),
                  "outcomes_sha256": _fingerprint(selected_outcomes.sort_values(
                      ["query_id", "arm_id"]).to_dict("records")),
                  "input_tokens_sha256": _fingerprint(counts)}
    model_digest = _fingerprint(model_state)
    trace = []
    total_reward = total_regret = 0.0
    action_counts = dict.fromkeys(policy.arms, 0)
    terminal = None
    for query_id in ids:
        predictions = (predictions_by_query[query_id] if adapted else
                       policy.predict_rewards(selected_contexts.get(query_id)))
        estimates = estimator.estimate(counts[query_id])
        options = policy.selection_options(query_id) if adapted else {}
        decision = cost_policy.select(predictions, estimates, **options)
        if decision.status != "selected":
            terminal = {"query_id": query_id, "decision": asdict(decision)}
            break
        feedback = environment.step(query_id, decision.arm_id)
        ledger.record(decision, feedback.actual_cost)
        oracle = float(evaluation.loc[query_id, "reward"].max())
        regret = oracle - feedback.reward
        total_reward += feedback.reward
        total_regret += regret
        if not math.isfinite(total_reward) or not math.isfinite(total_regret):
            raise ValueError("Nonfinite deployment metrics")
        action_counts[feedback.arm_id] += 1
        trace.append(DeploymentRound(query_id, decision, feedback.reward, feedback.actual_cost,
                     oracle, regret, total_reward, total_regret))
    served = len(trace)
    summary = {"schema_version": 1, "phase": "deployment", "status": cost_policy.report()["status"],
        "requested_queries": len(ids), "served_queries": served, "unserved_queries": len(ids)-served,
        "terminal_decision": terminal, "learning_split": asdict(learning_selection),
        "deployment_split": asdict(deployment_selection),
        "model": {"policy": f"{type(policy).__module__}.{type(policy).__qualname__}",
                  "sha256": model_digest, "prior_provenance": policy.provenance,
                  "learning_provenance": "caller_declared_not_attested"},
        "context": {"label": selected_contexts.label, "dimension": selected_contexts.dimension,
                    "sha256": selected_contexts.fingerprint(ids)},
        "data": provenance, "input_token_provenance": input_token_provenance,
        "estimator": estimator_metadata, "budget": ledger.report(),
        "metrics": {"total_reward": total_reward,
                    "mean_reward_served": total_reward / served if served else None,
                    "served_fraction": served / len(ids),
                    "reward_per_requested_query": total_reward / len(ids),
                    "empirical_oracle_regret_served": total_regret,
                    "action_counts": action_counts},
        "metric_interpretation": "unconstrained_empirical_oracle_on_served_stored_outcomes_only"}
    if adapted:
        summary['deployment_adapter'] = policy.metadata()
        summary['model']['policy'] = model_state['policy_type']
        all_scores = [score for predictions in predictions_by_query.values() for score in predictions.values()]
        summary['prediction_diagnostics'] = dict(scope='all_requested_queries_before_budget_selection',
            count=len(all_scores),below_zero=sum(v<0 for v in all_scores),above_one=sum(v>1 for v in all_scores),
            minimum=min(all_scores),maximum=max(all_scores))
    summary['service_utility_interpretation'] = 'total_observed_reward_divided_by_requested_count; unserved_service_utility_zero'
    result = DeploymentResult(tuple(trace), summary)
    json.dumps(asdict(result), allow_nan=False)
    return result


def run_policy_deployment(policy, *args, deployment_seed=42, **kwargs) -> DeploymentResult:
    """Explicitly snapshot a supported policy, then use the shared deployment runner."""
    frozen = FrozenDeploymentRouter(policy,seed=deployment_seed)
    return run_deployment(frozen,*args,**kwargs)
