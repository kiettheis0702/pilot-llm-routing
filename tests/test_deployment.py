import json
from dataclasses import asdict, replace

import numpy as np
import pandas as pd
import pytest

from scripts.bandit_contracts import ContextTable
from scripts.cost_policy import BinnedCostPolicy, TokenPrices, TuningTokenCostEstimator
from scripts.pilot import PILOT
from scripts.splits import SplitSelection


def selection(phase, ids, manifest="synthetic-manifest"):
    return SplitSelection(tuple(ids), phase, manifest, 42, 1, 10, len(ids))


def inputs():
    ids = ["t", "l", "d1", "d2", "d3"]
    queries = pd.DataFrame({"query_id": ids, "query": ["query "+q for q in ids], "task": "toy"})
    outcomes = pd.DataFrame([{"query_id": q, "arm_id": a, "reward": 1., "actual_cost": .1}
                             for q in ids for a in ["a", "b"]])
    for q, a, reward, cost in [("d1", "a", .25, .2), ("d2", "b", 1., .5), ("d3", "a", .5, .4)]:
        mask = (outcomes.query_id == q) & (outcomes.arm_id == a)
        outcomes.loc[mask, ["reward", "actual_cost"]] = [reward, cost]
    model = PILOT(("a", "b"), 2, priors={"a": [2, 0], "b": [0, 1]},
                  lambdas={"a": 2, "b": 1}, provenance="synthetic prior", alpha=100)
    estimator = TuningTokenCostEstimator.fit(selection("tuning", ["t"]),
        {"t": {"a": 1, "b": 1}}, {a: TokenPrices(1_000_000, 0) for a in ["a", "b"]},
        token_provenance="synthetic tuning counts", price_provenance="synthetic prices")
    return dict(policy=model, queries=queries, outcomes=outcomes,
                contexts=ContextTable({"d1": [1, 0], "d2": [0, 1], "d3": [1, 1]}, label="synthetic"),
                learning_selection=selection("learning", ["l"]),
                deployment_selection=selection("deployment", ["d1", "d2", "d3"]),
                estimator=estimator, input_tokens={"d1": 1, "d2": 1, "d3": 1},
                input_token_provenance="synthetic query-only counts", budget=6, bin_size=2,
                lower_bound=.01, upper_bound=1)


def run(**changes):
    from scripts.deployment import run_deployment
    values = inputs()
    values.update(changes)
    return run_deployment(**values)


def test_analytic_deployment_and_partial_bin():
    result = run()
    assert [r.query_id for r in result.trace] == ["d1", "d2", "d3"]
    assert [r.decision.arm_id for r in result.trace] == ["a", "b", "a"]
    assert [r.decision.bin_index for r in result.trace] == [1, 1, 2]
    assert result.trace[-1].decision.remaining_queries_in_bin == 1
    summary = result.summary
    assert summary["status"] == "complete"
    assert summary["requested_queries"] == summary["served_queries"] == 3
    assert summary["unserved_queries"] == 0
    assert summary["metrics"]["total_reward"] == 1.75
    assert summary["metrics"]["mean_reward_served"] == pytest.approx(1.75/3)
    assert summary["metrics"]["empirical_oracle_regret_served"] == 1.25
    assert summary["budget"]["estimated_total"] == 3
    assert summary["budget"]["actual_total"] == 1.1
    assert summary["model"]["learning_provenance"] == "caller_declared_not_attested"
    json.dumps(asdict(result), allow_nan=False)


def test_router_never_resets_or_observes_and_is_bitwise_unchanged(monkeypatch, tmp_path):
    from scripts.deployment import run_deployment
    values = inputs()
    model = values["policy"]
    model.observe("a", [1, 1], .25)
    model.save_checkpoint(tmp_path / "before.json")
    def forbidden(*args, **kwargs): raise AssertionError("deployment must not learn/reset/select UCB")
    for method in ["observe", "reset", "select"]:
        monkeypatch.setattr(model, method, forbidden)
    seen = []
    predict = model.predict_rewards
    def spy(context):
        assert isinstance(context, np.ndarray)
        seen.append(context.copy())
        return predict(context)
    monkeypatch.setattr(model, "predict_rewards", spy)
    run_deployment(**values)
    model.save_checkpoint(tmp_path / "after.json")
    assert (tmp_path / "before.json").read_bytes() == (tmp_path / "after.json").read_bytes()
    assert len(seen) == 3


@pytest.mark.parametrize("damage", ["phase", "overlap", "manifest", "settings", "duplicate", "unknown",
    "tuning_overlap", "estimator_manifest", "missing_context", "zero_context", "dimension",
    "missing_tokens", "extra_tokens", "negative_tokens", "bool_tokens", "token_catalog",
    "model_catalog", "outcome_grid", "estimator_catalog", "empty_token_provenance"])
def test_invalid_inputs_fail_before_any_decision(monkeypatch, damage):
    from scripts.deployment import run_deployment
    values = inputs()
    dep = values["deployment_selection"]
    if damage == "phase": values["deployment_selection"] = replace(dep, split="learning")
    elif damage == "overlap": values["learning_selection"] = selection("learning", ["d1"])
    elif damage == "manifest": values["learning_selection"] = selection("learning", ["l"], "other")
    elif damage == "settings": values["learning_selection"] = replace(values["learning_selection"], seed=8)
    elif damage == "duplicate": values["deployment_selection"] = replace(dep, query_ids=("d1", "d1"))
    elif damage == "unknown": values["learning_selection"] = selection("learning", ["missing"])
    elif damage == "tuning_overlap": values["deployment_selection"] = selection("deployment", ["t"])
    elif damage == "estimator_manifest":
        values["learning_selection"] = selection("learning", ["l"], "other")
        values["deployment_selection"] = replace(dep, manifest_sha256="other")
    elif damage == "missing_context": values["contexts"] = ContextTable({"d1": [1, 0]}, label="partial")
    elif damage == "zero_context": values["contexts"] = ContextTable(dict.fromkeys(dep.query_ids, [0, 0]), label="zero")
    elif damage == "dimension": values["contexts"] = ContextTable(dict.fromkeys(dep.query_ids, [1]), label="wrong")
    elif damage == "missing_tokens": del values["input_tokens"]["d3"]
    elif damage == "extra_tokens": values["input_tokens"]["l"] = 1
    elif damage == "negative_tokens": values["input_tokens"]["d3"] = -1
    elif damage == "bool_tokens": values["input_tokens"]["d3"] = True
    elif damage == "token_catalog": values["input_tokens"]["d3"] = {"a": 1}
    elif damage == "model_catalog":
        values["policy"] = PILOT(("a",), 2, priors={"a": [1, 0]}, lambdas={"a": 1}, provenance="toy")
    elif damage == "outcome_grid": values["outcomes"] = values["outcomes"].iloc[1:]
    elif damage == "estimator_catalog":
        values["estimator"] = TuningTokenCostEstimator.fit(selection("tuning", ["t"]),
            {"t": {"a": 1}}, {"a": TokenPrices(1, 1)}, token_provenance="toy", price_provenance="toy")
    else: values["input_token_provenance"] = " "
    decisions = []
    def decision_spy(*args, **kwargs):
        decisions.append(1)
        raise AssertionError("invalid inputs reached decision")
    monkeypatch.setattr(BinnedCostPolicy, "select", decision_spy)
    with pytest.raises(ValueError): run_deployment(**values)
    assert decisions == []


def test_outcomes_never_influence_decisions_and_overspend_is_visible():
    from scripts.deployment import run_deployment
    first = run()
    values = inputs()
    values["outcomes"]["actual_cost"] = 100
    values["outcomes"].loc[values["outcomes"].arm_id == "b", "reward"] = .1
    second = run_deployment(**values)
    assert [r.decision for r in first.trace] == [r.decision for r in second.trace]
    assert second.summary["budget"]["actual_total"] == 300
    assert second.summary["budget"]["actual_budget_violation_observed"] is True
    assert second.summary["budget"]["actual_budget_excess_observed"] == 294


def test_early_termination_has_no_imputed_rewards():
    result = run(budget=0)
    assert result.trace == ()
    assert result.summary["status"] == "insufficient_budget"
    assert result.summary["served_queries"] == 0
    assert result.summary["unserved_queries"] == 3
    assert result.summary["metrics"]["mean_reward_served"] is None
    assert result.summary["terminal_decision"]["query_id"] == "d1"
    assert result.summary["budget"]["actual_observations"] == 0


def test_later_termination_preserves_served_only_metrics():
    result = run(budget=2, bin_size=3)
    assert len(result.trace) == 2
    assert result.summary["unserved_queries"] == 1
    assert result.summary["metrics"]["total_reward"] == 1.25
    assert result.summary["metrics"]["mean_reward_served"] == .625
    assert result.summary["terminal_decision"]["query_id"] == "d3"


def test_arm_specific_query_token_counts():
    result = run(input_tokens={q: {"a": 1, "b": 2} for q in ["d1", "d2", "d3"]}, budget=10)
    assert [r.decision.estimated_cost for r in result.trace] == [1, 2, 1]


def test_linear_policy_cannot_be_silently_clipped_into_cosine():
    from scripts.policies import LinearUCB
    with pytest.raises(ValueError): run(policy=LinearUCB(("a", "b"), 2))


def test_decision_precedes_selected_feedback_and_actual_accounting(monkeypatch):
    from scripts.cost_policy import ActualCostLedger
    from scripts.data_gate import OfflineBandit
    from scripts.deployment import run_deployment
    values = inputs()
    events = []
    predict = values["policy"].predict_rewards
    estimate = values["estimator"].estimate
    select, step, record = BinnedCostPolicy.select, OfflineBandit.step, ActualCostLedger.record
    def prediction(context):
        events.append("predict")
        return predict(context)
    def estimation(counts):
        if events: events.append("estimate")  # Ignore stateless preflight.
        assert isinstance(counts, int)
        return estimate(counts)
    def selection_spy(self, rewards, costs):
        assert set(rewards) == set(costs) == {"a", "b"}
        events.append("select")
        return select(self, rewards, costs)
    def feedback(self, query_id, arm_id):
        assert (query_id, arm_id) in {("d1", "a"), ("d2", "b"), ("d3", "a")}
        events.append("selected_feedback")
        return step(self, query_id, arm_id)
    def accounting(self, decision, cost):
        events.append("record_actual")
        return record(self, decision, cost)
    monkeypatch.setattr(values["policy"], "predict_rewards", prediction)
    monkeypatch.setattr(values["estimator"], "estimate", estimation)
    monkeypatch.setattr(BinnedCostPolicy, "select", selection_spy)
    monkeypatch.setattr(OfflineBandit, "step", feedback)
    monkeypatch.setattr(ActualCostLedger, "record", accounting)
    run_deployment(**values)
    assert events == ["predict", "estimate", "select", "selected_feedback", "record_actual"] * 3
