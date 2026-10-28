"""Analytic fixtures catch wrong prices, leakage, budget branches and accounting."""
from dataclasses import replace
import math

import pytest

from scripts.splits import SplitSelection


def selection(split="tuning", ids=("t1", "t2"), digest="fixture-manifest"):
    return SplitSelection(ids, split, digest, 42, 2, 10, len(ids))


def estimator():
    from scripts.cost_policy import TokenPrices, TuningTokenCostEstimator
    return TuningTokenCostEstimator.fit(
        selection(), {"t1": {"a": 10, "b": 50}, "t2": {"a": 30, "b": 10}},
        {"a": TokenPrices(2, 4), "b": TokenPrices(1, 10)},
        token_provenance="synthetic counts", price_provenance="synthetic per-million prices")


def test_tuning_means_and_prices_match_hand_calculation():
    fitted = estimator()
    assert fitted.mean_output_tokens == {"a": 20, "b": 30}
    assert fitted.estimate(100) == pytest.approx({"a": 0.00028, "b": 0.0004})
    assert fitted.estimate({"a": 100, "b": 200}) == pytest.approx(
        {"a": 0.00028, "b": 0.0005})
    fitted.validate_deployment(selection("deployment", ("d1",)))
    metadata = fitted.metadata()
    assert metadata["tuning_query_ids"] == ["t1", "t2"]
    assert metadata["token_provenance"] == "synthetic counts"


def test_estimator_copies_tuning_prices_and_returned_reports():
    from scripts.cost_policy import TokenPrices, TuningTokenCostEstimator
    outputs = {"t1": {"a": 10}, "t2": {"a": 30}}
    prices = {"a": TokenPrices(2, 4)}
    fitted = TuningTokenCostEstimator.fit(selection(), outputs, prices,
        token_provenance="synthetic", price_provenance="synthetic")
    outputs["t1"]["a"] = 100000
    prices["a"] = TokenPrices(10000, 10000)
    fitted.mean_output_tokens["a"] = 100000
    fitted.metadata()["tuning_query_ids"].append("d1")
    fitted.estimate(100)["a"] = 999
    assert fitted.estimate(100)["a"] == pytest.approx(0.00028)
    fitted.validate_deployment(selection("deployment", ("d1",)))


@pytest.mark.parametrize("bad", [-1, 1.5, True, math.nan, math.inf, "10", None])
def test_invalid_token_counts_fail_for_fit_and_estimate(bad):
    from scripts.cost_policy import TokenPrices, TuningTokenCostEstimator
    with pytest.raises(ValueError):
        TuningTokenCostEstimator.fit(selection(), {"t1": {"a": bad}, "t2": {"a": 2}},
            {"a": TokenPrices(1, 1)}, token_provenance="toy", price_provenance="toy")
    with pytest.raises(ValueError):
        estimator().estimate(bad)


@pytest.mark.parametrize("bad", [-1, math.nan, math.inf, True, "1", None])
def test_invalid_prices_rejected(bad):
    from scripts.cost_policy import TokenPrices
    with pytest.raises(ValueError):
        TokenPrices(bad, 1)
    with pytest.raises(ValueError):
        TokenPrices(1, bad)


@pytest.mark.parametrize("selected,counts", [
    (selection("deployment"), {"t1": {"a": 1}, "t2": {"a": 2}}),
    (selection("learning"), {"t1": {"a": 1}, "t2": {"a": 2}}),
    (selection(ids=()), {}),
    (selection(ids=("t1", "t1")), {"t1": {"a": 1}}),
    (selection(), {"t1": {"a": 1}}),
    (selection(), {"t1": {"a": 1}, "t2": {"a": 2}, "d1": {"a": 999}}),
    (selection(), {"t1": {"a": 1}, "t2": {"b": 2}}),
    (selection(digest=""), {"t1": {"a": 1}, "t2": {"a": 2}}),
])
def test_estimator_rejects_non_tuning_incomplete_or_inconsistent_data(selected, counts):
    from scripts.cost_policy import TokenPrices, TuningTokenCostEstimator
    with pytest.raises(ValueError):
        TuningTokenCostEstimator.fit(selected, counts, {"a": TokenPrices(1, 1)},
            token_provenance="toy", price_provenance="toy")


@pytest.mark.parametrize("selected", [selection("learning", ("d1",)),
    selection("deployment", ("t2",)), selection("deployment", ("d1",), "other"),
    selection("deployment", ()), selection("deployment", ("d1", "d1"))])
def test_estimator_rejects_invalid_deployment_selection(selected):
    with pytest.raises(ValueError):
        estimator().validate_deployment(selected)


def test_zero_tokens_prices_and_overflow():
    from scripts.cost_policy import TokenPrices, TuningTokenCostEstimator
    fitted = TuningTokenCostEstimator.fit(selection(), {"t1": {"a": 0}, "t2": {"a": 0}},
        {"a": TokenPrices(0, 0)}, token_provenance="toy", price_provenance="toy")
    assert fitted.estimate(0) == {"a": 0}
    with pytest.raises(ValueError):
        estimator().estimate({"a": 2})
    with pytest.raises(ValueError):
        estimator().estimate(10 ** 1000)


@pytest.mark.parametrize("token,price", [("", "toy"), ("toy", " "), (None, "toy")])
def test_provenance_required(token, price):
    from scripts.cost_policy import TokenPrices, TuningTokenCostEstimator
    with pytest.raises(ValueError):
        TuningTokenCostEstimator.fit(selection(), {"t1": {"a": 1}, "t2": {"a": 2}},
            {"a": TokenPrices(1, 1)}, token_provenance=token, price_provenance=price)


def policy(**overrides):
    from scripts.cost_policy import BinnedCostPolicy
    settings = dict(arm_catalog=("a", "b"), budget=4, query_count=2, bin_size=2,
                    lower_bound=1, upper_bound=1)
    settings.update(overrides)
    return BinnedCostPolicy(**settings)


def test_threshold_precedes_average_budget_and_selects_max_reward():
    budget = policy(budget=2)
    chosen = budget.select({"a": 0.9, "b": 0.8}, {"a": 1.5, "b": 0.1})
    assert chosen.arm_id == "a"  # Above average budget 1, but threshold-eligible.
    assert chosen.branch == "threshold"
    assert chosen.eligible_arms == ("a", "b")
    assert chosen.available_before == 2
    assert chosen.available_after == 0.5
    assert chosen.utilization_after == 0.75
    assert budget.report()["estimated_total"] == 1.5


def test_threshold_formula_intermediate_utilization_and_inclusive_boundary():
    budget = policy(budget=2, query_count=3, bin_size=3,
                    lower_bound=math.e, upper_bound=math.e)
    first = budget.select({"a": 1, "b": 0.8}, {"a": 1, "b": 0.5})
    assert first.arm_id == "a"  # z=0: phi=1; exact equality is eligible.
    assert first.utilization_after == 0.5
    second = budget.select({"a": 0.9, "b": 0.8}, {"a": 0.55, "b": 0.45})
    assert second.arm_id == "b"  # z=.5: phi=sqrt(e), thresholds ~.546 and ~.485.
    assert second.eligible_arms == ("b",)
    assert second.branch == "threshold"


def test_hard_affordability_intersects_threshold():
    budget = policy(budget=1, query_count=1)
    decision = budget.select({"a": 1, "b": 0.5}, {"a": 2, "b": 1})
    assert decision.arm_id == "b"
    assert decision.eligible_arms == ("b",)
    assert decision.available_after == 0
    assert budget.report()["status"] == "complete"


def test_fallback_uses_remaining_queries_including_current_and_max_reward():
    budget = policy()
    first = budget.select({"a": -0.1, "b": -0.2}, {"a": 2, "b": 1})
    assert first.branch == "fallback"
    assert first.arm_id == "a"  # Average allowance 4/2=2, not cheapest model.
    assert first.remaining_queries_in_bin == 2
    second = budget.select({"a": -0.1, "b": -0.2}, {"a": 2, "b": 1})
    assert second.remaining_queries_in_bin == 1
    assert second.available_after == 0


def test_insufficient_budget_does_not_select_currently_affordable_above_average():
    budget = policy()
    result = budget.select({"a": -0.1, "b": -0.2}, {"a": 3, "b": 4})
    assert result.status == "insufficient_budget"
    assert result.arm_id is None
    assert result.estimated_cost == 0
    assert budget.report()["estimated_total"] == 0
    assert budget.report()["successful_queries"] == 0
    with pytest.raises(RuntimeError):
        budget.select({"a": 1, "b": 1}, {"a": 0, "b": 0})


def test_bin_rollover_last_partial_bin_and_utilization_reset():
    budget = policy(budget=4, query_count=3, bin_size=2)
    a = budget.select({"a": 0, "b": -1}, {"a": 0.5, "b": 0.5})
    b = budget.select({"a": 0, "b": -1}, {"a": 0.5, "b": 0.5})
    c = budget.select({"a": 0, "b": -1}, {"a": 3, "b": 3})
    assert [a.bin_index, b.bin_index, c.bin_index] == [1, 1, 2]
    assert [a.available_before, b.available_before, c.available_before] == [2, 1.5, 3]
    assert c.remaining_queries_in_bin == 1
    assert c.utilization_before == 0
    assert c.utilization_after == 1.5  # Base B_bin=2, not carryover allocation=3.
    assert budget.report()["estimated_total"] == 4
    assert budget.report()["unallocated_budget"] == 0
    with pytest.raises(RuntimeError):
        budget.select({"a": 1, "b": 1}, {"a": 0, "b": 0})


def test_rollover_utilization_above_one_is_not_clipped():
    budget = policy(budget=4, query_count=4, bin_size=2, lower_bound=0.01)
    budget.select({"a": 1, "b": 0}, {"a": 0.25, "b": 0.25})
    budget.select({"a": 1, "b": 0}, {"a": 0.25, "b": 0.25})
    third = budget.select({"a": 1, "b": 0}, {"a": 3, "b": 3})
    assert third.utilization_after == 1.5
    fourth = budget.select({"a": 1, "b": 0}, {"a": 0.1, "b": 0.1})
    assert fourth.utilization_before == 1.5
    assert fourth.branch == "fallback"  # Clipped z=1 would admit threshold cost .1.


def test_zero_budget_free_arms_and_negative_predictions():
    budget = policy(budget=0)
    first = budget.select({"a": 0, "b": -0.2}, {"a": 0, "b": 0})
    assert first.branch == "threshold"
    assert first.eligible_arms == ("a",)
    second = budget.select({"a": -0.5, "b": -0.2}, {"a": 0, "b": 0})
    assert second.branch == "fallback"
    assert second.arm_id == "b"
    assert second.utilization_after == 0
    assert budget.report()["estimated_total"] == 0


def test_equal_rewards_follow_declared_catalog_order():
    budget = policy(arm_catalog=("b", "a"))
    first = budget.select({"a": 1, "b": 1}, {"a": 1, "b": 1})
    assert first.arm_id == "b"


@pytest.mark.parametrize("settings", [
    {"budget": -1}, {"budget": math.nan}, {"budget": math.inf}, {"budget": True},
    {"query_count": 0}, {"query_count": 1.5}, {"query_count": True},
    {"bin_size": 0}, {"bin_size": -1}, {"bin_size": 1.5},
    {"lower_bound": 0}, {"lower_bound": -1}, {"lower_bound": math.nan},
    {"upper_bound": math.inf}, {"lower_bound": 2, "upper_bound": 1},
    {"arm_catalog": ()}, {"arm_catalog": ("a", "a")}, {"arm_catalog": ("",)},
])
def test_invalid_policy_settings(settings):
    with pytest.raises(ValueError):
        policy(**settings)


@pytest.mark.parametrize("rewards,costs", [
    ({"a": math.nan, "b": 1}, {"a": 1, "b": 1}),
    ({"a": 1.01, "b": 1}, {"a": 1, "b": 1}),
    ({"a": -1.01, "b": 1}, {"a": 1, "b": 1}),
    ({"a": True, "b": 1}, {"a": 1, "b": 1}),
    ({"a": 1, "b": 1}, {"a": math.inf, "b": 1}),
    ({"a": 1, "b": 1}, {"a": -1, "b": 1}),
    ({"a": 1, "b": 1}, {"a": True, "b": 1}),
    ({"a": 1}, {"a": 1, "b": 1}),
    ({"a": 1, "b": 1}, {"a": 1, "b": 1, "c": 1}),
])
def test_invalid_decision_inputs_preserve_policy_state(rewards, costs):
    budget = policy()
    before = budget.report()
    with pytest.raises(ValueError):
        budget.select(rewards, costs)
    assert budget.report() == before
    assert budget.select({"a": 1, "b": 1}, {"a": 1, "b": 1}).query_index == 1


def test_decimal_budget_accounting_no_roundoff_overspend():
    budget = policy(budget=0.3, query_count=3, bin_size=1)
    decisions = [budget.select({"a": 1, "b": 1}, {"a": 0.1, "b": 0.1})
                 for _ in range(3)]
    assert all(d.status == "selected" for d in decisions)
    assert budget.report()["estimated_total"] == 0.3
    assert budget.report()["available_budget"] == 0


def test_fractional_bins_never_borrow_from_future_allocation():
    budget = policy(budget=1, query_count=3, bin_size=1)
    first = budget.select({"a": 1, "b": 0}, {"a": 0.34, "b": 0.33})
    assert first.arm_id == "b"
    assert first.estimated_cost <= first.available_before


def test_extreme_bounds_use_stable_log_thresholds():
    budget = policy(lower_bound=1e-300, upper_bound=1e300)
    assert budget.select({"a": 1, "b": 0}, {"a": 2, "b": 2}).arm_id == "a"
    assert budget.select({"a": 1, "b": 0}, {"a": 1, "b": 1}).arm_id == "a"


def test_decision_and_report_mutation_cannot_change_budget():
    from dataclasses import FrozenInstanceError
    budget = policy()
    rewards, costs = {"a": 1, "b": 0}, {"a": 1, "b": 1}
    first = budget.select(rewards, costs)
    costs["a"] = 999
    rewards["a"] = -999
    budget.report()["estimated_total"] = 999
    with pytest.raises(FrozenInstanceError):
        first.estimated_cost = 999
    assert budget.report()["estimated_total"] == 1


def test_actual_overspend_is_reported_and_never_changes_estimated_state():
    from scripts.cost_policy import ActualCostLedger
    budget = policy(budget=3)
    ledger = ActualCostLedger(budget)
    first = budget.select({"a": 1, "b": 0}, {"a": 1, "b": 1})
    before = budget.report()
    ledger.record(first, 4)
    assert budget.report() == before
    second = budget.select({"a": 1, "b": 0}, {"a": 1, "b": 1})
    ledger.record(second, 1)
    report = ledger.report()
    assert report["estimated_total"] == 2
    assert report["actual_total"] == 5
    assert report["actual_budget_excess_observed"] == 2
    assert report["actual_budget_violation_observed"] is True
    assert report["observed_estimation_error"] == 3
    assert report["missing_actual_queries"] == []
    assert report["actual_costs_complete"] is True


def test_missing_actual_observations_are_not_zero_imputed():
    from scripts.cost_policy import ActualCostLedger
    budget = policy()
    ledger = ActualCostLedger(budget)
    first = budget.select({"a": 1, "b": 0}, {"a": 1, "b": 1})
    second = budget.select({"a": 1, "b": 0}, {"a": 1, "b": 1})
    ledger.record(second, 0.5)
    report = ledger.report()
    assert report["actual_total"] is None
    assert report["actual_total_observed"] == 0.5
    assert report["missing_actual_queries"] == [first.query_index]
    assert report["observed_estimation_error"] == -0.5
    assert report["actual_costs_complete"] is False


@pytest.mark.parametrize("bad", [-1, math.nan, math.inf, True, "1", None])
def test_actual_ledger_invalid_cost_does_not_mutate(bad):
    from scripts.cost_policy import ActualCostLedger
    budget = policy()
    ledger = ActualCostLedger(budget)
    chosen = budget.select({"a": 1, "b": 0}, {"a": 1, "b": 1})
    before = ledger.report()
    with pytest.raises(ValueError):
        ledger.record(chosen, bad)
    assert ledger.report() == before


def test_actual_ledger_rejects_duplicate_forged_and_foreign_decisions():
    from scripts.cost_policy import ActualCostLedger
    budget = policy()
    ledger = ActualCostLedger(budget)
    chosen = budget.select({"a": 1, "b": 0}, {"a": 1, "b": 1})
    with pytest.raises(ValueError):
        ledger.record(replace(chosen, estimated_cost=0), 0.5)
    foreign = policy().select({"a": 1, "b": 0}, {"a": 1, "b": 1})
    with pytest.raises(ValueError):
        ledger.record(foreign, 0.5)
    ledger.record(chosen, 0.5)
    with pytest.raises(ValueError):
        ledger.record(chosen, 0.5)
    assert ledger.report()["actual_total"] == 0.5


def test_actual_ledger_rejects_failure_and_overflow_atomically():
    from scripts.cost_policy import ActualCostLedger
    insufficient = policy(budget=0)
    failure = insufficient.select({"a": 1, "b": 0}, {"a": 1, "b": 1})
    with pytest.raises(ValueError):
        ActualCostLedger(insufficient).record(failure, 1)
    budget = policy()
    ledger = ActualCostLedger(budget)
    first = budget.select({"a": 1, "b": 0}, {"a": 1, "b": 1})
    second = budget.select({"a": 1, "b": 0}, {"a": 1, "b": 1})
    ledger.record(first, 1e308)
    before = ledger.report()
    with pytest.raises(ValueError):
        ledger.record(second, 1e308)
    assert ledger.report() == before


def test_actual_cost_mutation_cannot_change_actions_or_estimates():
    from scripts.cost_policy import ActualCostLedger
    traces = []
    for actual in (0, 10000):
        budget = policy()
        ledger = ActualCostLedger(budget)
        trace = []
        for _ in range(2):
            chosen = budget.select({"a": 0.9, "b": 0.8}, {"a": 1, "b": 0.5})
            ledger.record(chosen, actual)
            ledger.report()["actual_total"] = -100
            trace.append(chosen)
        traces.append(trace)
    assert traces[0] == traces[1]
