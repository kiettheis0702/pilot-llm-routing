"""Standalone deployment cost control; no outcomes, tokenizers or router updates.

Published PILOT Sections 2.2.3, 3.2 and Appendix C.2 Algorithm 3:
https://aclanthology.org/2025.findings-emnlp.1301.pdf
Interpretations and the affordability deviation are in docs/cost-policy.md.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from numbers import Integral, Real

from scripts.splits import SplitSelection


def _finite(value, name: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite number")
    try:
        result = float(value)
    except (OverflowError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(result) or (minimum is not None and result < minimum):
        raise ValueError(f"{name} must be finite and at least {minimum}")
    return result


def _count(value, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer at least {minimum}")
    _finite(value, name)
    return int(value)


def _text(value, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _catalog(values: Sequence[str]) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise ValueError("Arm catalog must be a sequence of identifiers")
    result = tuple(values)
    if not result or any(not isinstance(a, str) or not a.strip() for a in result):
        raise ValueError("Arm catalog must contain nonempty string IDs")
    if len(set(result)) != len(result):
        raise ValueError("Arm catalog must be unique")
    return result


def _coverage(values, keys: Sequence[str], name: str) -> None:
    if not isinstance(values, Mapping) or set(values) != set(keys):
        raise ValueError(f"{name} must cover exactly the declared IDs")


def _selection_ids(selection: SplitSelection, phase: str) -> tuple[str, ...]:
    if not isinstance(selection, SplitSelection) or selection.split != phase:
        raise ValueError(f"Expected a {phase} split selection")
    _text(selection.manifest_sha256, "Split manifest fingerprint")
    return _catalog(selection.query_ids)


def _fraction(value: float) -> Fraction:
    # Preserve the canonical decimal representation supplied by the caller.
    # Fractions keep repeated subtraction and B/N allocations exact.
    return Fraction(str(value))


def _reported(value: Fraction, name: str) -> float:
    result = _finite(value, name)
    if value and not result:
        raise ValueError(f"{name} is too small for a nonzero float report")
    return result


@dataclass(frozen=True)
class TokenPrices:
    """Prices in the caller's currency per one million input/output tokens."""

    input_per_million: float
    output_per_million: float

    def __post_init__(self):
        for name in ("input_per_million", "output_per_million"):
            object.__setattr__(self, name, _finite(getattr(self, name), name, minimum=0))


class TuningTokenCostEstimator:
    """Fit only explicit tuning counts; deployment outputs are never an input.

    Use fit(), then validate_deployment() before a held-out deployment run.
    SplitSelection declarations must originate from the verified split loader.
    """

    @classmethod
    def fit(cls, selection: SplitSelection,
            output_counts_by_query: Mapping[str, Mapping[str, int]],
            prices: Mapping[str, TokenPrices], *, token_provenance: str,
            price_provenance: str) -> TuningTokenCostEstimator:
        ids = _selection_ids(selection, "tuning")
        token_provenance = _text(token_provenance, "Token provenance")
        price_provenance = _text(price_provenance, "Price provenance")
        if not isinstance(prices, Mapping):
            raise ValueError("Prices must be an arm mapping")
        arms = _catalog(tuple(prices))
        if any(not isinstance(prices[a], TokenPrices) for a in arms):
            raise ValueError("Every arm must have explicit TokenPrices")
        _coverage(output_counts_by_query, ids, "Tuning output counts")
        totals = dict.fromkeys(arms, 0)
        for query in ids:
            counts = output_counts_by_query[query]
            _coverage(counts, arms, "Per-query output counts")
            for arm in arms:
                totals[arm] += _count(counts[arm], "Output token count")
        means = {a: Fraction(totals[a], len(ids)) for a in arms}
        for mean in means.values():
            _reported(mean, "Mean output token count")
        result = cls()
        result._ids = ids
        result._manifest = selection.manifest_sha256
        result._arms = arms
        result._means = means
        result._prices = dict(prices)
        result._token_provenance = token_provenance
        result._price_provenance = price_provenance
        return result

    @property
    def mean_output_tokens(self) -> dict[str, float]:
        return {a: float(v) for a, v in self._means.items()}

    def validate_deployment(self, selection: SplitSelection) -> None:
        ids = _selection_ids(selection, "deployment")
        if selection.manifest_sha256 != self._manifest:
            raise ValueError("Deployment and tuning split manifests differ")
        if set(ids).intersection(self._ids):
            raise ValueError("Deployment queries overlap tuning queries")

    def estimate(self, input_tokens: int | Mapping[str, int]) -> dict[str, float]:
        """Use query-only input counts; a mapping supports different tokenizers."""
        if isinstance(input_tokens, Mapping):
            _coverage(input_tokens, self._arms, "Input token counts")
            counts = {a: _count(input_tokens[a], "Input token count") for a in self._arms}
        else:
            counts = dict.fromkeys(self._arms, _count(input_tokens, "Input token count"))
        return {a: _reported((counts[a] * _fraction(self._prices[a].input_per_million)
                   + self._means[a] * _fraction(self._prices[a].output_per_million))
                   / 1_000_000, "Estimated cost") for a in self._arms}

    def metadata(self) -> dict:
        return {"tuning_query_ids": list(self._ids), "split_manifest_sha256": self._manifest,
                "token_provenance": self._token_provenance,
                "price_provenance": self._price_provenance,
                "price_unit": "caller_currency_per_million_tokens",
                "mean_output_tokens": self.mean_output_tokens,
                "prices": {a: {"input_per_million": p.input_per_million,
                               "output_per_million": p.output_per_million}
                           for a, p in self._prices.items()}}


@dataclass(frozen=True)
class BudgetDecision:
    """One atomic estimated reservation or a terminal insufficient-budget result."""

    status: str
    query_index: int
    bin_index: int
    remaining_queries_in_bin: int
    arm_id: str | None
    predicted_reward: float | None
    estimated_cost: float
    branch: str
    eligible_arms: tuple[str, ...]
    available_before: float
    available_after: float
    cumulative_estimated_cost: float
    utilization_before: float
    utilization_after: float


class BinnedCostPolicy:
    """Algorithm 3 with an explicit hard estimated-affordability intersection.

    select() consumes one query on success. Failure is terminal, as in the paper.
    Scores are fixed-router predictions, not exploration/UCB bonuses. The default
    cosine domain is bounded; the explicit linear domain keeps raw ridge means.
    Bounds are supplied assumptions on positive reward/cost ratios; zero or
    negative rewards and zero costs have defined behavior but not that guarantee.
    """

    def __init__(self, *, arm_catalog: Sequence[str], budget: float, query_count: int,
                 bin_size: int, lower_bound: float, upper_bound: float,
                 score_domain: str = 'cosine'):
        self._arms = _catalog(arm_catalog)
        if score_domain not in ('cosine', 'linear'):
            raise ValueError('Score domain must be cosine or linear')
        self._score_domain = score_domain
        self._budget = _fraction(_finite(budget, "Budget", minimum=0))
        self._query_count = _count(query_count, "Query count", minimum=1)
        self._bin_size = _count(bin_size, "Bin size", minimum=1)
        self._lb = _finite(lower_bound, "Lower bound", minimum=0)
        self._ub = _finite(upper_bound, "Upper bound", minimum=0)
        if self._lb == 0 or self._lb > self._ub:
            raise ValueError("Bounds must satisfy 0 < LB <= UB")
        self._bins = (self._query_count + self._bin_size - 1) // self._bin_size
        self._base = self._budget / self._bins
        _reported(self._base, "Base bin allocation")
        self._allocated = self._spent = self._bin_spent = Fraction(0)
        self._bin_index = 0
        self._status = "ready"
        self._decisions: list[BudgetDecision] = []
        self._decision_costs: list[Fraction] = []

    @property
    def decisions(self) -> tuple[BudgetDecision, ...]:
        return tuple(self._decisions)

    def select(self, predicted_rewards: Mapping[str, float],
               estimated_costs: Mapping[str, float], *,
               allowed_arms: Sequence[str] | None = None,
               tie_break_order: Sequence[str] | None = None) -> BudgetDecision:
        if self._status != "ready":
            raise RuntimeError(f"Cost policy is terminal: {self._status}")
        _coverage(predicted_rewards, self._arms, "Predicted rewards")
        _coverage(estimated_costs, self._arms, "Estimated costs")
        rewards = {a: _finite(predicted_rewards[a], "Predicted reward") for a in self._arms}
        if self._score_domain == 'cosine' and any(abs(r) > 1 for r in rewards.values()):
            raise ValueError("Predicted cosine rewards must lie in [-1, 1]")
        allowed = self._arms if allowed_arms is None else _catalog(allowed_arms)
        order = self._arms if tie_break_order is None else _catalog(tie_break_order)
        if not set(allowed) <= set(self._arms) or set(order) != set(self._arms):
            raise ValueError('Allowed arms/order must match the model catalog')
        costs = {a: _fraction(_finite(estimated_costs[a], "Estimated cost", minimum=0))
                 for a in self._arms}
        completed = len(self._decisions)
        bin_index = completed // self._bin_size + 1
        allocated = self._base * bin_index
        available = allocated - self._spent
        bin_spent = self._bin_spent if bin_index == self._bin_index else Fraction(0)
        remaining = min(self._bin_size - completed % self._bin_size,
                        self._query_count - completed)
        # Algorithm 3 uses B_bin, not B_left or the allocation plus rollover.
        z = _reported(bin_spent / self._base, "Utilization") if self._base else 0.0
        log_phi = _finite(math.log(self._lb) - 1
                         + z * (math.log(self._ub) - math.log(self._lb) + 1),
                         "Log threshold denominator")

        def threshold_eligible(arm: str) -> bool:
            reward, cost = rewards[arm], costs[arm]
            if cost > available:  # Explicit safety deviation from Algorithm 3.
                return False
            if reward < 0:
                return False
            if cost == 0:
                return True  # Literal 0 <= reward / phi, including reward=0.
            if reward == 0:
                return False
            return math.log(reward) - math.log(float(cost)) >= log_phi

        eligible = tuple(a for a in self._arms if a in allowed and threshold_eligible(a))
        branch = "threshold"
        if not eligible:
            branch = "fallback"
            eligible = tuple(a for a in self._arms if a in allowed and costs[a] <= available / remaining)
        arm = max((a for a in order if a in eligible), key=rewards.__getitem__) if eligible else None
        cost = costs[arm] if arm is not None else Fraction(0)
        new_spent = self._spent + cost
        new_bin_spent = bin_spent + cost
        new_z = (_reported(new_bin_spent / self._base, "Utilization")
                 if self._base else 0.0)
        decision = BudgetDecision(
            status="selected" if arm is not None else "insufficient_budget",
            query_index=completed + 1, bin_index=bin_index,
            remaining_queries_in_bin=remaining, arm_id=arm,
            predicted_reward=rewards[arm] if arm is not None else None,
            estimated_cost=_reported(cost, "Selected estimated cost"),
            branch=branch if arm is not None else "insufficient_budget",
            eligible_arms=eligible,
            available_before=_reported(available, "Available budget"),
            available_after=_reported(available - cost, "Remaining budget"),
            cumulative_estimated_cost=_reported(new_spent, "Estimated total"),
            utilization_before=z, utilization_after=new_z)
        # All validation/arithmetic precedes this commit, including bin opening.
        self._allocated, self._bin_index = allocated, bin_index
        self._spent, self._bin_spent = new_spent, new_bin_spent
        if arm is None:
            self._status = "insufficient_budget"
        else:
            self._decisions.append(decision)
            self._decision_costs.append(cost)
            if completed + 1 == self._query_count:
                self._status = "complete"
        return decision

    def report(self) -> dict:
        return {"status": self._status, "budget": float(self._budget),
                "query_count": self._query_count, "bin_size": self._bin_size,
                "bin_count": self._bins, "base_bin_budget": float(self._base),
                "lower_bound": self._lb, "upper_bound": self._ub,
                "successful_queries": len(self._decisions),
                "estimated_total": float(self._spent),
                "allocated_budget": float(self._allocated),
                "available_budget": float(self._allocated - self._spent),
                "unallocated_budget": float(self._budget - self._allocated),
                "remaining_total_budget": float(self._budget - self._spent),
                "utilization": float(self._bin_spent / self._base) if self._base else 0.0,
                "estimated_budget_violation": self._spent > self._budget,
                "arm_catalog": list(self._arms), "score_domain": self._score_domain}


class ActualCostLedger:
    """Post-choice accounting only; recording never changes policy state.

    Bind one ledger to a policy and record each issued decision at most once.
    Missing observations remain explicitly incomplete rather than costing zero.
    """

    def __init__(self, policy: BinnedCostPolicy):
        if not isinstance(policy, BinnedCostPolicy):
            raise ValueError("Actual ledger requires a BinnedCostPolicy")
        self._policy = policy
        self._actuals: dict[int, Fraction] = {}
        self._total = Fraction(0)

    def record(self, decision: BudgetDecision, actual_cost: float) -> None:
        if (not isinstance(decision, BudgetDecision) or decision.status != "selected"
                or not 1 <= decision.query_index <= len(self._policy._decisions)
                or self._policy._decisions[decision.query_index - 1] is not decision):
            raise ValueError("Actual cost must reference a decision issued by this policy")
        if decision.query_index in self._actuals:
            raise ValueError("Actual cost is already recorded for this query")
        cost = _fraction(_finite(actual_cost, "Actual cost", minimum=0))
        new_total = self._total + cost
        _reported(new_total, "Actual total")
        self._actuals[decision.query_index] = cost
        self._total = new_total

    def report(self) -> dict:
        report = self._policy.report()
        missing = [d.query_index for d in self._policy._decisions
                   if d.query_index not in self._actuals]
        matched_estimate = sum((self._policy._decision_costs[i - 1]
                               for i in self._actuals), Fraction(0))
        excess = max(Fraction(0), self._total - self._policy._budget)
        report.update({"actual_observations": len(self._actuals),
                       "actual_costs_complete": not missing,
                       "missing_actual_queries": missing,
                       "actual_total_observed": float(self._total),
                       "actual_total": None if missing else float(self._total),
                       "actual_budget_violation_observed": excess > 0,
                       "actual_budget_excess_observed": float(excess),
                       "observed_estimation_error": float(self._total - matched_estimate),
                       "actual_trace": [{"query_index": i,
                           "arm_id": self._policy._decisions[i - 1].arm_id,
                           "estimated_cost": float(self._policy._decision_costs[i - 1]),
                           "actual_cost": float(cost)}
                           for i, cost in sorted(self._actuals.items())]})
        return report
