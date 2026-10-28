"""Learning-only execution. Oracle rewards and costs never cross into a policy."""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from scripts.bandit_contracts import ContextTable, OnlinePolicy
from scripts.data_gate import Feedback, OfflineBandit
from scripts.splits import SplitSelection


@dataclass(frozen=True)
class RoundMetrics:
    round: int
    query_id: str
    arm_id: str
    reward: float
    actual_cost: float
    oracle_reward: float
    empirical_oracle_regret: float
    cumulative_reward: float
    cumulative_actual_cost: float
    cumulative_empirical_oracle_regret: float


@dataclass(frozen=True)
class RunResult:
    trace: tuple[RoundMetrics, ...]
    summary: dict


def _fingerprint(value) -> str:
    payload = json.dumps(value, ensure_ascii=False, allow_nan=False,
                         sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class _OfflineEvaluator:
    """Owned by the runner. Receives feedback only after the action is selected."""

    def __init__(self, outcomes: pd.DataFrame, catalog: tuple[str, ...]):
        self._outcomes = outcomes.set_index(["query_id", "arm_id"]).copy()
        self.trace: list[RoundMetrics] = []
        self.counts = dict.fromkeys(catalog, 0)
        self.reward = self.cost = self.regret = 0.0

    def record(self, feedback: Feedback) -> None:
        oracle = float(self._outcomes.loc[feedback.query_id, "reward"].max())
        regret = oracle - feedback.reward
        self.reward += feedback.reward
        self.cost += feedback.actual_cost
        self.regret += regret
        if not all(math.isfinite(v) for v in (self.reward, self.cost, self.regret)):
            raise ValueError("Accumulated metrics are nonfinite")
        self.counts[feedback.arm_id] += 1
        self.trace.append(RoundMetrics(len(self.trace) + 1, feedback.query_id,
            feedback.arm_id, feedback.reward, feedback.actual_cost, oracle, regret,
            self.reward, self.cost, self.regret))

    def metrics(self) -> dict:
        return {"rounds": len(self.trace), "total_reward": self.reward,
                "mean_reward": self.reward / len(self.trace), "total_actual_cost": self.cost,
                "cumulative_empirical_oracle_regret": self.regret,
                "action_counts": self.counts.copy()}


def run_learning(policy: OnlinePolicy, queries: pd.DataFrame, outcomes: pd.DataFrame,
                 contexts: ContextTable, selection: SplitSelection, *, seed: int) -> RunResult:
    """One pass through a learning selection; no budget and no held-out updates."""
    ids = selection.query_ids
    if selection.split != "learning":
        raise ValueError("Learning runner accepts only the learning split")
    if (not ids or not all(isinstance(q, str) and q.strip() for q in ids)
            or len(set(ids)) != len(ids)):
        raise ValueError("Learning query IDs must be nonempty and unique")
    if type(seed) is not int:
        raise ValueError("Run seed must be an integer")
    # Validate the complete catalog before subsetting, so a missing arm cannot
    # silently shrink a subset's catalog. Copies isolate policy side effects.
    environment = OfflineBandit(queries, outcomes)
    if not set(ids) <= set(queries.query_id):
        raise ValueError("Unknown learning query IDs")
    try:
        selected_contexts = ContextTable({q: contexts.get(q) for q in ids}, label=contexts.label)
    except KeyError as exc:
        raise ValueError(f"Missing context for query {exc.args[0]}") from exc
    selected_outcomes = outcomes.loc[outcomes.query_id.isin(ids),
        ["query_id", "arm_id", "reward", "actual_cost"]].copy()
    selected_outcomes = selected_outcomes.astype({"reward": float, "actual_cost": float})
    evaluator = _OfflineEvaluator(selected_outcomes, environment.arm_catalog)
    selected_queries = queries.set_index("query_id").loc[list(ids), ["query", "task"]].reset_index()
    # Fingerprints are captured before invoking any policy code.
    data_hashes = {"queries_sha256": _fingerprint(selected_queries.to_dict("records")),
                   "outcomes_sha256": _fingerprint(selected_outcomes.sort_values(
                       ["query_id", "arm_id"]).to_dict("records"))}
    policy.reset(seed)
    for query_id in ids:
        arm = policy.select(selected_contexts.get(query_id))
        if not isinstance(arm, str) or arm not in environment.arm_catalog:
            raise ValueError("Policy selected an unknown arm")
        feedback = environment.step(query_id, arm)
        policy.observe(feedback.arm_id, selected_contexts.get(query_id), feedback.reward)
        evaluator.record(feedback)
    split_metadata = asdict(selection)
    split_metadata["query_ids"] = list(ids)
    summary = {"schema_version": 1, "phase": "learning", "seed": seed,
               "policy": f"{type(policy).__module__}.{type(policy).__qualname__}",
               "arm_catalog": list(environment.arm_catalog), "split": split_metadata,
               "context": {"label": selected_contexts.label,
                           "dimension": selected_contexts.dimension,
                           "sha256": selected_contexts.fingerprint(ids)},
               "data": data_hashes, "metrics": evaluator.metrics(),
               "metric_interpretation": "empirical_oracle_regret_on_stored_outcomes"}
    return RunResult(tuple(evaluator.trace), summary)


def write_run(result: RunResult, output_dir: str | Path) -> None:
    """Write a new run directory only; reject nonfinite values before filesystem writes."""
    summary = json.dumps(result.summary, ensure_ascii=False, allow_nan=False, indent=2)
    rows = [asdict(row) for row in result.trace]
    json.dumps(rows, allow_nan=False)
    trace = pd.DataFrame(rows).to_csv(index=False)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    (output / "summary.json").write_text(summary + "\n", encoding="utf-8")
    (output / "trace.csv").write_text(trace, encoding="utf-8", newline="")
