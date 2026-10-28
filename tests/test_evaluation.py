import json
from dataclasses import asdict, replace

import numpy as np
import pandas as pd
import pytest

from scripts.bandit_contracts import ContextTable
from scripts.splits import SplitSelection


def fixture_inputs():
    q = pd.DataFrame({"query_id": ["01", "02"], "query": ["first", "second"], "task": "toy"})
    o = pd.DataFrame({"query_id": ["01", "01", "02", "02"], "arm_id": ["a", "b", "a", "b"],
                      "reward": [0.25, 1.0, 1.0, 0.0], "actual_cost": [0.1, 0.3, 0.2, 0.4]})
    contexts = ContextTable({"01": [1, 0], "02": [0, 1]}, label="synthetic-identity")
    selection = SplitSelection(("01", "02"), "learning", "toy-unpinned", 42, 1, 10, 2)
    return q, o, contexts, selection


class FixedSpy:
    def reset(self, seed):
        self.seed = seed
        self.events = ["reset"]
        self.updates = []

    def select(self, context):
        assert isinstance(context, np.ndarray) and context.shape == (2,)
        self.events.append("select")
        context[:] = 99  # Must not change the original context used for observe.
        return "a"

    def observe(self, arm_id, context, reward):
        self.events.append("observe")
        self.updates.append((arm_id, context.tolist(), reward))

    def predict_rewards(self, context):
        raise AssertionError("Runner must not ask for oracle or prediction access")


def test_analytic_metrics_and_selected_reward_only_lifecycle():
    from scripts.evaluation import run_learning
    policy = FixedSpy()
    result = run_learning(policy, *fixture_inputs(), seed=9)
    assert policy.seed == 9
    assert policy.events == ["reset", "select", "observe", "select", "observe"]
    assert policy.updates == [("a", [1, 0], 0.25), ("a", [0, 1], 1)]
    metrics = result.summary["metrics"]
    assert metrics["total_reward"] == 1.25
    assert metrics["mean_reward"] == 0.625
    assert metrics["total_actual_cost"] == pytest.approx(0.3)
    assert metrics["cumulative_empirical_oracle_regret"] == 0.75
    assert metrics["action_counts"] == {"a": 2, "b": 0}
    assert [r.empirical_oracle_regret for r in result.trace] == [0.75, 0]
    assert [r.cumulative_reward for r in result.trace] == [0.25, 1.25]
    assert result.summary["context"]["label"] == "synthetic-identity"
    assert result.summary["split"]["query_ids"] == ["01", "02"]


def test_unchosen_outcomes_do_not_change_policy_inputs_or_actions():
    from scripts.evaluation import run_learning
    first, second = FixedSpy(), FixedSpy()
    q, o, ctx, selection = fixture_inputs()
    original = run_learning(first, q, o, ctx, selection, seed=9)
    o.loc[o.arm_id == "b", "reward"] = 0.5
    o.loc[o.arm_id == "b", "actual_cost"] = 10000
    changed = run_learning(second, q, o, ctx, selection, seed=9)
    assert first.updates == second.updates
    assert [r.arm_id for r in original.trace] == [r.arm_id for r in changed.trace]
    assert changed.summary["metrics"]["cumulative_empirical_oracle_regret"] == 0.25
    assert changed.summary["metrics"]["total_actual_cost"] == pytest.approx(0.3)


@pytest.mark.parametrize("split,ids", [("tuning", ("01",)), ("deployment", ("01",)),
    ("learning", ()), ("learning", ("01", "01")), ("learning", ("absent",))])
def test_invalid_learning_selection_fails_before_policy_reset(split, ids):
    from scripts.evaluation import run_learning
    q, o, ctx, selection = fixture_inputs()
    policy = FixedSpy()
    with pytest.raises(ValueError):
        run_learning(policy, q, o, ctx, replace(selection, split=split, query_ids=ids), seed=9)
    assert not hasattr(policy, "events")


def test_missing_context_fails_before_policy_reset():
    from scripts.evaluation import run_learning
    q, o, _, selection = fixture_inputs()
    policy = FixedSpy()
    with pytest.raises(ValueError, match="context"):
        run_learning(policy, q, o, ContextTable({"01": [1, 0]}, label="partial"), selection, seed=9)
    assert not hasattr(policy, "events")


def test_invalid_action_never_updates_policy():
    from scripts.evaluation import run_learning

    class Invalid(FixedSpy):
        def select(self, context):
            return "unknown"

    policy = Invalid()
    with pytest.raises(ValueError, match="arm"):
        run_learning(policy, *fixture_inputs(), seed=9)
    assert policy.updates == []


def test_seeded_replay_and_log_roundtrip(tmp_path):
    from scripts.evaluation import run_learning, write_run

    class Seeded(FixedSpy):
        def reset(self, seed):
            super().reset(seed)
            self.rng = np.random.default_rng(seed)

        def select(self, context):
            return self.rng.choice(["a", "b"])

    first = run_learning(Seeded(), *fixture_inputs(), seed=7)
    second = run_learning(Seeded(), *fixture_inputs(), seed=7)
    assert first == second
    target = tmp_path / "run"
    write_run(first, target)
    saved = json.loads((target / "summary.json").read_text(encoding="utf-8"))
    assert saved == first.summary
    trace = pd.read_csv(target / "trace.csv", dtype={"query_id": str})
    assert trace.query_id.tolist() == ["01", "02"]
    assert trace.reward.sum() == first.summary["metrics"]["total_reward"]
    assert trace.columns.tolist() == list(asdict(first.trace[0]))
    before = (target / "summary.json").read_bytes()
    with pytest.raises(FileExistsError):
        write_run(second, target)
    assert (target / "summary.json").read_bytes() == before


def test_nonfinite_summary_is_rejected_before_writing(tmp_path):
    from scripts.evaluation import run_learning, write_run
    result = run_learning(FixedSpy(), *fixture_inputs(), seed=9)
    result.summary["metrics"]["total_reward"] = np.nan
    with pytest.raises(ValueError):
        write_run(result, tmp_path / "bad")
    assert not (tmp_path / "bad").exists()
