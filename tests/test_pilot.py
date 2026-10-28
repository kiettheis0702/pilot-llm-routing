import hashlib
import json

import numpy as np
import pandas as pd
import pytest


def policy(**changes):
    from scripts.pilot import PILOT
    args = dict(arms=("a", "b"), dimension=2, priors={"a": [2., 0.], "b": [0., 1.]},
                lambdas={"a": 2., "b": 1.}, provenance="synthetic-hand-constructed", alpha=0.5)
    args.update(changes)
    return PILOT(**args)


def test_prior_initialization_cosine_and_chosen_update_reset():
    p = policy()
    np.testing.assert_array_equal(p.A, [2*np.eye(2), np.eye(2)])
    np.testing.assert_array_equal(p.b, [[4, 0], [0, 1]])
    assert p.predict_rewards([3, 0]) == {"a": 1., "b": 0.}
    assert p.scores([3, 0]) == pytest.approx({"a": 1+0.5/np.sqrt(2), "b": 0.5})
    assert p.select([3, 0]) == "a"
    p.observe("a", [3, 0], 1)
    np.testing.assert_array_equal(p.A, [[[3, 0], [0, 2]], np.eye(2)])
    np.testing.assert_array_equal(p.b, [[5, 0], [0, 1]])
    p.reset(9)
    assert p.rounds == 0
    np.testing.assert_array_equal(p.b, [[4, 0], [0, 1]])


def test_scale_invariance_and_copied_prior():
    priors = {"a": [2., 0.], "b": [0., 1.]}
    p, q = policy(priors=priors), policy()
    priors["a"][0] = -100
    for scale in [1e-300, 1., 1e300]:
        assert p.scores(np.array([3., 4.])*scale) == pytest.approx(q.scores([3, 4]))
    p.observe("a", [3e200, 4e200], 0.7)
    q.observe("a", [3, 4], 0.7)
    np.testing.assert_allclose(p.A, q.A)
    np.testing.assert_allclose(p.b, q.b)


def test_zero_arm_convention_and_ties():
    p = policy(priors={"a": [0, 0], "b": [0, 0]}, lambdas={"a": 1, "b": 1})
    assert p.predict_rewards([1, 0]) == {"a": 0., "b": 0.}
    assert p.select([1, 0]) == "a"


@pytest.mark.parametrize("changes", [
    {"arms": ("a", "a")}, {"dimension": 0}, {"dimension": True},
    {"priors": {"a": [1, 0]}}, {"priors": {"a": [1], "b": [1]}},
    {"priors": {"a": [np.nan, 0], "b": [0, 1]}},
    {"lambdas": {"a": 0, "b": 1}}, {"lambdas": {"a": np.inf, "b": 1}},
    {"lambdas": {"a": 1}}, {"provenance": " "}, {"alpha": -1},
    {"priors": {"a": [1e308, 0], "b": [0, 1]}, "lambdas": {"a": 10, "b": 1}},
])
def test_bad_initialization(changes):
    with pytest.raises(ValueError):
        policy(**changes)


@pytest.mark.parametrize("arm,x,reward", [("a", [0, 0], 1), ("a", [np.nan, 1], 1),
    ("a", [1], 1), ("unknown", [1, 0], 1), ("a", [1, 0], np.inf), ("a", [1, 0], -1)])
def test_rejected_feedback_is_atomic(arm, x, reward):
    p = policy()
    before = p.A, p.b
    with pytest.raises(ValueError):
        p.observe(arm, x, reward)
    np.testing.assert_array_equal(p.A, before[0])
    np.testing.assert_array_equal(p.b, before[1])
    assert p.rounds == 0


def test_normalized_update_rejects_lost_spd():
    p = policy(lambdas={"a": 1e-20, "b": 1})
    before = p.A, p.b
    with pytest.raises(ValueError):
        p.observe("a", [1, 1], 1)
    np.testing.assert_array_equal(p.A, before[0])
    np.testing.assert_array_equal(p.b, before[1])


def test_checkpoint_resume_and_overwrite(tmp_path):
    from scripts.pilot import PILOT
    p = policy()
    p.reset(19)
    p.observe("a", [1, 1], 0.25)
    path = tmp_path / "pilot.json"
    p.save_checkpoint(path)
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        p.save_checkpoint(path)
    assert path.read_bytes() == before
    q = PILOT.load_checkpoint(path)
    assert q.rounds == p.rounds == 1
    for x, r in [([2, -1], 0.4), ([0, 3], 1), ([-1, -1], 0)]:
        assert p.scores(x) == q.scores(x)
        assert p.select(x) == q.select(x)
        arm = p.select(x)
        p.observe(arm, x, r)
        q.observe(arm, x, r)
    np.testing.assert_array_equal(p.A, q.A)
    np.testing.assert_array_equal(p.b, q.b)
    q.reset(19)
    np.testing.assert_array_equal(q.A, policy().A)


def resign(document):
    document["sha256"] = hashlib.sha256(json.dumps(document["payload"], sort_keys=True,
        separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()).hexdigest()


@pytest.mark.parametrize("damage", ["checksum", "version", "extra", "missing", "shape", "asymmetric",
    "singular", "negative", "rounds", "seed", "catalog", "lambda", "nonfinite"])
def test_checkpoint_validation(tmp_path, damage):
    from scripts.pilot import PILOT
    path = tmp_path / "pilot.json"
    policy().save_checkpoint(path)
    document = json.loads(path.read_text())
    payload = document["payload"]
    if damage == "checksum":
        payload["b"][0][0] += 1
    elif damage == "version": payload["version"] = 2
    elif damage == "extra": payload["extra"] = 1
    elif damage == "missing": del payload["provenance"]
    elif damage == "shape": payload["b"] = [[1]]
    elif damage == "asymmetric": payload["A"][0][0][1] = 0.5
    elif damage == "singular": payload["A"][0] = [[1, 1], [1, 1]]
    elif damage == "negative": payload["A"][0][0][0] = -1
    elif damage == "rounds": payload["rounds"] = -1
    elif damage == "seed": payload["seed"] = True
    elif damage == "catalog": payload["arms"] = ["a", "a"]
    elif damage == "lambda": payload["lambdas"]["a"] = 0
    else: payload["b"][0][0] = float("nan")
    if damage not in {"checksum", "nonfinite"}: resign(document)
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError): PILOT.load_checkpoint(path)


def test_learning_evaluator_and_model_checkpoint(tmp_path):
    from scripts.bandit_contracts import ContextTable
    from scripts.evaluation import run_learning
    from scripts.pilot import PILOT
    from scripts.splits import SplitSelection
    queries = pd.DataFrame({"query_id": ["01", "02"], "query": ["first", "second"], "task": "toy"})
    outcomes = pd.DataFrame({"query_id": ["01", "01", "02", "02"], "arm_id": ["a", "b", "a", "b"],
        "reward": [1., 0., 0., 1.], "actual_cost": [0.1, 0.2, 0.1, 0.2]})
    context = ContextTable({"01": [3, 0], "02": [0, 4]}, label="synthetic-toy")
    selection = SplitSelection(("01", "02"), "learning", "synthetic-not-a-manifest", 42, 1, 10, 2)
    p = policy()
    result = run_learning(p, queries, outcomes, context, selection, seed=7)
    assert [r.arm_id for r in result.trace] == ["a", "b"]
    assert result.summary["metrics"]["total_reward"] == 2
    path = tmp_path / "learned.json"
    p.save_checkpoint(path)
    loaded = PILOT.load_checkpoint(path)
    assert loaded.rounds == 2
    assert loaded.scores([1, 1]) == p.scores([1, 1])
