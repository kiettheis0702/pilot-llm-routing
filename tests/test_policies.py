import numpy as np
import pytest


def test_ridge_ucb_and_chosen_only_update():
    from scripts.policies import LinearUCB
    p = LinearUCB(("a", "b"), 2, alpha=1, ridge=1)
    p.reset(7)
    before_a, before_b = p.A.copy(), p.b.copy()
    p.observe("a", np.array([1., 0.]), 1.)
    np.testing.assert_array_equal(p.A[0], [[2, 0], [0, 1]])
    np.testing.assert_array_equal(p.b[0], [1, 0])
    np.testing.assert_array_equal(p.A[1], before_a[1])
    np.testing.assert_array_equal(p.b[1], before_b[1])
    assert p.predict_rewards(np.array([1., 0.])) == {"a": 0.5, "b": 0.0}
    assert p.scores(np.array([1., 0.])) == pytest.approx({"a": 0.5 + np.sqrt(0.5), "b": 1})
    assert p.select(np.array([1., 0.])) == "a"
    p.reset(7)
    np.testing.assert_array_equal(p.A, before_a)
    np.testing.assert_array_equal(p.b, before_b)


def test_seeded_random_and_fixed_routing():
    from scripts.policies import RandomPolicy, FixedPolicy
    x = np.array([1., 0.])
    p = RandomPolicy(("a", "b", "c"), 2)
    p.reset(7)
    first = [p.select(x) for _ in range(30)]
    p.reset(7)
    assert first == [p.select(x) for _ in range(30)]
    p.reset(8)
    assert first != [p.select(x) for _ in range(30)]
    f = FixedPolicy(("a", "b"), 2, arm_id="b")
    f.reset(1)
    f.observe("b", x, 0)
    assert f.select(x) == "b"


def test_uncertainty_only_differs_from_exploitation():
    from scripts.policies import ExploreOnly
    p = ExploreOnly(("a", "b"), 2)
    p.reset(1)
    x = np.array([1., 0.])
    assert p.select(x) == "a"
    p.observe("a", x, 1)
    assert p.select(x) == "b"


def test_epoch_boundaries_and_reset():
    from scripts.policies import EpochGreedy
    p = EpochGreedy(("a", "b"), 2, window=3)
    x = np.array([1., 0.])
    p.reset(9)
    modes, trace = [], []
    for _ in range(6):
        modes.append(p.exploring)
        arm = p.select(x)
        trace.append(arm)
        p.observe(arm, x, float(arm == "b"))
    assert modes == [True, False, False, True, False, False]
    p.reset(9)
    assert p.exploring and p.rounds == 0
    assert p.select(x) == trace[0]


@pytest.mark.parametrize("x,reward", [([1, 2, 3], 1), ([np.nan, 1], 1),
    ([np.inf, 0], 1), ([1, 0], np.nan), ([1, 0], -1), ([1, 0], 1.1),
    ([1e308, 0], 1), ([1e8, 1e8], 1)])
def test_invalid_feedback_never_partially_mutates_state(x, reward):
    from scripts.policies import LinearUCB
    p = LinearUCB(("a", "b"), 2)
    before = p.A.copy(), p.b.copy()
    with pytest.raises(ValueError):
        p.observe("a", np.array(x), reward)
    np.testing.assert_array_equal(p.A, before[0])
    np.testing.assert_array_equal(p.b, before[1])
    assert p.rounds == 0


@pytest.mark.parametrize("kwargs", [{"ridge": 0}, {"ridge": np.nan}, {"alpha": -1}, {"alpha": np.inf}])
def test_invalid_ridge_parameters(kwargs):
    from scripts.policies import LinearUCB
    with pytest.raises(ValueError):
        LinearUCB(("a", "b"), 2, **kwargs)


def test_zero_context_ties_and_wrong_arm():
    from scripts.policies import LinearUCB
    p = LinearUCB(("b", "a"), 2, alpha=0)
    assert p.select(np.zeros(2)) == "b"
    with pytest.raises(ValueError):
        p.observe("missing", np.zeros(2), 0)
