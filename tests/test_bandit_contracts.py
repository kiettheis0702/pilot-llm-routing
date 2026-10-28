import numpy as np
import pandas as pd
import pytest

from scripts.data_gate import OfflineBandit


def tables():
    queries = pd.DataFrame({"query_id": ["01", "02"], "query": ["first", "second"], "task": ["t", "t"]})
    outcomes = pd.DataFrame({"query_id": ["01", "01", "02", "02"],
                             "arm_id": ["a", "b", "a", "b"],
                             "reward": [0.25, 1.0, 1.0, 0.0],
                             "actual_cost": [0.1, 0.3, 0.2, 0.4]})
    return queries, outcomes


@pytest.mark.parametrize("field,value", [("reward", np.nan), ("reward", np.inf),
    ("reward", -0.1), ("reward", 1.1), ("actual_cost", np.nan),
    ("actual_cost", np.inf), ("actual_cost", -1), ("reward", "bad")])
def test_environment_rejects_invalid_numeric_inputs(field, value):
    q, o = tables()
    o[field] = o[field].astype(object)
    o.loc[0, field] = value
    with pytest.raises(ValueError):
        OfflineBandit(q, o)


@pytest.mark.parametrize("damage", ["missing_pair", "extra_query", "null_id", "duplicate_id", "empty_id", "empty", "unknown_arm"])
def test_environment_rejects_invalid_grid(damage):
    q, o = tables()
    if damage == "missing_pair":
        o = o.iloc[1:]
    elif damage == "extra_query":
        o.loc[0, "query_id"] = "missing"
    elif damage == "null_id":
        q.loc[0, "query_id"] = None
    elif damage == "duplicate_id":
        q.loc[1, "query_id"] = "01"
    elif damage == "empty_id":
        o.loc[0, "arm_id"] = ""
    elif damage == "empty":
        q, o = q.iloc[:0], o.iloc[:0]
    else:
        o.loc[0, "arm_id"] = "c"
    with pytest.raises(ValueError):
        OfflineBandit(q, o)


def test_declared_catalog_and_defensive_copy():
    q, o = tables()
    env = OfflineBandit(q, o, arm_catalog=("b", "a"))
    q.loc[0, "query"] = "changed"
    o.loc[0, "reward"] = 0.9
    assert env.observe("01").query == "first"
    assert env.observe("01").arm_ids == ("b", "a")
    assert env.step("01", "a").reward == 0.25
    with pytest.raises(ValueError):
        OfflineBandit(q, o, arm_catalog=("a",))
    with pytest.raises(ValueError):
        OfflineBandit(q, o, arm_catalog=("a", "a", "b"))
    single = OfflineBandit(q, o[o.arm_id == "a"], arm_catalog=("a",))
    assert single.step("02", "a").reward == 1.0


@pytest.mark.parametrize("vectors", [{}, {"01": []}, {"01": [[1, 2]]},
    {"01": [np.nan]}, {"01": [np.inf]}, {"01": [1], "02": [1, 2]}, {"": [1]}])
def test_context_validation(vectors):
    from scripts.bandit_contracts import ContextTable
    with pytest.raises(ValueError):
        ContextTable(vectors, label="synthetic-test")


def test_context_is_copied_at_input_and_output():
    from scripts.bandit_contracts import ContextTable
    vector = np.array([1.0, 2.0])
    context = ContextTable({"01": vector}, label="synthetic-test")
    vector[0] = 100
    context.get("01")[1] = 99
    np.testing.assert_array_equal(context.get("01"), [1, 2])
    assert context.dimension == 2
    with pytest.raises(KeyError):
        context.get("missing")
