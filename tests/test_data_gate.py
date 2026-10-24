import hashlib
import io
import json
import pickle
from dataclasses import asdict

import numpy as np
import pandas as pd
import pytest

from scripts.audit_data import preference_profile, response_profile
from scripts.data_gate import OfflineBandit, make_splits, normalize_routerbench, query_hash, run_bandit_round, text_segments
from scripts.data_io import RouterBenchUnpickler, load_routerbench


def fixture_frame():
    return pd.DataFrame({
        "sample_id": ["a", "b"], "prompt": ["['system', 'question']", "['other']"],
        "eval_name": ["task", "task"], "model-a": [0.25, 1.0], "model-b": [1.0, 0.0],
        "model-a|total_cost": [0.1, 0.2], "model-b|total_cost": [0.3, 0.4],
        "model-a|model_response": ["['answer-a']", "['answer-a']"],
        "model-b|model_response": ["['answer-b']", "['answer-b']"],
        "oracle_model_to_route_to": ["model-b", "model-a"],
    })


def test_normalization_preserves_fractional_rewards_and_prompt_segments():
    queries, outcomes = normalize_routerbench(fixture_frame())
    assert queries.loc[0, "query"] == "system\n\nquestion"
    assert json.loads(queries.loc[0, "query_segments_json"]) == ["system", "question"]
    assert outcomes.set_index(["query_id", "arm_id"]).loc[("a", "model-a"), "reward"] == 0.25
    assert len(outcomes) == 4
    assert outcomes.estimated_cost.isna().all()
    assert not {"reward", "actual_cost", "oracle_model_to_route_to"} & set(queries)


@pytest.mark.parametrize("column,value", [
    ("model-a", np.nan), ("model-a", np.inf), ("model-a", -0.1), ("model-a", 1.1),
    ("model-a|total_cost", np.inf), ("model-a|total_cost", -0.1),
])
def test_invalid_reward_or_cost_fails_closed(column, value):
    frame = fixture_frame()
    frame.loc[0, column] = value
    with pytest.raises(ValueError):
        normalize_routerbench(frame)


def test_duplicate_source_ids_rejected():
    frame = fixture_frame()
    frame.loc[1, "sample_id"] = "a"
    with pytest.raises(ValueError, match="unique"):
        normalize_routerbench(frame)


def test_split_is_stable_and_groups_whitespace_duplicates():
    queries = pd.DataFrame({"query_id": [str(i) for i in range(60)],
                            "query": [f"query {i}" for i in range(60)], "task": "test"})
    queries.loc[1, "query"] = " query   0 "
    queries["query_hash"] = queries["query"].map(query_hash)
    first = make_splits(queries, seed=42, tuning_size=10)
    shuffled = make_splits(queries.sample(frac=1, random_state=7), seed=42, tuning_size=10)
    pd.testing.assert_frame_equal(first, shuffled)
    assert first.split.value_counts().to_dict() == {"learning": 45, "tuning": 10, "deployment": 5}
    assert first.groupby("query_hash").split.nunique().max() == 1
    assert set(first.query_id) == set(queries.query_id)
    assert len(first) == len(queries)
    assert not first.equals(make_splits(queries, seed=43, tuning_size=10))


def test_learner_only_receives_selected_arm_reward():
    queries, outcomes = normalize_routerbench(fixture_frame())
    env = OfflineBandit(queries, outcomes)

    class Spy:
        def select(self, observation):
            assert set(asdict(observation)) == {"query_id", "query", "task", "arm_ids"}
            assert "answer" not in observation.query
            return "model-a"

        def update(self, feedback):
            self.feedback = feedback
            assert set(asdict(feedback)) == {"query_id", "arm_id", "reward", "actual_cost"}

    spy = Spy()
    result = run_bandit_round(env, "a", spy)
    assert result is spy.feedback
    assert result.arm_id == "model-a"
    assert result.reward == 0.25  # unselected model-b has reward 1.0
    with pytest.raises(ValueError, match="already revealed"):
        env.step("a", "model-b")


def test_invalid_arm_does_not_consume_the_query():
    queries, outcomes = normalize_routerbench(fixture_frame())
    env = OfflineBandit(queries, outcomes)
    with pytest.raises(KeyError):
        env.step("a", "invalid")
    assert env.step("a", "model-a").reward == 0.25


def test_preference_audit_keeps_ties_invalid_labels_and_versions_separate():
    frame = pd.DataFrame({
        "id": [1, 2, 3, 4, 5], "model_a": ["gpt-4-1106-preview"] * 5,
        "model_b": ["gpt-3.5-turbo-1106", "gpt-3.5-turbo-1106", "gpt-3.5-turbo-1106", "gpt-4-0613", "alias-b"],
        "prompt": ['["question"]'] * 5, "response_a": ['["a"]'] * 5, "response_b": ['["b"]'] * 5,
        "winner_model_a": [1, 0, 1, 1, 1], "winner_model_b": [0, 0, 1, 0, 0], "winner_tie": [0, 1, 0, 0, 0],
    })
    mappings = [
        {"routerbench": "gpt-4-1106-preview", "arena": "gpt-4-1106-preview", "status": "exact_id"},
        {"routerbench": "gpt-3.5-turbo-1106", "arena": "gpt-3.5-turbo-1106", "status": "exact_id"},
        {"routerbench": "other-model", "arena": "alias-b", "status": "proposed_alias"},
    ]
    queries, _ = normalize_routerbench(fixture_frame())
    result = preference_profile(frame, mappings, queries)
    assert result["invalid_winner_rows"] == 1
    assert result["mapping_scenarios"]["exact_ids_only"]["decisive_valid_pairs"] == 1
    assert result["mapping_scenarios"]["all_proposed_mappings"]["decisive_valid_pairs"] == 2
    assert not result["training_ready"]


def test_text_parser_never_evaluates_code():
    with pytest.raises((ValueError, SyntaxError)):
        text_segments("__import__('os').getcwd()")


def test_nested_missing_responses_are_not_confused_with_complete_outer_cells():
    counts = response_profile(['[null]', '[""]', '[]', '["valid"]', 'not json'], json_format=True)
    assert counts == {"malformed_or_nonlist": 1, "empty_sequences": 2, "null_segments": 1, "other_nonstring_segments": 0}


def test_pickle_rejects_unapproved_globals():
    with pytest.raises(pickle.UnpicklingError, match="Disallowed"):
        RouterBenchUnpickler(io.BytesIO(pickle.dumps(eval))).load()


def test_hash_mismatch_rejected_before_deserialization(tmp_path):
    path = tmp_path / "data.pkl"
    path.write_bytes(b"not a pickle")
    source = {"size": path.stat().st_size, "sha256": hashlib.sha256(b"different").hexdigest()}
    with pytest.raises(ValueError, match="SHA-256"):
        load_routerbench(path, source)
