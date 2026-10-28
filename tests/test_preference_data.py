import json

import pandas as pd
import pytest

from scripts import preference_data as pref
from scripts.data_gate import query_hash


def mappings():
    return [
        {"arena": "a", "routerbench": "a", "status": "exact_id"},
        {"arena": "b", "routerbench": "b", "status": "exact_id"},
        {"arena": "old-c", "routerbench": "c", "status": "proposed_alias"},
    ]


def row(identity="0", **changes):
    value = {"id": identity, "model_a": "a", "model_b": "b",
             "prompt": '["private prompt"]', "response_a": '["private response a"]',
             "response_b": '["private response b"]',
             "winner_model_a": 1, "winner_model_b": 0, "winner_tie": 0}
    return value | changes


def test_preparation_keeps_exact_distinct_decisive_pairs_without_response_text():
    frame = pd.DataFrame([row("2"), row("1", winner_model_a=0, winner_model_b=1),
                          row("alias", model_b="old-c"), row("same", model_b="a")])
    pairs, diagnostic = pref.prepare_preferences(frame, mappings())
    assert pairs.battle_id.tolist() == ["1", "2"]
    assert pairs.winner_arm.tolist() == ["b", "a"]
    assert pairs.loser_arm.tolist() == ["a", "b"]
    assert not {"response_a", "response_b"} & set(pairs)
    assert diagnostic["eligible_rows"] == 2
    assert diagnostic["exclusion_reason_counts"]["unapproved_model_mapping"] == 1
    assert diagnostic["exclusion_reason_counts"]["same_model"] == 1


@pytest.mark.parametrize("changes,reason", [
    ({"winner_model_a": 0, "winner_tie": 1}, "tie"),
    ({"winner_model_a": 1, "winner_model_b": 1}, "invalid_winner"),
    ({"winner_model_a": "1"}, "invalid_winner"),
    ({"winner_model_a": float("nan")}, "invalid_winner"),
    ({"prompt": None}, "prompt_missing"),
    ({"prompt": "bad"}, "prompt_malformed_json"),
    ({"prompt": '{}'}, "prompt_nonlist"),
    ({"prompt": '[]'}, "prompt_empty"),
    ({"prompt": '[" "]'}, "prompt_blank_segment"),
    ({"response_a": '[null]'}, "response_a_nonstring_segment"),
    ({"response_b": '[""]'}, "response_b_blank_segment"),
    ({"response_b": '[4]'}, "response_b_nonstring_segment"),
    ({"prompt": '["one", "two"]'}, "multi_turn"),
    ({"response_a": '["one", "two"]'}, "multi_turn"),
    ({"model_a": None}, "missing_model"),
])
def test_preparation_counts_each_ineligible_condition(changes, reason):
    pairs, diagnostic = pref.prepare_preferences(pd.DataFrame([row(**changes)]), mappings())
    assert pairs.empty
    assert diagnostic["exclusion_reason_counts"][reason] == 1
    assert sum(diagnostic["primary_exclusion_counts"].values()) == 1


def test_duplicate_ids_and_ambiguous_or_false_exact_mappings_fail_closed():
    with pytest.raises(ValueError, match="identit"):
        pref.prepare_preferences(pd.DataFrame([row(), row()]), mappings())
    with pytest.raises(ValueError, match="identit"):
        pref.prepare_preferences(pd.DataFrame([row(None)]), mappings())
    with pytest.raises(ValueError, match="mapping"):
        pref.prepare_preferences(pd.DataFrame([row()]), mappings() + [mappings()[0]])
    false_exact = mappings()
    false_exact[2]["status"] = "exact_id"
    with pytest.raises(ValueError, match="literal"):
        pref.prepare_preferences(pd.DataFrame([row()]), false_exact)


def many_pairs():
    frame = pd.DataFrame([row(str(i), prompt=json.dumps([f"prompt {i}"])) for i in range(10)])
    frame.loc[9, "prompt"] = '["  prompt   0 "]'
    return pref.prepare_preferences(frame, mappings())[0]


def test_split_is_seeded_order_independent_and_never_splits_prompt_group():
    pairs = many_pairs()
    first = pref.split_preferences(pairs, seed=42, holdout_fraction=0.2)
    shuffled = pref.split_preferences(pairs.sample(frac=1, random_state=9), seed=42, holdout_fraction=0.2)
    pd.testing.assert_frame_equal(first, shuffled)
    assert first.groupby("prompt_hash").split.nunique().eq(1).all()
    assert first[first.split == "holdout"].prompt_hash.nunique() == 2
    assert first[first.split == "train"].prompt_hash.nunique() == 7
    assert set(first.battle_id) == set(str(i) for i in range(10))
    assert pref.table_fingerprint(first) == pref.table_fingerprint(shuffled)
    altered = first.copy()
    altered.loc[0, "winner_arm"] = "changed"
    assert pref.table_fingerprint(first) != pref.table_fingerprint(altered)
    assert not first.equals(pref.split_preferences(pairs, seed=43, holdout_fraction=0.2))


@pytest.mark.parametrize("fraction", [0, 1, -0.1, float("nan"), float("inf")])
def test_split_rejects_invalid_fraction(fraction):
    with pytest.raises(ValueError, match="fraction"):
        pref.split_preferences(many_pairs(), holdout_fraction=fraction)


def test_split_rejects_one_group_and_forged_prompt_hash():
    pairs = many_pairs()
    with pytest.raises(ValueError, match="groups"):
        pref.split_preferences(pairs.iloc[:1])
    pairs.loc[0, "prompt_hash"] = "forged"
    with pytest.raises(ValueError, match="hash"):
        pref.split_preferences(pairs)


def test_readiness_reports_missing_pool_coverage_overlap_and_no_private_text():
    frame = pd.DataFrame([row(str(i), prompt=json.dumps([f"private prompt {i}"])) for i in range(10)])
    queries = pd.DataFrame({"query_hash": [query_hash("private prompt 0")],
                            "query_segments_json": ['["private prompt 1"]']})
    report = pref.preference_readiness(frame, mappings(), queries)
    assert report["all_arm_prior_ready"] is False
    assert report["missing_eligible_arms"] == ["c"]
    assert report["split"]["query_hash_overlap"] == 0
    assert report["routerbench_overlap"]["full_query_rows"] == 1
    assert report["routerbench_overlap"]["segment_rows"] == 1
    assert report["eligible"]["per_arm"]["a"] == {"comparisons": 10, "wins": 10, "losses": 0}
    assert report["eligible"]["pair_counts"] == {"a / b": 10}
    assert report["triplets"]["valid_triplets"] is None
    assert "private prompt" not in json.dumps(report)
    assert "private response" not in json.dumps(report)


def test_no_eligible_data_produces_blocked_readiness_not_false_success():
    frame = pd.DataFrame([row("1", model_b="old-c")])
    queries = pd.DataFrame(columns=["query_hash", "query_segments_json"])
    report = pref.preference_readiness(frame, mappings(), queries)
    assert report["split"]["status"] == "blocked"
    assert report["all_arm_prior_ready"] is False
    assert report["missing_eligible_arms"] == ["a", "b", "c"]


def test_embedding_inventory_is_scoped_and_never_guesses_from_array_filename(tmp_path):
    scoped = tmp_path / "chosen"
    scoped.mkdir()
    (scoped / "query_embeddings.npy").write_bytes(b"not an actual embedding")
    (scoped / ".venv").mkdir()
    (scoped / ".venv" / "secret_embeddings.npy").write_bytes(b"ignored")
    (tmp_path / "outside_embeddings.npy").write_bytes(b"outside scope")
    inventory = pref.embedding_inventory([scoped])
    assert len(inventory["candidates"]) == 1
    assert inventory["candidates"][0]["relative_path"] == "query_embeddings.npy"
    assert inventory["embedding_model"] is None
    assert inventory["dimension"] is None
    assert inventory["validated_cache_ready"] is False


def test_prepared_query_hash_verification_fails_before_reading(tmp_path):
    path = tmp_path / "queries.parquet"
    path.write_bytes(b"invalid parquet")
    with pytest.raises(ValueError, match="fingerprint"):
        pref.load_verified_queries(path, "0" * 64)
