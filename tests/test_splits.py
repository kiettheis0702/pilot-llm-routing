import hashlib

import pandas as pd
import pytest

from scripts.data_gate import make_splits, query_hash


def split_fixture(tmp_path):
    queries = pd.DataFrame({"query_id": [f"{i:03}" for i in range(60)],
                            "query": [f"question {i}" for i in range(60)], "task": "toy"})
    queries.loc[1, "query"] = " question  0 "
    queries["query_hash"] = queries["query"].map(query_hash)
    manifest = make_splits(queries, seed=42, tuning_size=10)
    path = tmp_path / "splits.csv"
    manifest.to_csv(path, index=False)
    return queries, manifest, path


def load(path, queries, **kwargs):
    from scripts.splits import load_split_manifest
    return load_split_manifest(path, queries, expected_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                               tuning_size=10, **kwargs)


def test_frozen_manifest_and_subset_preserve_ids_order_and_bytes(tmp_path):
    q, frame, path = split_fixture(tmp_path)
    original = path.read_bytes()
    splits = load(path, q)
    assert splits.counts == {"deployment": 5, "learning": 45, "tuning": 10}
    full = splits.select("learning")
    assert full.query_ids == tuple(frame[frame.split == "learning"].query_id)
    assert all(len(qid) == 3 for qid in full.query_ids)
    small = splits.select("learning", limit=8, subset_seed=7)
    assert small == splits.select("learning", limit=8, subset_seed=7)
    assert small.query_ids != splits.select("learning", limit=8, subset_seed=8).query_ids
    assert small.query_ids == tuple(qid for qid in full.query_ids if qid in small.query_ids)
    assert small.full_size == 45 and small.limit == 8 and small.subset_seed == 7
    assert full.limit is None and full.seed == 42
    assert small.manifest_sha256 == hashlib.sha256(original).hexdigest()
    assert path.read_bytes() == original


@pytest.mark.parametrize("damage", ["duplicate", "missing", "unknown_id", "split", "group",
                                   "hash", "task", "position", "fractional_position"])
def test_checksum_valid_semantic_corruption_is_rejected(tmp_path, damage):
    q, m, path = split_fixture(tmp_path)
    if damage == "duplicate":
        m = pd.concat([m, m.iloc[:1]])
    elif damage == "missing":
        m = m.iloc[1:]
    elif damage == "unknown_id":
        m.loc[0, "query_id"] = "unknown"
    elif damage == "split":
        m.loc[0, "split"] = "train"
    elif damage == "group":
        m.loc[m.query_id == "000", "split"] = "tuning" if m.loc[m.query_id == "001", "split"].item() != "tuning" else "learning"
    elif damage in {"hash", "task"}:
        m.loc[0, "query_hash" if damage == "hash" else "task"] = "wrong"
    elif damage == "position":
        m.loc[0, "stream_position"] = 999
    else:
        m["stream_position"] = m.stream_position.astype(float)
        m.loc[0, "stream_position"] = 0.5
    m.to_csv(path, index=False)
    with pytest.raises(ValueError):
        load(path, q)


def test_checksum_seed_and_query_text_are_verified(tmp_path):
    from scripts.splits import load_split_manifest
    q, _, path = split_fixture(tmp_path)
    with pytest.raises(ValueError, match="SHA-256"):
        load_split_manifest(path, q, expected_sha256="0" * 64, tuning_size=10)
    with pytest.raises(ValueError):
        load(path, q, seed=43)
    q.loc[0, "query"] = "edited"
    with pytest.raises(ValueError):
        load(path, q)


@pytest.mark.parametrize("dtype", ["string", "string[pyarrow]"])
@pytest.mark.parametrize("columns", [("query_hash",), ("query_id", "task"),
                                    ("query_id", "query", "query_hash", "task")])
def test_string_extension_dtypes_do_not_change_split_semantics(tmp_path, dtype, columns):
    q, _, path = split_fixture(tmp_path)
    expected = load(path, q).select("learning")
    converted = q.astype({column: dtype for column in columns})
    original_dtypes = converted.dtypes.copy()
    assert load(path, converted).select("learning") == expected
    pd.testing.assert_series_equal(converted.dtypes, original_dtypes)


@pytest.mark.parametrize("kwargs", [{"split": "other"}, {"split": "learning", "limit": 0},
    {"split": "learning", "limit": -1}, {"split": "learning", "limit": 46},
    {"split": "learning", "limit": 1.5}, {"split": "learning", "limit": True}])
def test_invalid_subset_is_not_silently_truncated(tmp_path, kwargs):
    q, _, path = split_fixture(tmp_path)
    splits = load(path, q)
    with pytest.raises(ValueError):
        splits.select(**kwargs)
