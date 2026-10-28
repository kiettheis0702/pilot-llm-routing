"""Read and verify frozen split files; never rewrite prepared data."""
from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.data_gate import make_splits, query_hash


@dataclass(frozen=True)
class SplitSelection:
    query_ids: tuple[str, ...]
    split: str
    manifest_sha256: str
    seed: int
    tuning_size: int
    learning_ratio: int
    full_size: int
    limit: int | None = None
    subset_seed: int | None = None


class FrozenSplits:
    def __init__(self, frame: pd.DataFrame, *, sha256: str, seed: int,
                 tuning_size: int, learning_ratio: int):
        self._frame = frame.copy()
        self._sha256 = sha256
        self._seed = seed
        self._tuning_size = tuning_size
        self._learning_ratio = learning_ratio

    @property
    def counts(self) -> dict[str, int]:
        return self._frame.groupby("split").size().to_dict()

    def select(self, split: str, *, limit: int | None = None,
               subset_seed: int = 0) -> SplitSelection:
        if split not in {"tuning", "learning", "deployment"}:
            raise ValueError("Unknown split")
        bucket = self._frame[self._frame.split == split].sort_values("stream_position")
        ids = tuple(bucket.query_id)
        if not ids:
            raise ValueError("Split is empty")
        chosen = ids
        if limit is not None:
            if type(limit) is not int or not 1 <= limit <= len(ids):
                raise ValueError("Subset size must be an integer within the split size")
            if type(subset_seed) is not int:
                raise ValueError("Subset seed must be an integer")
            ranked = sorted(ids, key=lambda q: (hashlib.sha256(
                f"{subset_seed}:subset:{q}".encode()).hexdigest(), q))
            selected = set(ranked[:limit])
            chosen = tuple(q for q in ids if q in selected)
        return SplitSelection(chosen, split, self._sha256, self._seed,
                              self._tuning_size, self._learning_ratio, len(ids),
                              limit, subset_seed if limit is not None else None)


def load_split_manifest(path: str | Path, queries: pd.DataFrame, *, expected_sha256: str,
                        seed: int = 42, tuning_size: int = 1000,
                        learning_ratio: int = 10) -> FrozenSplits:
    for value in (seed, tuning_size, learning_ratio):
        if type(value) is not int:
            raise ValueError("Split settings must be integers")
    raw = Path(path).read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != expected_sha256:
        raise ValueError("Split SHA-256 mismatch")
    required = {"query_id", "query", "query_hash", "task"}
    if queries.empty or not queries.columns.is_unique or not required <= set(queries):
        raise ValueError("Missing or duplicate query columns")
    for column in required:
        if not queries[column].map(lambda v: isinstance(v, str) and bool(v.strip())).all():
            raise ValueError("Query metadata must contain nonempty strings")
    if queries.query_id.duplicated().any():
        raise ValueError("Duplicate query IDs")
    # Already validated values are strings. Compare their content independently
    # of pandas object/string/Arrow storage, without changing the caller's frame.
    queries = queries[["query_id", "query", "query_hash", "task"]].astype(object)
    if not queries["query"].map(query_hash).equals(queries.query_hash):
        raise ValueError("Query text does not match stored hashes")
    frame = pd.read_csv(io.BytesIO(raw), dtype=str, keep_default_na=False)
    columns = ["query_id", "query_hash", "task", "split", "stream_position"]
    if set(frame) != set(columns) or frame.query_id.duplicated().any():
        raise ValueError("Invalid split columns or duplicate IDs")
    if set(frame.query_id) != set(queries.query_id):
        raise ValueError("Split query coverage mismatch")
    if not frame.split.isin(["tuning", "learning", "deployment"]).all():
        raise ValueError("Unknown split name")
    positions = pd.to_numeric(frame.stream_position, errors="raise")
    if not np.isfinite(positions).all() or (positions < 0).any() or (positions % 1 != 0).any():
        raise ValueError("Invalid stream positions")
    frame["stream_position"] = positions.astype("int64")
    if frame.groupby("query_hash").split.nunique().gt(1).any():
        raise ValueError("Prompt group crosses split boundaries")
    expected = make_splits(queries, seed=seed, tuning_size=tuning_size, learning_ratio=learning_ratio)
    # Compare by ID so CSV row order cannot silently change the learning stream.
    actual = frame[columns].sort_values("query_id").reset_index(drop=True)
    expected = expected[columns].sort_values("query_id").reset_index(drop=True)
    if not actual.equals(expected):
        raise ValueError("Split metadata, assignments or order disagree with the declared protocol")
    return FrozenSplits(frame, sha256=digest, seed=seed,
                        tuning_size=tuning_size, learning_ratio=learning_ratio)
