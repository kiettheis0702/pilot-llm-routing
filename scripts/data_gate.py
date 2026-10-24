"""Offline Data Gate: validation, provisional normalization/splits, selected-arm smoke."""
from __future__ import annotations

import ast
import hashlib
import math
import random
from dataclasses import dataclass

import numpy as np
import pandas as pd


def text_segments(value: str, *, json_format: bool = False) -> list[str]:
    import json

    result = json.loads(value) if json_format else ast.literal_eval(value)
    if not isinstance(result, list) or not result or not all(isinstance(x, str) for x in result):
        raise ValueError("Expected a nonempty list of text segments")
    if not any(x.strip() for x in result):
        raise ValueError("Prompt has no nonempty text")
    return result


def query_hash(text: str) -> str:
    # Whitespace-equivalent prompts stay in the same split; original text is retained.
    return hashlib.sha256(" ".join(text.split()).encode("utf-8")).hexdigest()


def arm_ids(frame: pd.DataFrame) -> list[str]:
    arms = sorted(c.removesuffix("|total_cost") for c in frame if c.endswith("|total_cost"))
    if not arms:
        raise ValueError("No arm cost columns found")
    required = {"sample_id", "prompt", "eval_name"}
    required.update(arms)
    required.update(a + "|model_response" for a in arms)
    if missing := required - set(frame.columns):
        raise ValueError(f"Missing RouterBench columns: {sorted(missing)}")
    return arms


def normalize_routerbench(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    import json

    arms = arm_ids(frame)
    if frame.sample_id.isna().any() or frame.sample_id.duplicated().any():
        raise ValueError("RouterBench IDs must be unique and non-null")
    if frame[["prompt", "eval_name"]].isna().any().any():
        raise ValueError("Missing prompt/task")
    scores = frame[arms].apply(pd.to_numeric, errors="raise").to_numpy(dtype=float)
    costs = frame[[a + "|total_cost" for a in arms]].to_numpy(dtype=float)
    if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError("Rewards must be finite and within [0, 1]")
    if not np.isfinite(costs).all() or (costs < 0).any():
        raise ValueError("Costs must be finite and nonnegative")
    parts = frame.prompt.map(text_segments)
    queries = pd.DataFrame({
        "query_id": frame.sample_id.astype(str),
        "query": parts.map(lambda xs: "\n\n".join(xs)),
        "query_segments_json": parts.map(lambda xs: json.dumps(xs, ensure_ascii=False)),
        "task": frame.eval_name,
    }).reset_index(drop=True)
    queries["query_hash"] = queries["query"].map(query_hash)
    outcomes = pd.DataFrame({
        "query_id": np.repeat(queries.query_id.to_numpy(), len(arms)),
        "arm_id": np.tile(arms, len(frame)),
        "reward": scores.ravel(),
        "actual_cost": costs.ravel(),
        # A tuning-only token estimator is not implemented yet. Never substitute actual cost.
        "estimated_cost": np.nan,
    })
    return queries, outcomes


def make_splits(queries: pd.DataFrame, seed: int = 42, tuning_size: int = 1000,
                learning_ratio: int = 10) -> pd.DataFrame:
    if tuning_size < 1 or learning_ratio < 1 or len(queries) <= tuning_size + 1:
        raise ValueError("Insufficient rows or invalid split sizes")
    if queries.query_id.isna().any() or queries.query_id.duplicated().any():
        raise ValueError("Split query IDs must be unique and non-null")
    groups = queries.groupby("query_hash", sort=True).size().to_dict()
    order = sorted(groups, key=lambda h: hashlib.sha256(f"{seed}:{h}".encode()).hexdigest())
    assigned = {}
    targets = {"tuning": tuning_size, "deployment": math.ceil((len(queries) - tuning_size) / (learning_ratio + 1))}
    for label, target in targets.items():
        remaining = target
        for group in order:
            if group not in assigned and groups[group] <= remaining:
                assigned[group] = label
                remaining -= groups[group]
        if remaining:
            raise ValueError(f"Cannot preserve prompt groups and reach exact {label} size")
    result = queries[["query_id", "query_hash", "task"]].copy()
    result["split"] = result.query_hash.map(lambda h: assigned.get(h, "learning"))
    # Rank within each bucket independently of incoming row order.
    result["_order"] = result.query_id.map(lambda q: hashlib.sha256(f"{seed}:stream:{q}".encode()).hexdigest())
    result = result.sort_values(["split", "_order", "query_id"]).drop(columns="_order").reset_index(drop=True)
    result["stream_position"] = result.groupby("split").cumcount()
    if result.groupby("query_hash").split.nunique().max() != 1:
        raise ValueError("Prompt group leaked across splits")
    return result


@dataclass(frozen=True)
class Observation:
    query_id: str
    query: str
    task: str
    arm_ids: tuple[str, ...]


@dataclass(frozen=True)
class Feedback:
    query_id: str
    arm_id: str
    reward: float
    actual_cost: float


class OfflineBandit:
    """One-use feedback interface; full outcomes remain inside the environment."""

    def __init__(self, queries: pd.DataFrame, outcomes: pd.DataFrame):
        if queries.query_id.duplicated().any() or outcomes.duplicated(["query_id", "arm_id"]).any():
            raise ValueError("Duplicate query/arm keys")
        self._queries = queries.set_index("query_id")
        self._outcomes = outcomes.set_index(["query_id", "arm_id"])
        self._seen: set[str] = set()

    def observe(self, query_id: str) -> Observation:
        row = self._queries.loc[query_id]
        arms = tuple(sorted(self._outcomes.loc[query_id].index))
        return Observation(query_id, row["query"], row["task"], arms)

    def step(self, query_id: str, arm_id: str) -> Feedback:
        if query_id in self._seen:
            raise ValueError("Reward already revealed for this query")
        row = self._outcomes.loc[(query_id, arm_id)]
        self._seen.add(query_id)
        return Feedback(query_id, arm_id, float(row.reward), float(row.actual_cost))


class RandomSmokeLearner:
    """I/O smoke learner, not an implementation of any paper baseline."""

    def __init__(self, seed: int):
        self._random = random.Random(seed)
        self.updates: list[Feedback] = []

    def select(self, observation: Observation) -> str:
        return self._random.choice(observation.arm_ids)

    def update(self, feedback: Feedback) -> None:
        self.updates.append(feedback)


def run_bandit_round(environment: OfflineBandit, query_id: str, learner) -> Feedback:
    observation = environment.observe(query_id)
    chosen = learner.select(observation)
    feedback = environment.step(query_id, chosen)
    learner.update(feedback)
    return feedback
