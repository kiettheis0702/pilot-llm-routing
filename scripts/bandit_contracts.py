"""Numeric learning boundary. No outcome, cost or oracle is a policy input."""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Protocol

import numpy as np


class OnlinePolicy(Protocol):
    """Policies own their arm catalog. reset must restore all learned/RNG state."""

    def reset(self, seed: int) -> None: ...
    def select(self, context: np.ndarray) -> str: ...
    def observe(self, arm_id: str, context: np.ndarray, reward: float) -> None: ...
    def predict_rewards(self, context: np.ndarray) -> dict[str, float]: ...


class ContextTable:
    """Copy finite vectors; label their recipe without implying paper embeddings."""

    def __init__(self, vectors: Mapping[str, Sequence[float]], *, label: str):
        if not vectors or not isinstance(label, str) or not label.strip():
            raise ValueError("Contexts and their recipe label must be nonempty")
        self._vectors: dict[str, np.ndarray] = {}
        dimension = None
        for query_id, value in vectors.items():
            if not isinstance(query_id, str) or not query_id.strip():
                raise ValueError("Context query IDs must be nonempty strings")
            try:
                vector = np.array(value, dtype=float, copy=True)
            except (TypeError, ValueError) as exc:
                raise ValueError("Context must be a numeric vector") from exc
            if vector.ndim != 1 or vector.size == 0 or not np.isfinite(vector).all():
                raise ValueError("Context must be a finite nonempty 1D vector")
            if dimension is not None and vector.size != dimension:
                raise ValueError("Context dimensions must agree")
            dimension = vector.size
            self._vectors[query_id] = vector
        self._dimension = int(dimension)
        self._label = label

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def label(self) -> str:
        return self._label

    def get(self, query_id: str) -> np.ndarray:
        return self._vectors[query_id].copy()

    def fingerprint(self, query_ids: Sequence[str]) -> str:
        payload = [(q, self._vectors[q].tolist()) for q in query_ids]
        return hashlib.sha256(json.dumps(payload, ensure_ascii=False, allow_nan=False,
                                         separators=(",", ":")).encode("utf-8")).hexdigest()
