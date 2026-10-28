"""Small disjoint contextual-bandit baselines; no outcome table or costs are inputs."""
from __future__ import annotations

import numpy as np


class _Policy:
    def __init__(self, arms: tuple[str, ...], dimension: int):
        self.arms = tuple(arms)
        if (not self.arms or any(not isinstance(a, str) or not a.strip() for a in self.arms)
                or len(set(self.arms)) != len(self.arms)):
            raise ValueError("Arms must be unique nonempty strings")
        if type(dimension) is not int or dimension < 1:
            raise ValueError("Dimension must be a positive integer")
        self.dimension = dimension
        _Policy.reset(self, 0)

    def reset(self, seed: int) -> None:
        if type(seed) is not int or seed < 0:
            raise ValueError("Seed must be a nonnegative integer")
        self.rng = np.random.default_rng(seed)
        self.rounds = 0

    def _context(self, context) -> np.ndarray:
        x = np.asarray(context, dtype=float)
        if x.shape != (self.dimension,) or not np.isfinite(x).all():
            raise ValueError("Context must have the configured dimension and finite values")
        return x

    def _feedback(self, arm_id, context, reward):
        if arm_id not in self.arms:
            raise ValueError("Unknown arm")
        x = self._context(context)
        reward = float(reward)
        if not np.isfinite(reward) or not 0 <= reward <= 1:
            raise ValueError("Reward must be finite in [0,1]")
        return self.arms.index(arm_id), x, reward

    def observe(self, arm_id, context, reward) -> None:
        self._feedback(arm_id, context, reward)
        self.rounds += 1


class RandomPolicy(_Policy):
    def select(self, context) -> str:
        self._context(context)
        return self.arms[int(self.rng.integers(len(self.arms)))]

    def predict_rewards(self, context) -> dict[str, float]:
        self._context(context)
        return dict.fromkeys(self.arms, 1 / len(self.arms))


class FixedPolicy(_Policy):
    def __init__(self, arms, dimension, *, arm_id: str):
        super().__init__(arms, dimension)
        if arm_id not in self.arms:
            raise ValueError("Fixed arm must belong to catalog")
        self.arm_id = arm_id

    def select(self, context) -> str:
        self._context(context)
        return self.arm_id

    def predict_rewards(self, context) -> dict[str, float]:
        self._context(context)
        return {arm: float(arm == self.arm_id) for arm in self.arms}


class LinearUCB(_Policy):
    def __init__(self, arms, dimension, *, alpha: float = 1.0, ridge: float = 1.0):
        self.alpha, self.ridge = float(alpha), float(ridge)
        if not np.isfinite(self.alpha) or self.alpha < 0 or not np.isfinite(self.ridge) or self.ridge <= 0:
            raise ValueError("Alpha must be finite nonnegative; ridge finite positive")
        super().__init__(arms, dimension)
        self.reset(0)

    def reset(self, seed: int) -> None:
        super().reset(seed)
        self._A = np.repeat((self.ridge * np.eye(self.dimension))[None, :, :], len(self.arms), axis=0)
        self._b = np.zeros((len(self.arms), self.dimension))

    @property
    def A(self) -> np.ndarray:
        return self._A.copy()

    @property
    def b(self) -> np.ndarray:
        return self._b.copy()

    def _estimates(self, context) -> tuple[np.ndarray, np.ndarray]:
        x = self._context(context)
        try:
            theta = np.linalg.solve(self._A, self._b[..., None])[..., 0]
            directions = np.linalg.solve(self._A, np.broadcast_to(x, self._b.shape)[..., None])[..., 0]
            with np.errstate(over="ignore", invalid="ignore"):
                means, variance = theta @ x, directions @ x
        except np.linalg.LinAlgError as exc:
            raise ValueError("Invalid ridge state") from exc
        if not np.isfinite(means).all() or not np.isfinite(variance).all() or (variance < -1e-10).any():
            raise ValueError("Nonfinite prediction or invalid uncertainty")
        return means, np.sqrt(np.maximum(variance, 0))

    def predict_rewards(self, context) -> dict[str, float]:
        means, _ = self._estimates(context)
        return dict(zip(self.arms, map(float, means)))

    def scores(self, context) -> dict[str, float]:
        means, uncertainty = self._estimates(context)
        with np.errstate(over="ignore", invalid="ignore"):
            values = means + self.alpha * uncertainty
        if not np.isfinite(values).all():
            raise ValueError("Nonfinite UCB scores")
        return dict(zip(self.arms, map(float, values)))

    def select(self, context) -> str:
        scores = self.scores(context)
        return max(scores, key=scores.get)

    def observe(self, arm_id, context, reward) -> None:
        index, x, value = self._feedback(arm_id, context, reward)
        with np.errstate(over="ignore", invalid="ignore"):
            next_A = self._A[index] + np.outer(x, x)
            next_b = self._b[index] + value * x
        if not np.isfinite(next_A).all() or not np.isfinite(next_b).all():
            raise ValueError("Feedback would create nonfinite ridge state")
        try:
            np.linalg.cholesky(next_A)
        except np.linalg.LinAlgError as exc:
            raise ValueError("Feedback would destroy positive-definite ridge state") from exc
        self._A[index], self._b[index] = next_A, next_b
        self.rounds += 1


class ExploreOnly(LinearUCB):
    """PAPER_INFERRED: maximize uncertainty; retain ridge learning for later rankings."""

    def scores(self, context) -> dict[str, float]:
        _, uncertainty = self._estimates(context)
        return dict(zip(self.arms, map(float, uncertainty)))


class EpochGreedy(LinearUCB):
    """PAPER_INFERRED: one random round per fixed window, then linear exploitation."""

    def __init__(self, arms, dimension, *, window: int = 10, ridge: float = 1.0):
        if type(window) is not int or window < 1:
            raise ValueError("Window must be a positive integer")
        self.window = window
        super().__init__(arms, dimension, alpha=0, ridge=ridge)

    @property
    def exploring(self) -> bool:
        return self.rounds % self.window == 0

    def select(self, context) -> str:
        self._context(context)
        if self.exploring:
            return self.arms[int(self.rng.integers(len(self.arms)))]
        scores = self.predict_rewards(context)
        return max(scores, key=scores.get)
