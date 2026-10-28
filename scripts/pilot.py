"""PILOT online mathematics with explicit priors; no pretraining is implied."""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path

import numpy as np

from scripts.policies import LinearUCB


def _array(value, shape) -> np.ndarray:
    try:
        raw = np.asarray(value)
        if raw.dtype.kind not in "iuf":
            raise ValueError("Expected real numeric values")
        result = np.array(raw, dtype=float, copy=True)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Expected real numeric values") from exc
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError("Invalid array shape or nonfinite values")
    return result


def _unit(vector: np.ndarray, *, allow_zero: bool = False) -> np.ndarray:
    scale = np.max(np.abs(vector))
    if scale == 0:
        if allow_zero:
            return np.zeros_like(vector)
        raise ValueError("Zero context has no direction")
    scaled = vector / scale
    return scaled / np.linalg.norm(scaled)


def _canonical(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False)


def _digest(value) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _unique_object(pairs):
    obj = {}
    for key, value in pairs:
        if key in obj:
            raise ValueError("Duplicate JSON key")
        obj[key] = value
    return obj


class PILOT(LinearUCB):
    """Cosine UCB with normalized contexts and a required, labeled caller prior."""

    def __init__(self, arms, dimension: int, *, priors: Mapping, lambdas: Mapping,
                 provenance: str, alpha: float = 1.0):
        arms = tuple(arms)
        if (not arms or any(not isinstance(a, str) or not a.strip() for a in arms)
                or len(set(arms)) != len(arms)):
            raise ValueError("Arms must be unique nonempty strings")
        if type(dimension) is not int or dimension < 1:
            raise ValueError("Dimension must be a positive integer")
        if (not isinstance(priors, Mapping) or not isinstance(lambdas, Mapping)
                or set(priors) != set(arms) or set(lambdas) != set(arms)):
            raise ValueError("Prior and lambda mappings must match the catalog exactly")
        if not isinstance(provenance, str) or not provenance.strip():
            raise ValueError("Prior provenance must be nonempty")
        self._priors = _array([priors[a] for a in arms], (len(arms), dimension))
        self._lambdas = _array([lambdas[a] for a in arms], (len(arms),))
        if (self._lambdas <= 0).any():
            raise ValueError("Each lambda must be positive")
        self.provenance = provenance
        super().__init__(arms, dimension, alpha=alpha)

    def reset(self, seed: int) -> None:
        # Construct/validate before replacing a learned model.
        A = self._lambdas[:, None, None] * np.eye(self.dimension)[None, :, :]
        with np.errstate(over="ignore", invalid="ignore"):
            b = self._lambdas[:, None] * self._priors
        self._validate_state(A, b)
        super().reset(seed)
        self._A, self._b, self._seed = A, b, seed

    def _context(self, context) -> np.ndarray:
        return _unit(_array(context, (self.dimension,)))

    def _validate_state(self, A, b) -> None:
        if not np.isfinite(A).all() or not np.isfinite(b).all():
            raise ValueError("Nonfinite model state")
        if not np.array_equal(A, A.swapaxes(-1, -2)):
            raise ValueError("Model matrices must be symmetric")
        try:
            np.linalg.cholesky(A)
            with np.errstate(over="ignore", invalid="ignore"):
                theta = np.linalg.solve(A, b[..., None])[..., 0]
        except np.linalg.LinAlgError as exc:
            raise ValueError("Model matrices must be positive definite") from exc
        if not np.isfinite(theta).all():
            raise ValueError("Nonfinite model estimates")

    def _estimates(self, context) -> tuple[np.ndarray, np.ndarray]:
        x = self._context(context)
        try:
            with np.errstate(over="ignore", invalid="ignore"):
                theta = np.linalg.solve(self._A, self._b[..., None])[..., 0]
                directions = np.linalg.solve(self._A, np.broadcast_to(x, self._b.shape)[..., None])[..., 0]
                variance = directions @ x
        except np.linalg.LinAlgError as exc:
            raise ValueError("Invalid ridge state") from exc
        if (not np.isfinite(theta).all() or not np.isfinite(variance).all()
                or (variance < -1e-10).any()):
            raise ValueError("Nonfinite prediction or invalid uncertainty")
        means = np.array([_unit(t, allow_zero=True) @ x for t in theta])
        return np.clip(means, -1, 1), np.sqrt(np.maximum(variance, 0))

    def save_checkpoint(self, path: str | Path) -> None:
        """Save this model exclusively; a checksum detects accidental alteration."""
        self._validate_state(self._A, self._b)
        payload = {"version": 1, "arms": list(self.arms), "dimension": self.dimension,
                   "alpha": self.alpha, "provenance": self.provenance,
                   "priors": dict(zip(self.arms, self._priors.tolist())),
                   "lambdas": dict(zip(self.arms, self._lambdas.tolist())),
                   "seed": self._seed, "rounds": self.rounds,
                   "A": self._A.tolist(), "b": self._b.tolist()}
        encoded = _canonical({"payload": payload, "sha256": _digest(payload)}) + "\n"
        with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(encoded)

    @classmethod
    def load_checkpoint(cls, path: str | Path) -> PILOT:
        """Return a validated new model; never mutate an existing model while loading."""
        try:
            document = json.loads(Path(path).read_text(encoding="utf-8"),
                                  object_pairs_hook=_unique_object)
            if not isinstance(document, dict) or set(document) != {"payload", "sha256"}:
                raise ValueError("Invalid checkpoint envelope")
            payload = document["payload"]
            fields = {"version", "arms", "dimension", "alpha", "provenance", "priors",
                      "lambdas", "seed", "rounds", "A", "b"}
            if not isinstance(payload, dict) or set(payload) != fields:
                raise ValueError("Invalid checkpoint schema")
            if document["sha256"] != _digest(payload):
                raise ValueError("Checkpoint checksum mismatch")
            if type(payload["version"]) is not int or payload["version"] != 1:
                raise ValueError("Unsupported checkpoint version")
            if not isinstance(payload["arms"], list):
                raise ValueError("Catalog must be an ordered list")
            for field in ("seed", "rounds"):
                if type(payload[field]) is not int or payload[field] < 0:
                    raise ValueError("Invalid checkpoint counter or seed")
            model = cls(payload["arms"], payload["dimension"], priors=payload["priors"],
                        lambdas=payload["lambdas"], provenance=payload["provenance"],
                        alpha=payload["alpha"])
            model.reset(payload["seed"])
            A = _array(payload["A"], model._A.shape)
            b = _array(payload["b"], model._b.shape)
            model._validate_state(A, b)
            model._A, model._b, model.rounds = A, b, payload["rounds"]
            return model
        except (TypeError, KeyError, OverflowError) as exc:
            raise ValueError("Malformed checkpoint") from exc
