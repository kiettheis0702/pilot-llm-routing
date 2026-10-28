"""Explicit-input NumPy preference mathematics; execution evidence is synthetic only."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def _digest(value):
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _schema(value, fields, label):
    if not isinstance(value, dict) or set(value) != set(fields):
        raise ValueError(f"Invalid {label} schema")


def _names(values, label, *, unique=True):
    if (not isinstance(values, (list, tuple)) or not values
            or any(not isinstance(v, str) or not v.strip() for v in values)
            or (unique and len(set(values)) != len(values))):
        raise ValueError(f"Invalid {label}: expected nonempty unique names")
    return tuple(values)


def _hash(value):
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError("Invalid SHA-256 fingerprint")
    return value


def _hashes(values):
    if not isinstance(values, (list, tuple)) or not values:
        raise ValueError("Prompt hashes must be an ordered nonempty sequence")
    return tuple(_hash(value) for value in values)


def _integer(value, label, minimum=1):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


def _positive(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or value <= 0:
        raise ValueError(f"{label} must be finite and positive")
    return float(value)


def _array(value, *, shape=None, ndim=None):
    try:
        # Inspect nested input before NumPy can coerce mixed bool/float lists.
        if not isinstance(value, np.ndarray) and any(isinstance(v, (bool, np.bool_))
                for v in np.asarray(value, dtype=object).flat):
            raise ValueError("Boolean numeric elements are not accepted")
        raw = np.asarray(value)
        if raw.dtype.kind not in "iuf":
            raise ValueError("Expected real numbers, without coercing strings or booleans")
        result = np.array(raw, dtype=np.float64, copy=True)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Invalid numeric array") from exc
    if (not result.size or not np.isfinite(result).all() or (shape is not None and result.shape != shape)
            or (ndim is not None and result.ndim != ndim)):
        raise ValueError("Nonfinite values or invalid array dimensions")
    return result


def _recipe(value):
    _schema(value, {"model", "version", "source", "normalization", "synthetic"}, "embedding provenance")
    if any(not isinstance(value[k], str) or not value[k].strip() for k in ("model", "version", "source", "normalization")):
        raise ValueError("Embedding provenance fields must be nonempty")
    if type(value["synthetic"]) is not bool:
        raise ValueError("Embedding provenance must explicitly label synthetic status")
    return dict(value)


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON key")
        result[key] = value
    return result


def _write_json(path, payload):
    encoded = _canonical({"payload": payload, "sha256": _digest(payload)}) + "\n"
    if len(encoded.encode("utf-8")) > 16 * 1024 * 1024:
        raise ValueError("Artifact exceeds the 16 MiB small-core JSON limit")
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(encoded)


def _read_json(path):
    path = Path(path)
    if path.stat().st_size > 16 * 1024 * 1024:
        raise ValueError("Artifact exceeds the 16 MiB small-core JSON limit")
    try:
        document = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique_object,
                              parse_constant=lambda _: (_ for _ in ()).throw(ValueError("Nonfinite JSON number")))
        _schema(document, {"payload", "sha256"}, "artifact envelope")
        if document["sha256"] != _digest(document["payload"]):
            raise ValueError("Artifact checksum mismatch")
        return document["payload"]
    except (TypeError, OverflowError, RecursionError) as exc:
        raise ValueError("Malformed JSON artifact") from exc


class EmbeddingCache:
    """Caller-supplied vectors with complete declared recipe; no embedding generation."""

    def __init__(self, ids, prompt_hashes, vectors, provenance):
        self.ids = _names(ids, "query IDs")
        self.prompt_hashes = _hashes(prompt_hashes)
        self._vectors = _array(vectors, ndim=2)
        if len(self.prompt_hashes) != len(self.ids) or self._vectors.shape[0] != len(self.ids):
            raise ValueError("Embedding dimension/ID/hash alignment mismatch")
        self._provenance = _recipe(provenance)
        self.dimension = int(self._vectors.shape[1])
        self._index = {q: i for i, q in enumerate(self.ids)}

    @property
    def vectors(self):
        return self._vectors.copy()

    @property
    def provenance(self):
        return dict(self._provenance)

    def get(self, query_id):
        if query_id not in self._index:
            raise ValueError("Unknown embedding query ID")
        return self._vectors[self._index[query_id]].copy()

    def _payload(self):
        return {"version": 1, "kind": "embedding-cache", "query_ids": list(self.ids),
                "prompt_hashes": list(self.prompt_hashes), "dimension": self.dimension,
                "vectors": self._vectors.tolist(), "provenance": self.provenance}

    def fingerprint(self):
        return _digest(self._payload())

    def save(self, path):
        _write_json(path, self._payload())

    @classmethod
    def load(cls, path):
        payload = _read_json(path)
        _schema(payload, {"version", "kind", "query_ids", "prompt_hashes", "dimension", "vectors", "provenance"}, "cache")
        if type(payload["version"]) is not int or payload["version"] != 1 or payload["kind"] != "embedding-cache":
            raise ValueError("Unsupported cache version/kind")
        result = cls(payload["query_ids"], payload["prompt_hashes"], payload["vectors"], payload["provenance"])
        if _integer(payload["dimension"], "dimension") != result.dimension:
            raise ValueError("Cache dimension mismatch")
        return result


def validate_comparisons(cache, comparisons, arms):
    arms = _names(arms, "arm catalog")
    if not isinstance(cache, EmbeddingCache) or not isinstance(comparisons, (list, tuple)) or not comparisons:
        raise ValueError("Expected cache and nonempty comparisons")
    rows, battles, group_splits = [], set(), {}
    prompt_hashes = dict(zip(cache.ids, cache.prompt_hashes))
    support = {split: set() for split in ("train", "holdout")}
    for row in comparisons:
        _schema(row, {"battle_id", "query_id", "winner", "loser", "split"}, "comparison")
        if not all(isinstance(value, str) and value.strip() for value in row.values()):
            raise ValueError("Comparison fields must be nonempty strings")
        if row["battle_id"] in battles:
            raise ValueError("Duplicate battle identity")
        battles.add(row["battle_id"])
        if row["query_id"] not in prompt_hashes or row["winner"] not in arms or row["loser"] not in arms or row["winner"] == row["loser"]:
            raise ValueError("Unknown query/arm or self-comparison")
        if row["split"] not in support:
            raise ValueError("Split must be train or holdout")
        group = prompt_hashes[row["query_id"]]
        if group in group_splits and group_splits[group] != row["split"]:
            raise ValueError("Prompt group leaks across train and holdout")
        group_splits[group] = row["split"]
        support[row["split"]].update((row["winner"], row["loser"]))
        rows.append(dict(row))
    if any(pool != set(arms) for pool in support.values()):
        raise ValueError("Every arm needs comparisons in both train and holdout")
    return sorted(rows, key=lambda row: row["battle_id"])


def _sizes(arms, sizes, provenance):
    if not isinstance(sizes, dict) or set(sizes) != set(arms):
        raise ValueError("Token-cost size order must cover the exact arm catalog")
    if not isinstance(provenance, str) or not provenance.strip():
        raise ValueError("Token-cost size provenance is required")
    return {a: _positive(sizes[a], "token-cost size") for a in arms}


def sample_triplets(cache, comparisons, arms, sizes, size_provenance, seed=42):
    rows = validate_comparisons(cache, comparisons, arms)
    sizes = _sizes(arms, sizes, size_provenance)
    rng = np.random.default_rng(_integer(seed, "seed", 0))
    train = [row for row in rows if row["split"] == "train"]
    hashes = dict(zip(cache.ids, cache.prompt_hashes))
    triplets = []
    for anchor in train:
        aq, winner = anchor["query_id"], anchor["winner"]
        positive = [row for row in train if row["winner"] == winner and hashes[row["query_id"]] != hashes[aq]]
        negative = [row for row in train if row["loser"] == winner and sizes[row["winner"]] < sizes[winner]
                    and hashes[row["query_id"]] != hashes[aq]]
        choices = [(p["query_id"], n["query_id"]) for p in positive for n in negative
                   if hashes[p["query_id"]] != hashes[n["query_id"]]]
        if choices:
            pq, nq = choices[int(rng.integers(len(choices)))]
            triplets.append((aq, pq, nq))
    if not triplets:
        raise ValueError("No valid preference triplets under the supplied strict size order")
    return {"triplets": triplets, "eligible_anchors": len(triplets), "unusable_anchors": len(train) - len(triplets),
            "train_anchors": len(train), "seed": seed, "size_provenance": size_provenance}


def _unit(vector):
    vector = _array(vector, ndim=1)
    scale = float(np.max(np.abs(vector)))
    if scale == 0:
        raise ValueError("Cosine requires a nonzero direction")
    scaled = vector / scale
    norm = scale * float(np.linalg.norm(scaled))
    if not np.isfinite(norm) or norm <= 1e-12:
        raise ValueError("Cosine norm is nonfinite or too close to zero")
    return scaled / np.linalg.norm(scaled), norm


def _cosine_grad(left, right):
    u, nu = _unit(left)
    v, nv = _unit(right)
    if u.shape != v.shape:
        raise ValueError("Cosine dimensions disagree")
    cosine = float(u @ v)
    return cosine, (v - cosine * u) / nu, (u - cosine * v) / nv


def _projection(W, b, input_dimension):
    W = _array(W, ndim=2)
    if W.shape[1] != input_dimension:
        raise ValueError("Projection input dimension mismatch")
    return W, _array(b, shape=(W.shape[0],))


def triplet_loss_grad(W, b, anchor, positive, negative, margin=.2):
    anchor = _array(anchor, ndim=1)
    positive = _array(positive, shape=anchor.shape)
    negative = _array(negative, shape=anchor.shape)
    W, b = _projection(W, b, len(anchor))
    margin = _positive(margin, "margin")
    za, zp, zn = (W @ x + b for x in (anchor, positive, negative))
    ap, da_p, dp = _cosine_grad(za, zp)
    an, da_n, dn = _cosine_grad(za, zn)
    loss = margin - ap + an
    if loss <= 0:
        return 0., np.zeros_like(W), np.zeros_like(b)
    ga, gp, gn = da_n - da_p, -dp, dn
    return float(loss), np.outer(ga, anchor) + np.outer(gp, positive) + np.outer(gn, negative), ga + gp + gn


def pairwise_loss_grad(context, theta, winner, loser):
    context = _array(context, ndim=1)
    theta = _array(theta, ndim=2)
    if theta.shape[1] != len(context) or type(winner) is not int or type(loser) is not int or winner == loser or not (0 <= winner < len(theta) and 0 <= loser < len(theta)):
        raise ValueError("Invalid pair indices or dimensions")
    sw, _, gw = _cosine_grad(context, theta[winner])
    sl, _, gl = _cosine_grad(context, theta[loser])
    probability_loser = 1. / (1. + np.exp(sw - sl))
    gradient = np.zeros_like(theta)
    gradient[winner], gradient[loser] = -probability_loser * gw, probability_loser * gl
    return float(np.logaddexp(0., sl - sw)), gradient


def train_projection(cache, comparisons, arms, sizes, size_provenance, *, dimension=2,
                     epochs=30, learning_rate=.03, margin=.2, seed=42):
    _integer(dimension, "projection dimension")
    _integer(epochs, "epochs")
    rate = _positive(learning_rate, "learning rate")
    _positive(margin, "margin")
    sampled = sample_triplets(cache, comparisons, arms, sizes, size_provenance, seed)
    rng = np.random.default_rng(seed)
    W = rng.normal(size=(dimension, cache.dimension)) / np.sqrt(cache.dimension)
    b = np.zeros(dimension)
    triplets = [tuple(cache.get(q) for q in ids) for ids in sampled["triplets"]]
    def objective():
        return float(np.mean([triplet_loss_grad(W, b, *vectors, margin=margin)[0] for vectors in triplets]))
    losses = [objective()]
    for _ in range(epochs):
        for index in rng.permutation(len(triplets)):
            _, dW, db = triplet_loss_grad(W, b, *triplets[index], margin=margin)
            W, b = _array(W - rate * dW, shape=W.shape), _array(b - rate * db, shape=b.shape)
        losses.append(objective())
    return {"W": W, "b": b, "losses": losses, "sampling": sampled}


def train_prior(cache, comparisons, arms, W, b, *, epochs=40, learning_rate=.03, seed=42):
    rows = validate_comparisons(cache, comparisons, arms)
    W, b = _projection(W, b, cache.dimension)  # Copies: these never receive an update.
    _integer(epochs, "epochs")
    rate = _positive(learning_rate, "learning rate")
    rng = np.random.default_rng(_integer(seed, "seed", 0))
    theta = rng.normal(size=(len(arms), W.shape[0]))
    positions = {a: i for i, a in enumerate(arms)}
    examples = [(W @ cache.get(row["query_id"]) + b, positions[row["winner"]], positions[row["loser"]])
                for row in rows if row["split"] == "train"]
    def objective():
        return float(np.mean([pairwise_loss_grad(x, theta, win, lose)[0] for x, win, lose in examples]))
    losses = [objective()]
    for _ in range(epochs):
        for index in rng.permutation(len(examples)):
            x, win, lose = examples[index]
            _, gradient = pairwise_loss_grad(x, theta, win, lose)
            theta = _array(theta - rate * gradient, shape=theta.shape)
        losses.append(objective())
    return {"theta": theta, "losses": losses}


def inverse_accuracy(statistics):
    if not isinstance(statistics, dict) or not statistics:
        raise ValueError("Per-arm accuracy support is required")
    _names(list(statistics), "accuracy arms")
    metrics, blockers = {}, {}
    for arm, counts in statistics.items():
        _schema(counts, {"support", "correct"}, "accuracy")
        support = _integer(counts["support"], "support", 0)
        correct = _integer(counts["correct"], "correct", 0)
        if correct > support:
            raise ValueError("Correct count exceeds support")
        accuracy = correct / support if support else None
        metrics[arm] = dict(counts, accuracy=accuracy, **{"lambda": support / correct if correct else None})
        if not support:
            blockers[arm] = "No held-out support"
        elif not correct:
            blockers[arm] = "Zero held-out accuracy; inverse undefined"
    return {"per_arm": metrics, "all_lambdas_ready": not blockers, "blockers": blockers}


def evaluate_prior(cache, comparisons, arms, W, b, theta):
    rows = validate_comparisons(cache, comparisons, arms)
    W, b = _projection(W, b, cache.dimension)
    theta = _array(theta, shape=(len(arms), W.shape[0]))
    positions = {a: i for i, a in enumerate(arms)}
    statistics = {arm: {"support": 0, "correct": 0} for arm in arms}
    for row in rows:
        if row["split"] != "holdout":
            continue
        context = W @ cache.get(row["query_id"]) + b
        sw = _cosine_grad(context, theta[positions[row["winner"]]])[0]
        sl = _cosine_grad(context, theta[positions[row["loser"]]])[0]
        for arm in (row["winner"], row["loser"]):
            statistics[arm]["support"] += 1
            statistics[arm]["correct"] += int(sw > sl)
    return inverse_accuracy(statistics)


def fit_preference_prior(cache, comparisons, arms, sizes, size_provenance, *, dimension=2,
                         projection_epochs=30, prior_epochs=40, learning_rate=.03, margin=.2, seed=42):
    rows = validate_comparisons(cache, comparisons, arms)
    projection = train_projection(cache, rows, arms, sizes, size_provenance, dimension=dimension,
                                  epochs=projection_epochs, learning_rate=learning_rate, margin=margin, seed=seed)
    prior = train_prior(cache, rows, arms, projection["W"], projection["b"], epochs=prior_epochs,
                        learning_rate=learning_rate, seed=seed)
    validation = evaluate_prior(cache, rows, arms, projection["W"], projection["b"], prior["theta"])
    return {"version": 1, "kind": "preference-prior", "arms": list(arms), "dimension": dimension,
            "input_dimension": cache.dimension, "W": projection["W"], "b": projection["b"], "theta": prior["theta"],
            "embedding": {"recipe": cache.provenance, "query_ids": list(cache.ids),
                          "prompt_hashes": list(cache.prompt_hashes), "cache_sha256": cache.fingerprint()},
            "metrics": {"projection_losses": projection["losses"], "prior_losses": prior["losses"], "validation": validation},
            "protocol": {"interpretation": "PAPER_INFERRED", "seed": seed, "projection_epochs": projection_epochs,
                         "prior_epochs": prior_epochs, "learning_rate": learning_rate, "margin": margin,
                         "optimizer": "seeded shuffled SGD", "triplets_per_eligible_anchor": 1,
                         "distinct_prompt_members": True, "prediction_ties": "incorrect"},
            "data": {"comparisons_sha256": _digest(rows),
                     "split_sha256": _digest([{k: row[k] for k in ("battle_id", "query_id", "split")} for row in rows]),
                     "cost_sizes": _sizes(arms, sizes, size_provenance), "cost_size_provenance": size_provenance,
                     "sampled_triplets": [list(t) for t in projection["sampling"]["triplets"]]}}


def _validate_checkpoint(result):
    _schema(result, {"version", "kind", "arms", "dimension", "input_dimension", "W", "b", "theta", "embedding", "metrics", "protocol", "data"}, "checkpoint")
    if type(result["version"]) is not int or result["version"] != 1 or result["kind"] != "preference-prior":
        raise ValueError("Unsupported checkpoint version/kind")
    arms = _names(result["arms"], "arm catalog")
    dimension = _integer(result["dimension"], "dimension")
    inputs = _integer(result["input_dimension"], "input dimension")
    W, b = _projection(result["W"], result["b"], inputs)
    if W.shape[0] != dimension:
        raise ValueError("Checkpoint projection dimension mismatch")
    if not W.any() and not b.any():
        raise ValueError("All-zero projection cannot define any cosine direction")
    theta = _array(result["theta"], shape=(len(arms), dimension))
    for vector in theta:
        _unit(vector)
    embedding = result["embedding"]
    _schema(embedding, {"recipe", "query_ids", "prompt_hashes", "cache_sha256"}, "checkpoint embedding")
    _recipe(embedding["recipe"])
    ids = _names(embedding["query_ids"], "query IDs")
    hashes = _hashes(embedding["prompt_hashes"])
    if len(hashes) != len(ids):
        raise ValueError("Checkpoint ID/hash dimension mismatch")
    _hash(embedding["cache_sha256"])
    metrics = result["metrics"]
    _schema(metrics, {"projection_losses", "prior_losses", "validation"}, "metrics")
    for name in ("projection_losses", "prior_losses"):
        losses = _array(metrics[name], ndim=1)
        if len(losses) < 2 or (losses < 0).any():
            raise ValueError("Invalid training loss history")
    validation = metrics["validation"]
    _schema(validation, {"per_arm", "all_lambdas_ready", "blockers"}, "validation")
    if type(validation["all_lambdas_ready"]) is not bool:
        raise ValueError("Lambda readiness must be Boolean")
    if not isinstance(validation["per_arm"], dict) or set(validation["per_arm"]) != set(arms):
        raise ValueError("Validation catalog mismatch")
    statistics = {}
    for arm, values in validation["per_arm"].items():
        _schema(values, {"support", "correct", "accuracy", "lambda"}, "arm validation")
        for field in ("accuracy", "lambda"):
            if values[field] is not None and (isinstance(values[field], bool)
                    or not isinstance(values[field], (int, float)) or not np.isfinite(values[field])):
                raise ValueError("Accuracy/lambda must be finite numbers or unavailable")
        statistics[arm] = {k: values[k] for k in ("support", "correct")}
    if validation != inverse_accuracy(statistics):
        raise ValueError("Inconsistent validation accuracy/lambda diagnostics")
    protocol = result["protocol"]
    _schema(protocol, {"interpretation", "seed", "projection_epochs", "prior_epochs", "learning_rate", "margin", "optimizer", "triplets_per_eligible_anchor", "distinct_prompt_members", "prediction_ties"}, "protocol")
    if (protocol["interpretation"] != "PAPER_INFERRED" or protocol["optimizer"] != "seeded shuffled SGD"
            or type(protocol["triplets_per_eligible_anchor"]) is not int or protocol["triplets_per_eligible_anchor"] != 1
            or protocol["distinct_prompt_members"] is not True
            or protocol["prediction_ties"] != "incorrect"):
        raise ValueError("Unsupported training protocol")
    _integer(protocol["seed"], "seed", 0)
    for field in ("projection_epochs", "prior_epochs"):
        _integer(protocol[field], field)
    for field in ("learning_rate", "margin"):
        _positive(protocol[field], field)
    if len(metrics["projection_losses"]) != protocol["projection_epochs"] + 1 or len(metrics["prior_losses"]) != protocol["prior_epochs"] + 1:
        raise ValueError("Loss history and epoch count mismatch")
    data = result["data"]
    _schema(data, {"comparisons_sha256", "split_sha256", "cost_sizes", "cost_size_provenance", "sampled_triplets"}, "data provenance")
    for field in ("comparisons_sha256", "split_sha256"):
        _hash(data[field])
    _sizes(arms, data["cost_sizes"], data["cost_size_provenance"])
    by_id = dict(zip(ids, hashes))
    if not isinstance(data["sampled_triplets"], list) or not data["sampled_triplets"]:
        raise ValueError("Missing sampled triplets")
    for triplet in data["sampled_triplets"]:
        if not isinstance(triplet, list) or len(triplet) != 3 or any(not isinstance(q, str) or q not in by_id for q in triplet) or len({by_id[q] for q in triplet}) != 3:
            raise ValueError("Invalid triplet prompt identities")
    cleaned = copy.deepcopy(result)
    cleaned.update(W=W, b=b, theta=theta)
    return cleaned


def save_checkpoint(path, result):
    payload = _validate_checkpoint(result)
    for field in ("W", "b", "theta"):
        payload[field] = payload[field].tolist()
    _write_json(path, payload)


def load_checkpoint(path):
    return _validate_checkpoint(_read_json(path))


def project_cache(checkpoint, cache):
    checkpoint = _validate_checkpoint(checkpoint)
    if cache.provenance != checkpoint["embedding"]["recipe"] or cache.dimension != checkpoint["input_dimension"]:
        raise ValueError("Embedding recipe/dimension differs from the trained projection")
    result = cache.vectors @ checkpoint["W"].T + checkpoint["b"]
    for vector in result:
        _unit(vector)
    return result
