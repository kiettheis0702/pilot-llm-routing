import copy
import hashlib
import json

import numpy as np
import pytest

from scripts import pretraining as pre


ARMS = ["small", "large"]
SIZES = {"small": 1.0, "large": 4.0}
SOURCE = "synthetic token-cost units v1; not actual model prices"
RECIPE = {"model": "toy-supplied-vectors", "version": "1", "source": "analytic fixture",
          "normalization": "none", "synthetic": True}


def fixture():
    ids = [f"q{i}" for i in range(8)]
    hashes = [hashlib.sha256(q.encode()).hexdigest() for q in ids]
    # Shared nuisance coordinate initially masks the winner-discriminating x axis.
    vectors = np.array([[1., 3.], [.9, 2.9], [-1., 3.], [-.9, 3.1],
                        [1.1, 3.2], [-1.1, 2.8], [1., 3.05], [-1., 2.95]])
    cache = pre.EmbeddingCache(ids, hashes, vectors, RECIPE)
    winners = ["large", "large", "small", "small", "large", "small", "large", "small"]
    comparisons = [{"battle_id": f"b{i}", "query_id": ids[i], "winner": winner,
                    "loser": "small" if winner == "large" else "large",
                    "split": "train" if i < 6 else "holdout"} for i, winner in enumerate(winners)]
    return cache, comparisons


def test_cache_copies_supplied_vectors_and_json_readback_is_exact(tmp_path):
    ids = ["q"]
    hashes = ["a" * 64]
    vectors = np.array([[1., 2.]])
    cache = pre.EmbeddingCache(ids, hashes, vectors, RECIPE)
    vectors[:] = 9
    assert cache.vectors.tolist() == [[1., 2.]]
    path = tmp_path / "cache.json"
    cache.save(path)
    restored = pre.EmbeddingCache.load(path)
    np.testing.assert_array_equal(restored.vectors, cache.vectors)
    assert restored.fingerprint() == cache.fingerprint()
    assert restored.provenance == RECIPE
    with pytest.raises(FileExistsError):
        cache.save(path)


@pytest.mark.parametrize("change", ["duplicate", "dimension", "nan", "hash", "provenance", "numeric_strings", "mixed_boolean"])
def test_cache_rejects_untraceable_or_malformed_vectors(change):
    ids, hashes, vectors, recipe = ["q", "r"], ["a" * 64, "b" * 64], [[1., 2.], [2., 3.]], dict(RECIPE)
    if change == "duplicate": ids[1] = "q"
    if change == "dimension": vectors[1] = [3.]
    if change == "nan": vectors[1][0] = float("nan")
    if change == "hash": hashes[0] = "not a hash"
    if change == "provenance": recipe.pop("version")
    if change == "numeric_strings": vectors[0][0] = "1"
    if change == "mixed_boolean": vectors[0][0] = True
    with pytest.raises(ValueError):
        pre.EmbeddingCache(ids, hashes, vectors, recipe)


def test_triplets_use_train_distinct_prompts_and_strictly_smaller_negative():
    cache, comparisons = fixture()
    sampled = pre.sample_triplets(cache, comparisons, ARMS, SIZES, SOURCE, seed=42)
    assert sampled["eligible_anchors"] == 3
    assert sampled["unusable_anchors"] == 3
    for anchor, positive, negative in sampled["triplets"]:
        assert anchor in {"q0", "q1", "q4"}
        assert positive in {"q0", "q1", "q4"} - {anchor}
        assert negative in {"q2", "q3", "q5"}
    assert sampled == pre.sample_triplets(cache, list(reversed(comparisons)), ARMS, SIZES, SOURCE, seed=42)
    with pytest.raises(ValueError, match="triplet"):
        pre.sample_triplets(cache, comparisons, ARMS, {"small": 4., "large": 4.}, SOURCE)
    with pytest.raises(ValueError, match="size"):
        pre.sample_triplets(cache, comparisons, ARMS, {}, SOURCE)
    with pytest.raises(ValueError, match="provenance"):
        pre.sample_triplets(cache, comparisons, ARMS, SIZES, "")


def test_triplet_sampling_excludes_same_prompt_even_with_different_query_id():
    cache, comparisons = fixture()
    hashes = list(cache.prompt_hashes)
    hashes[1] = hashes[0]
    duplicate = pre.EmbeddingCache(cache.ids, hashes, cache.vectors, RECIPE)
    result = pre.sample_triplets(duplicate, comparisons, ARMS, SIZES, SOURCE)
    by_id = dict(zip(duplicate.ids, duplicate.prompt_hashes))
    assert all(len({by_id[q] for q in triplet}) == 3 for triplet in result["triplets"])


@pytest.mark.parametrize("change", ["prompt_leak", "query_leak", "unknown_arm", "self_pair", "duplicate_battle", "missing_holdout", "missing_arm"])
def test_training_contract_rejects_leakage_or_incomplete_pool(change):
    cache, comparisons = fixture()
    if change == "prompt_leak":
        hashes = list(cache.prompt_hashes); hashes[6] = hashes[0]
        cache = pre.EmbeddingCache(cache.ids, hashes, cache.vectors, RECIPE)
    if change == "query_leak": comparisons[6]["query_id"] = "q0"
    if change == "unknown_arm": comparisons[0]["winner"] = "unknown"
    if change == "self_pair": comparisons[0]["loser"] = comparisons[0]["winner"]
    if change == "duplicate_battle": comparisons[1]["battle_id"] = comparisons[0]["battle_id"]
    if change == "missing_holdout": comparisons = comparisons[:6]
    arms = ARMS + ["missing"] if change == "missing_arm" else ARMS
    with pytest.raises(ValueError):
        pre.validate_comparisons(cache, comparisons, arms)


def finite_difference(value, objective, epsilon=1e-6):
    result = np.zeros_like(value)
    for index in np.ndindex(value.shape):
        plus, minus = value.copy(), value.copy()
        plus[index] += epsilon; minus[index] -= epsilon
        result[index] = (objective(plus) - objective(minus)) / (2 * epsilon)
    return result


def test_triplet_gradient_matches_finite_differences_for_projection_and_bias():
    W = np.array([[.8, .3], [-.2, .7]])
    b = np.array([.1, -.1])
    anchor, positive, negative = np.array([1., .2]), np.array([-.3, 1.]), np.array([.8, .1])
    loss, gW, gb = pre.triplet_loss_grad(W, b, anchor, positive, negative, margin=.5)
    assert loss > .1  # Strictly inside the active hinge, away from a kink.
    np.testing.assert_allclose(gW, finite_difference(W, lambda x: pre.triplet_loss_grad(x, b, anchor, positive, negative, margin=.5)[0]), rtol=1e-5, atol=1e-7)
    np.testing.assert_allclose(gb, finite_difference(b, lambda x: pre.triplet_loss_grad(W, x, anchor, positive, negative, margin=.5)[0]), rtol=1e-5, atol=1e-7)


def test_inactive_triplet_has_zero_loss_and_gradients():
    loss, gW, gb = pre.triplet_loss_grad(np.eye(2), np.zeros(2), [1., 0.], [1., .1], [-1., 0.], margin=.2)
    assert loss == 0
    assert not gW.any() and not gb.any()


def test_pairwise_softmax_bce_gradient_and_unchosen_arm_are_correct():
    theta = np.array([[.4, .8], [-.3, .6], [.9, -.5]])
    context = np.array([.8, -.2])
    loss, gradient = pre.pairwise_loss_grad(context, theta, winner=0, loser=1)
    sw = context @ theta[0] / (np.linalg.norm(context) * np.linalg.norm(theta[0]))
    sl = context @ theta[1] / (np.linalg.norm(context) * np.linalg.norm(theta[1]))
    assert loss == pytest.approx(-np.log(np.exp(sw) / (np.exp(sw) + np.exp(sl))))
    np.testing.assert_allclose(gradient, finite_difference(theta, lambda x: pre.pairwise_loss_grad(context, x, winner=0, loser=1)[0]), rtol=1e-5, atol=1e-7)
    np.testing.assert_array_equal(gradient[2], [0., 0.])


@pytest.mark.parametrize("vector", [[0., 0.], [1e-15, 0.], [float("nan"), 1.], [float("inf"), 1.]])
def test_cosine_math_rejects_undefined_directions(vector):
    with pytest.raises(ValueError):
        pre.pairwise_loss_grad(vector, np.eye(2), winner=0, loser=1)


def test_tiny_training_decreases_losses_is_seeded_and_freezes_projection():
    cache, comparisons = fixture()
    projection = pre.train_projection(cache, comparisons, ARMS, SIZES, SOURCE, dimension=2, epochs=30, learning_rate=.03, margin=.5, seed=7)
    assert projection["losses"][-1] < projection["losses"][0]
    W, b = projection["W"].copy(), projection["b"].copy()
    prior = pre.train_prior(cache, comparisons, ARMS, W, b, epochs=40, learning_rate=.05, seed=7)
    assert prior["losses"][-1] < prior["losses"][0]
    np.testing.assert_array_equal(W, projection["W"])
    np.testing.assert_array_equal(b, projection["b"])
    again = pre.train_prior(cache, list(reversed(comparisons)), ARMS, W, b, epochs=40, learning_rate=.05, seed=7)
    np.testing.assert_array_equal(prior["theta"], again["theta"])


def test_holdout_values_and_labels_never_change_fitted_parameters():
    cache, comparisons = fixture()
    settings = dict(dimension=2, projection_epochs=8, prior_epochs=8, learning_rate=.03, margin=.5, seed=7)
    result = pre.fit_preference_prior(cache, comparisons, ARMS, SIZES, SOURCE, **settings)
    vectors = cache.vectors
    vectors[6:] *= -3
    altered_cache = pre.EmbeddingCache(cache.ids, cache.prompt_hashes, vectors, RECIPE)
    altered = copy.deepcopy(comparisons)
    for row in altered[6:]: row["winner"], row["loser"] = row["loser"], row["winner"]
    other = pre.fit_preference_prior(altered_cache, altered, ARMS, SIZES, SOURCE, **settings)
    for field in ("W", "b", "theta"):
        np.testing.assert_array_equal(result[field], other[field])


def test_inverse_accuracy_blocks_zero_and_absent_without_floor():
    result = pre.inverse_accuracy({"good": {"support": 4, "correct": 3},
                                   "zero": {"support": 2, "correct": 0},
                                   "absent": {"support": 0, "correct": 0}})
    assert result["per_arm"]["good"]["lambda"] == pytest.approx(4 / 3)
    assert result["per_arm"]["zero"]["lambda"] is None
    assert result["per_arm"]["absent"]["accuracy"] is None
    assert result["all_lambdas_ready"] is False
    assert set(result["blockers"]) == {"zero", "absent"}


def test_heldout_accuracy_counts_each_pair_for_its_two_arms_and_ties_fail():
    cache, comparisons = fixture()
    metrics = pre.evaluate_prior(cache, comparisons, ARMS, np.eye(2), np.zeros(2), np.array([[-1., 0.], [1., 0.]]))
    assert metrics["per_arm"]["small"] == {"support": 2, "correct": 2, "accuracy": 1., "lambda": 1.}
    assert metrics["all_lambdas_ready"] is True
    ties = pre.evaluate_prior(cache, comparisons, ARMS, np.eye(2), np.zeros(2), np.array([[1., 0.], [1., 0.]]))
    assert ties["per_arm"]["small"]["correct"] == 0
    assert ties["all_lambdas_ready"] is False


def test_checkpoint_reload_preserves_parameters_predictions_and_provenance(tmp_path):
    cache, comparisons = fixture()
    result = pre.fit_preference_prior(cache, comparisons, ARMS, SIZES, SOURCE, dimension=2, projection_epochs=5, prior_epochs=5, seed=7)
    path = tmp_path / "prior.json"
    pre.save_checkpoint(path, result)
    restored = pre.load_checkpoint(path)
    for field in ("W", "b", "theta"):
        np.testing.assert_array_equal(restored[field], result[field])
    np.testing.assert_array_equal(pre.project_cache(restored, cache), pre.project_cache(result, cache))
    assert restored["embedding"]["cache_sha256"] == cache.fingerprint()
    assert restored["embedding"]["query_ids"] == list(cache.ids)
    assert restored["protocol"]["interpretation"] == "PAPER_INFERRED"
    with pytest.raises(FileExistsError): pre.save_checkpoint(path, result)
    wrong = pre.EmbeddingCache(cache.ids, cache.prompt_hashes, cache.vectors, RECIPE | {"version": "2"})
    with pytest.raises(ValueError, match="recipe"): pre.project_cache(restored, wrong)
    document = json.loads(path.read_text())
    document["payload"]["theta"][0][0] += 1
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="checksum"): pre.load_checkpoint(path)


def test_json_cache_rejects_duplicate_keys_and_structural_tampering(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text('{"payload": {}, "payload": {}, "sha256": "x"}')
    with pytest.raises(ValueError, match="Duplicate"): pre.EmbeddingCache.load(path)
    cache, _ = fixture()
    valid = tmp_path / "valid.json"; cache.save(valid)
    document = json.loads(valid.read_text())
    document["payload"]["dimension"] = 99
    document["sha256"] = hashlib.sha256(json.dumps(document["payload"], sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()).hexdigest()
    valid.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="dimension"): pre.EmbeddingCache.load(valid)


@pytest.mark.parametrize("hashes", [None, {"a" * 64: 1}, "a" * 64])
def test_cache_requires_an_ordered_hash_sequence(hashes):
    with pytest.raises(ValueError):
        pre.EmbeddingCache(["q"], hashes, [[1., 2.]], RECIPE)


@pytest.mark.parametrize("mutation", ["zero_projection", "boolean_accuracy", "boolean_lambda", "boolean_triplet_count", "integer_ready"])
def test_checkpoint_rejects_invalid_state_even_with_recomputed_checksum(tmp_path, mutation):
    cache, comparisons = fixture()
    result = pre.fit_preference_prior(cache, comparisons, ARMS, SIZES, SOURCE,
                                      dimension=2, projection_epochs=30, prior_epochs=40, seed=7, margin=.5)
    path = tmp_path / "state.json"
    pre.save_checkpoint(path, result)
    document = json.loads(path.read_text())
    payload = document["payload"]
    if mutation == "zero_projection":
        payload["W"] = [[0., 0.], [0., 0.]]; payload["b"] = [0., 0.]
    elif mutation == "boolean_triplet_count":
        payload["protocol"]["triplets_per_eligible_anchor"] = True
    elif mutation == "integer_ready":
        payload["metrics"]["validation"]["all_lambdas_ready"] = 1
    else:
        field = "accuracy" if mutation == "boolean_accuracy" else "lambda"
        # Correct numerical value 1 must not make Boolean metadata valid.
        assert payload["metrics"]["validation"]["per_arm"]["small"][field] == 1.
        payload["metrics"]["validation"]["per_arm"]["small"][field] = True
    document["sha256"] = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()).hexdigest()
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError):
        pre.load_checkpoint(path)


def test_oversized_cache_save_rejects_before_creating_unreadable_artifact(tmp_path):
    # This exceeds the same 16 MiB boundary the loader enforces.
    cache = pre.EmbeddingCache(["q"], ["a" * 64], np.full((1, 900000), .1234567890123456), RECIPE)
    path = tmp_path / "oversized.json"
    with pytest.raises(ValueError, match="16 MiB"):
        cache.save(path)
    assert not path.exists()
