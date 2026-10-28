# NumPy preference pretraining core — 2026-10-28

**The numerical and artifact core is implemented and tested on synthetic data only. Actual eleven-arm pretraining remains blocked.** This provides reusable P02 cache contracts, P03 projection mathematics, P05 frozen-projection fitting, P06 diagnostics, and P07 JSON interchange. It does not supply real embeddings, verify nine unresolved model aliases, or train a benchmark prior. [Toy evidence](../reports/pretraining-toy.json) and [OpenSpec source of truth](../openspec/changes/pretraining-core/proposal.md) record the scope.

## Inputs and API

The module is `scripts.pretraining`. It uses NumPy and standard-library JSON; no Torch, model download, embedding generation or inference API is involved.

| API | Input / result |
| --- | --- |
| `EmbeddingCache(ids, prompt_hashes, vectors, provenance)` | Copies a finite numeric matrix; validates ordered unique query IDs and aligned SHA-256 prompt hashes. Duplicate hashes are allowed within one partition. |
| `cache.save(path)` / `EmbeddingCache.load(path)` | Safe JSON with declared recipe, dimension, ordered IDs/hashes and content checksum. `cache.fingerprint()` identifies the complete payload. |
| `validate_comparisons(cache, comparisons, arms)` | Validates schema, unique battles, known distinct arms, cache IDs and disjoint train/holdout prompt groups; returns battle-ID-sorted rows. Every catalog arm must have support in both partitions. |
| `sample_triplets(cache, comparisons, arms, sizes, size_provenance, seed=42)` | Samples train-only triplets and reports eligible/unusable anchor counts. Missing or invalid size input and zero usable triplets raise `ValueError`. |
| `triplet_loss_grad(W,b,anchor,positive,negative,margin=.2)` | Returns scalar loss and gradients for W,b. |
| `pairwise_loss_grad(context,theta,winner,loser)` | Returns scalar BCE loss and theta gradient; all unselected-arm gradient rows are zero. |
| `train_projection(...)` | Returns W,b, loss history and sampled triplets. Keyword settings: `dimension=2`, `epochs=30`, `learning_rate=.03`, `margin=.2`, `seed=42`. |
| `train_prior(cache,comparisons,arms,W,b,...)` | Fits theta only; returns theta and loss history. Settings: `epochs=40`, `learning_rate=.03`, `seed=42`. |
| `fit_preference_prior(...)` | Runs the two phases then held-out evaluation; returns a checkpoint dictionary. Settings: `dimension`, `projection_epochs`, `prior_epochs`, `learning_rate`, `margin`, `seed`. |
| `evaluate_prior(...)` / `inverse_accuracy(statistics)` | Reports supported per-arm held-out accuracy and inverse-accuracy lambda, or explicit zero/absent-support blockers. |
| `save_checkpoint(path,result)` / `load_checkpoint(path)` | Validated JSON checkpoint with exact parameter reload. |
| `project_cache(checkpoint,cache)` | Applies W,b after checking declared embedding recipe and dimension. Output is unnormalized; cosine consumers normalize it. New query IDs are permitted under the same recipe. |

Embedding provenance has exactly `model`, `version`, `source`, `normalization` (nonempty strings), and `synthetic` (Boolean). These declarations are required and recorded; they do not independently authenticate their source. Numeric strings and Boolean vector elements are rejected instead of coerced.

Each comparison has exactly `battle_id`, `query_id`, `winner`, `loser`, and `split`; all values are nonempty strings, and split is `train` or `holdout`. Prompt hashes should come from the preparation module's whitespace normalization. A query cannot leak across partitions, including when separate IDs share its hash. Extra cache IDs are permitted but fitting never reads their values.

`sizes` must provide positive finite token-cost size values for exactly the arm catalog. `size_provenance` is required. There is no model-name heuristic, price lookup, assumed model order, or default size table.

## Mathematical interpretation

The paper describes preference-conditioned triplets and a smaller-model condition based on token cost. Its second phase freezes the query projection and learns model vectors using a pairwise cosine softmax and binary cross-entropy. [Paper §2.2.1](https://arxiv.org/html/2508.21141v2#S2.SS2.SSS1)

This implementation uses:

```text
z(q) = W x(q) + b
triplet_loss = max(0, margin - cos(z_anchor,z_positive) + cos(z_anchor,z_negative))
pair_loss = logaddexp(0, cos(z,theta_loser) - cos(z,theta_winner))
```

For an anchor whose winner is arm A, a positive also prefers A. A negative must contain A as the **loser**, with a winner whose supplied size is strictly smaller than A's. A comparison won by any unrelated cheaper arm is insufficient. The three query prompt hashes must differ. Only train rows form candidates. The deterministic sampler picks one valid positive/negative combination per eligible anchor from sorted candidates with the supplied seed, then SGD shuffles that fixed sampled set each epoch.

The following are **PAPER_INFERRED**, explicitly stored/documented rather than claimed author settings: margin, seeded Gaussian initialization, per-example shuffled SGD, learning rate, fixed epoch counts, one triplet per eligible anchor, exclusion of repeated prompt groups within a triplet, and validation denominator/tie handling. The default margin is 0.2; the recorded verification fixture uses 0.5 to exercise an active loss. No hyperparameter is selected from holdout performance and no holdout-driven early stopping exists.

Cosine gradients use the derivative of normalized dot products. Norms at or below `1e-12`, nonfinite directions and invalid dimensions fail clearly. No epsilon manufactures a direction. Very large steps can therefore fail; decreasing learning rate is an explicit caller decision, not an automatic hidden retry. Phase two copies W,b and never updates them; its only learned values are theta rows for the current comparison. Tests change held-out vectors and labels while confirming identical W,b and theta.

For each held-out comparison, a prediction is correct only when the observed winner's cosine is strictly higher. Exact score ties count incorrect. Both participating arms receive that comparison's correctness, giving per-arm denominators over all held-out comparisons involving the arm. Positive supported accuracy yields `lambda = support/correct`; absent or zero accuracy yields `None` and a named blocker. No floor, default prior or lambda=1 fallback is substituted. The exact denominator and zero handling are implementation interpretations of the paper's inverse-accuracy instruction. [Paper §2.2.2](https://arxiv.org/html/2508.21141v2#S2.SS2.SSS2)

## Artifact behavior

JSON artifacts contain a schema version, exact expected keys and canonical-payload SHA-256 envelope. Duplicate keys, nonfinite values, malformed dimensions/catalogs, numeric Boolean substitutes, invalid projected/model directions, inconsistent accuracy/lambda histories, and checksum mismatch are rejected. Save uses exclusive creation and checks the **UTF-8 byte size** against the same 16 MiB limit used by the reader before opening the path. Existing artifacts are not overwritten, and rejected oversized writes leave no file.

Checkpoints retain W,b,theta; ordered arms; input/output dimensions; declared embedding recipe, IDs, prompt hashes and cache fingerprint; comparison/split fingerprints; supplied cost-size order/provenance; sampled query triplets; settings and both loss histories; held-out metrics and lambda blockers. They contain no raw prompt/response text. A checksum detects alteration but is not an authenticity signature. Reload also validates structure when a modified payload has a recomputed checksum.

The core deliberately uses a simple dense implementation and enumerates positive/negative combinations per anchor. It is suitable for correctness checks and small inputs, with no full-scale runtime guarantee. Actual data must remain gated by [preference readiness](preference-readiness.md): nine arms lack strict coverage, no validated real embedding cache is available, and no actual token-cost order has been pinned. Synthetic successful validation is not evidence that these blockers are resolved.

## Executed verification

```powershell
.venv\Scripts\python.exe -B -m pytest tests/test_pretraining.py -q -p no:cacheprovider
.venv\Scripts\python.exe -B -m pytest -q -p no:cacheprovider
& 'C:\Users\ASUS\AppData\Roaming\npm\openspec.cmd' validate pretraining-core --strict
git diff --check
```

- Initial TDD: 29 expected failures from absent helper behavior, exit 1. The first implemented run passed 28; the loss-decrease fixture was already at zero loss. It was corrected to contain a large shared nuisance coordinate, preserving the strict decrease assertion.
- Additional negative fixtures reproduced and fixed malformed hash sequences, all-zero projection checkpoints, Boolean numeric fields/readiness/counts, mixed Boolean/float input coercion, and oversized artifact saves. No failing assertion was weakened.
- Latest focused run: **39 passed in 10.97s**, exit 0; independent reviewer rerun: **39 passed in 1.81s**. Runtime varied with concurrent local work.
- Full current suite after the final input fix: **310 passed in 50.37s**, exit 0. Other agents were independently extending the checkout; this is the suite actually observed.
- Strict OpenSpec validation and whitespace check: exit 0. The actual new-module diff was reviewed (`git diff --no-index -- NUL scripts/pretraining.py`; exit 1 is the normal added-file diff status).

Recorded toy run: eight supplied 2D vectors, six train/two held-out comparisons, two synthetic arms, seed 7, 30 projection epochs, 40 prior epochs, learning rate .03, margin .5. Three of six train anchors provide valid strict-size triplets. Projection loss decreased **0.4980766569 → 0**; prior BCE decreased **0.4412338615 → 0.2532473589**. Each toy arm scored 2/2 held-out pair predictions, giving lambda=1. This is a deliberately simple mathematical fixture, not a benchmark result.

Central finite-difference maximum absolute errors were **9.84e-11 for W**, **6.21e-11 for b**, and **6.03e-11 for theta**. Checkpoint parameters and projected vectors matched exactly after JSON readback. A separate reviewing agent independently checked raw objectives over 30 seeded random cases and reported maximum absolute error **3.78e-10**. Its three-arm sampler probe across 20 seeds also excluded cheaper winners that did not defeat the anchor winner. Independent review discovered the save-size/Boolean-marker issues described above; both received failing regression tests before fixes. Final reviewer feedback reported no remaining findings.

The toy evidence was generated locally from `tests/test_pretraining.py::fixture` with `fit_preference_prior`, the finite-difference helper, and temporary-directory checkpoint readback. Its JSON includes source/test/readiness-report hashes, configuration, measured losses, gradient errors and explicit real-data blockers. Only synthetic fixture fitting ran. No external API, dependency/model download, raw data write, actual benchmark training or Git mutation occurred in this scoped work. No lint/typecheck/build/E2E/security scanner result is implied by pytest.
