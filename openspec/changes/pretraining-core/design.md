# Design

## Context

The preference preparation change keeps actual eleven-arm training blocked. Existing dependencies include NumPy/pandas, and `scripts/pilot.py` accepts explicit prior/lambda mappings with provenance. This change supplies only a separate training core and toy verification; it does not wire synthetic priors into benchmark runs.

## Goals / Non-Goals

**Goals:** verify sampling conditions, gradient arithmetic, frozen-projection fitting, support-aware held-out metrics, and JSON artifact contracts on CPU.

**Non-Goals:** generate embeddings, discover model sizes, resolve real model aliases, run actual data training, perform model selection, or claim reproduced paper performance.

## Decisions

- A validated embedding cache carries an ordered ID list, normalized prompt hashes, finite 2D matrix, explicit model/version/source/normalization/synthetic metadata, dimension and content fingerprint. IDs are unique; duplicate prompt hashes are allowed within one split but never across splits. Cache inputs are copied.
- Comparisons identify battle, query, winner, loser and train/holdout partition. Validation requires distinct known arms, unique battles, present cache IDs, nonempty partitions and support for each arm in both partitions. Extra unrelated cache IDs are permitted but never fitted.
- Size values are caller-supplied positive finite token-cost order values covering the exact catalog, with a nonempty source. Smaller means strict numeric inequality. No mapping of model names to size, price inference or order default exists.
- Sampling uses train comparisons only. For each anchor, positives share the winner; negatives have the anchor winner as loser against a smaller winner. Members must have three different prompt hashes. Uniformly choose one valid positive/negative combination per eligible anchor from sorted candidates with a seeded RNG. Anchors without candidates are counted; zero total triplets blocks fitting.
- Projection is `z=W*x+b`; triplet loss is `max(0, margin - cos(anchor,positive) + cos(anchor,negative))`. Pair loss is `logaddexp(0, cos(z,theta_loser)-cos(z,theta_winner))`. Gradients follow the cosine derivative, with stable normalization and explicit rejection of near-zero/nonfinite vectors; no epsilon-generated directions.
- Deterministic seeded SGD uses shuffled train triplets/comparisons each epoch, finite validated updates and no holdout-driven stopping. Phase one initializes W from seeded Gaussian values and b=0; phase two initializes arm vectors independently, copies W,b, and updates only selected pair vectors. Optimizer, initialization, margin, epochs, sampling and prompt-distinctness are `PAPER_INFERRED` choices.
- Held-out comparison accuracy is one when the observed winner has strictly greater cosine score; exact prediction ties count incorrect. Both participating arms receive that comparison's correctness. Lambda is `1/accuracy` for supported positive accuracy, otherwise unavailable with a reason. Full lambda export is blocked if any arm is unavailable.
- Save only JSON, with exact schema keys and a canonical-payload SHA-256 envelope. Reject duplicate JSON keys, nonfinite numbers, unexpected fields, bad dimensions/catalogs and checksum mismatch. Saves are exclusive-create, preventing silent overwrite. Integrity hashes are not signatures or evidence that supplied provenance is truthful.
- A checkpoint includes cache IDs/fingerprint/recipe, comparison and split fingerprints, cost-order provenance, learned W,b and arm matrix, initial/final losses, held-out metrics and seed/hyperparameters. Readback validates all structural numeric contracts. Future inference accepts vectors from the same declared embedding recipe/dimension; dataset identity is separately retained.

## Risks / Trade-offs

- Cosine is undefined at zero → fail clearly rather than invent a direction; overly large SGD steps may fail and require explicit configuration changes.
- Toy loss decrease does not establish convergence on real data → label all execution synthetic, retain actual-data readiness blockers.
- Dense NumPy SGD is intended for correctness and tiny fixtures → no performance/full-scale training claims.
- Caller provenance can be false → record and validate completeness; do not claim independent source verification.

## Migration Plan

Add standalone source/tests/docs and a toy report. Existing preparation and PILOT APIs remain unchanged. No migration or external side effect occurs.
