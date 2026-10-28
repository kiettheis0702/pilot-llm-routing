# Design

## Context

See proposal.md. Existing `SplitSelection` carries IDs, phase and manifest digest; `ContextTable` and the learning runner already separate numeric policy inputs from outcomes. No tokenizer counts or explicit pricing catalog is present in prepared RouterBench schemas. Published source: https://aclanthology.org/2025.findings-emnlp.1301.pdf, Sections 2.2.3 and 3.2; Appendix C.2 Algorithm 3 (PDF index 12, printed page 23946).

## Goals / Non-Goals

**Goals:** Pure reusable A05–A07 APIs, deterministic decisions and finite auditable state. Synthetic fixtures establish correctness independently of unavailable research inputs.

**Non-Goals:** Deploy/learn runner integration (A08), tokenizers, guessed prices, observed-output estimation, router updates, prior training, full runs or performance claims.

## Decisions

- Use `TokenPrices` and `TuningTokenCostEstimator.fit(selection, output_counts_by_query, prices, token_provenance=..., price_provenance=...)`. Require exact tuning query/arm coverage. Copy all numeric input data; return fresh dictionaries. `validate_deployment` rejects different manifests and overlapping IDs. Provenance states source/tokenizer recipe and price source; these checks cannot authenticate a caller-forged selection.
- The formula is input count times input price plus per-arm tuning mean output count times output price, with prices explicitly per million tokens. Accept scalar input counts or per-arm counts because tokenizers may differ; never count tokens internally.
- `BinnedCostPolicy.select(predicted_rewards, estimated_costs)` commits the estimated reservation atomically and returns an immutable decision. It accepts no context, outcomes, actual costs or mutable router, so it cannot update bandit state. Invalid inputs leave state unchanged. Constructor requires explicit Q, S, B, LB, UB and ordered catalog.
- Preserve Algorithm 3's base-bin denominator `z = spent_in_bin / (B/N)`, not the carryover-augmented allocation. Reset at each bin. With carryover, z can exceed one: preserve and report this source inconsistency rather than silently clamp it. Use log-domain threshold comparisons to avoid overflow; do not divide by zero costs. Positive bounds are caller assumptions, not inferred or guaranteed observed ratios.
- `PAPER_INFERRED`: Q_left includes the current query; stable catalog-order ties; negative cosine gives no positive-cost threshold eligibility but remains available through the fallback. Zero reward/cost can qualify by the literal inequality. Zero budget permits zero-cost decisions with z=0, an explicit boundary extension.
- `DEVIATION`: intersect the paper threshold set with affordability. Algorithm 3 omits this safety guard. If the set is empty, use its stated average-budget fallback, even if a more expensive arm is currently affordable. Return an explicit terminal failure rather than silently skipping/reallocating queries.
- Use exact rational accounting of canonical decimal numeric representations for allocation, spending and average-budget comparisons, converting to finite floats only for public reports. This avoids fractional-bin and accumulated-float overspend; log calculations alone use floating point. Extremely unrepresentable report/utilization values fail atomically.
- `ActualCostLedger` binds to the independent policy only to validate issued decisions and read estimated totals, never to modify it. Record selected decision IDs once; report incomplete coverage, actual budget violation/excess and actual-minus-estimate error on recorded decisions.

## Risks / Trade-offs

- [Positive bounded ratios conflict with zero/negative cosine or free arms] → preserve numerical behavior, disclose theoretical assumptions not guaranteed; no competitive-ratio claim.
- [Mean output tokens underestimate actual output] → independent actual ledger reports violations; no actual-budget guarantee.
- [Token units/provenance and price vintage unknown] → require caller-supplied metadata and values; RouterBench integration remains blocked until verified.
- [Python callers can falsify provenance or reach private state] → API isolation and mutation tests prevent accidental leakage, not malicious-code sandboxing.
- [Strict affordability may differ from authors' undocumented code] → label the deviation and test it explicitly.

## Migration Plan

Add standalone module/tests/docs. Existing learning/data paths remain untouched; rollback removes these additions. Integrate with a future deployment runner only after token/pricing provenance and fixed-router split validation are ready.
