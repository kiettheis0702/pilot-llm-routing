# Scientific and implementation decisions

Recorded 2026-10-28 during G0-AUDIT. This register distinguishes established behavior from proposals; it does not approve a new experiment or change a data split.

| ID | Question / current choice | Fidelity status | Next owner ticket / effect |
| --- | --- | --- | --- |
| DATA-01 | Pinned 0-shot has complete rewards and 36,497 rows, but the authors' exact file/revision is unknown. | `PAPER_INFERRED`, provisional | E01: confirm or explicitly document a substitution before comparing to paper figures. |
| DATA-02 | Local files contain 86 raw task labels; paper reports 64 tasks. No undocumented relabeling/filtering. | Unresolved | E01: establish grouping/filtering, or report the discrepancy. Does not prevent toy contract work. |
| DATA-03 | Seed 42, prompt-group split, hash ordering and integer rounding are local choices. The 1,000 tuning and 10:1 sizes follow the stated protocol. | Sizes `PAPER_EXACT`; implementation `PAPER_INFERRED` | B02: reuse the saved split, validate it on load. |
| PREF-01 | Arena 55K is downloaded, but equivalence to the authors' preference release is unconfirmed. | Potential `DEVIATION` if adopted as a substitution | P01: establish provenance before training. |
| PREF-02 | Two literal exact IDs; six proposed aliases; three checkpoint-unverified mappings. 3,477 pairs assumes all mappings. | Unresolved; mismatched versions would be `DEVIATION` | P01: verify mappings; never manufacture a prior for a missing arm. |
| PREF-03 | Ties, 484 multi-turn decisive pairs, repeated prompts, and nested missing responses need a training protocol. | Unresolved | P01/P04: document eligibility and group holdout. Current counts are audit counts, not a training set. |
| CONTEXT-01 | Joining all RouterBench prompt segments preserves the available text, but does not reconstruct roles or establish the paper's embedding input. | `PAPER_INFERRED`, provisional | P02: inspect prompt construction; approve embedding provenance before any external calls. |
| COST-01 | Stored `actual_cost` is a historical response-dependent benchmark estimate; `estimated_cost` is null. | Known limitation | A05: build tuning-only estimation. Never replace null estimates with realized deployment costs. |
| METRIC-01 | Appendix C.1 defines a sum of per-query best-arm reward minus selected-arm reward. On the fixed offline table, call this empirical oracle regret, not expected-reward pseudo-regret. | Formula `PAPER_EXACT`; offline interpretation `PAPER_INFERRED` | B03: evaluator computes it only after selection; document reference and model pool. |
| MODEL-01 | No embeddings, learned prior, LinUCB, or PILOT implementation currently exists. | Not implemented | B01–B03 use explicit synthetic numeric contexts for analytic tests; no paper-result claim. |
| PAPER-01 | The handoff primarily references arXiv v2; the published ACL PDF has a different appendix numbering (e.g. ablation in D.6). | Source-version distinction | Use the published ACL PDF as authoritative; record section names and paper version in future manifests. |

Reference: [published PILOT paper](https://aclanthology.org/2025.findings-emnlp.1301.pdf), especially Sections 2.1, 3.1–3.2 and Appendix C.1. The previous source/data findings remain in [data-gate.md](data-gate.md).

B01–B03 implementation update: DATA-03 verification and METRIC-01 evaluation are
implemented under [the OpenSpec change](../openspec/changes/baseline-infrastructure/tasks.md).
Deterministic hash-ranked subsets restore the existing stream order and are an
engineering `EXTENSION` for smoke verification, not a paper experiment protocol.
Synthetic two-dimensional contexts remain explicit test inputs (MODEL-01).
Preference identity/version, task aggregation and estimated-cost questions remain open.

## Implementation update after B01–B07 and numerical core work

The table above preserves the original G0 findings. MODEL-01 is now superseded:
baseline, PILOT, pretraining and deployment mathematical cores exist, with synthetic
end-to-end evidence. No real embedding cache or eleven-arm trained prior exists.
PREF-03 preparation now excludes ties, multiple turns and incomplete responses,
and uses a deterministic prompt-group holdout; see [preference-readiness.md](preference-readiness.md).

| Decision | Current explicit interpretation / remaining question |
| --- | --- |
| BASELINE-01 | ExploreOnly uses uncertainty alone; EpochGreedy explores at rounds 0,w,2w. These schedules are PAPER_INFERRED; [baselines.md](baselines.md). |
| PREF-04 | Triplets require the anchor winner to lose to a strictly cheaper arm in the negative pair. Cost-size order is caller supplied. Margin, one triplet per eligible anchor, SGD settings and initialization are PAPER_INFERRED; [pretraining-core.md](pretraining-core.md). |
| PREF-05 | Per-arm held-out pair accuracy counts both participants; ties count incorrect. Zero/absent accuracy blocks inverse lambda rather than choosing a floor. Denominator/tie policy is PAPER_INFERRED. |
| COST-02 | An explicit affordability intersection prevents estimated overspend (DEVIATION); rollover, final-bin query count and utilization conventions are documented in [cost-policy.md](cost-policy.md). Bounds are explicit inputs, never fit on deployment outcomes. |
| COST-03 | Tuning-only estimation is implemented, but actual tokenizer provenance, output-token counts, historical prices and real LB/UB remain unresolved. Synthetic counts/prices cannot close this gate. |
| DEPLOY-01 | The current integrated runner accepts PILOT cosine predictions. A fair multi-method deployment experiment still needs a declared score/cost interface for each baseline; raw unbounded linear scores are not silently clipped. |
| SCALE-01 | The NumPy pretraining core enumerates triplet candidates and JSON caches are capped at 16 MiB. This is a small-input correctness implementation; full-scale caching/sampling requires separate design and resource measurement. |

DATA-01/02, PREF-01/02 and CONTEXT-01 still prevent paper-equivalent claims.
