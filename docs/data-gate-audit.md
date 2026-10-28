# G0-AUDIT — 2026-10-28

**Verdict: PARTIAL.** Existing routing data and smoke behavior are verified. The project is ready for scoped B01–B03 contract/evaluator work; it is not ready for full preference training or faithful paper-result claims. No research implementation or dataset was rewritten during this audit.

Audited code: `f07ac1cee551fda203083aa3f160f3a09aa932fd`, branch `codex/data-gate`. The checkout was clean at audit start. The new handoff is the three files in the supplied `PILOT_CODEX_FILES_RESENT_2026-10-28` directory; its roadmap is a plan, not evidence that every gate is complete.

## Existing files and behavior

| Path | Role / evidence |
| --- | --- |
| `pyproject.toml`, `uv.lock` | Python environment; NumPy, pandas, PyArrow, pytest. No project lint, typecheck, build, E2E or CI gate configured. |
| `configs/data_sources.json` | Three immutable source revisions, byte sizes and SHA-256 hashes. |
| `configs/model_mapping.json` | Two exact IDs, six proposed aliases, three checkpoint-unverified mappings. |
| `scripts/fetch_data.py` | Download and checksum verification; not executed in G0 because local files exist. |
| `scripts/data_io.py` | Hash check followed by restricted pandas/NumPy pickle loading. |
| `scripts/data_gate.py` | Normalization, grouped deterministic splits, observation/feedback DTOs, one-use environment and smoke learner. |
| `scripts/audit_data.py` | Full-file profiling and preparation. Not rerun: it rewrites prepared files. |
| `tests/test_data_gate.py` | 16 checks covering validation, splits, feedback exposure, parsing and pickle/hash guards. |
| `reports/data-gate.json`, `docs/data-gate.md` | Original conditional Data Gate evidence; preserved. |
| `data/README.md`, `README.md`, `CONTEXT.md` | Existing commands and unresolved research choices. |

There is no production policy contract, feature-vector provider, regret evaluator, baseline implementation, preference training, cost estimator, or deployment policy yet. The existing `RandomSmokeLearner` records feedback for I/O verification; it is not a complete experimental baseline.

## Data verification

Local root: `D:\DOWNLOAD GG\PILOT_DATA`. All three source files pass their pinned size and SHA-256 checks. All prepared-file hashes and configuration hashes match the original report.

| Source | Pinned revision | Bytes |
| --- | --- | ---: |
| RouterBench 0-shot | `784021482c3f320c6619ed4b3bb3b41a21424fcb` | 99,567,659 |
| RouterBench 5-shot | same revision | 171,091,324 |
| Arena 55K | `18c298340948c0e7f7727399fd459cca6ce0ca6f` | 184,175,071 |

Full hashes are retained in [data_sources.json](../configs/data_sources.json) and the [fresh G0 evidence](../reports/audits/g0-evidence-2026-10-28.json). RouterBench dataset licensing remains unresolved; the publisher Arena 55K card's license declaration was recorded in the original audit. No access/license conditions were changed or accepted during G0.

Prepared schemas:

- `queries.parquet`: `query_id`, `query`, `query_segments_json`, `task`, `query_hash` — 36,497 rows.
- `outcomes.parquet`: `query_id`, `arm_id`, `reward`, `actual_cost`, `estimated_cost` — 401,467 rows.
- `splits.csv`: `query_id`, `query_hash`, `task`, `split`, `stream_position`.

Fresh checks confirm unique query IDs and query/arm pairs, exactly 11 arms per stored query, finite rewards in [0,1], finite nonnegative costs, and null estimated costs. Saved split assignments equal an in-memory recomputation: 1,000 tuning, 32,270 learning, 3,227 deployment, seed 42. Query IDs and normalized prompt groups do not cross split boundaries. Saved stream positions are contiguous within each split.

The 11 arms are `WizardLM/WizardLM-13B-V1.2`, `claude-instant-v1`, `claude-v1`, `claude-v2`, `gpt-3.5-turbo-1106`, `gpt-4-1106-preview`, `meta/code-llama-instruct-34b-chat`, `meta/llama-2-70b-chat`, `mistralai/mistral-7b-chat`, `mistralai/mixtral-8x7b-chat`, and `zero-one-ai/Yi-34B-Chat`.

The unchanged hashes preserve the earlier findings: 0-shot is complete but provisional; 5-shot has duplicate IDs and missing rewards; 86 raw task labels differ from the paper's 64. Arena 55K is available locally, but only 434 decisive pairs use exact IDs. The 3,477-pair scenario depends on nine unresolved mappings. Prior-training readiness remains false.

## Fresh commands and outcomes

```powershell
uv run --locked --offline --no-sync python -B -m pytest -q -p no:cacheprovider
```

Exit **0**; `16 passed in 5.18s`; measured command wall time **6.760 s**. `--offline --no-sync` uses the existing environment without downloading dependencies; `-B` and disabling pytest's cache avoid changing code/cache artifacts.

```powershell
.venv/Scripts/python.exe -B -m reports.audits.g0_checks_2026_10_28 --data-dir 'D:\DOWNLOAD GG\PILOT_DATA'
```

The [audit probe](../reports/audits/g0_checks_2026_10_28.py) reads existing source/prepared files, recomputes split assignments in memory, and runs two fresh 10-round smoke replays. It does not download, normalize/export data, train a model, or call an API. Both traces match, and the first feedback matches the original saved one-round smoke. Trace SHA-256: `4055cdecd1e06344b4ff9f5b81938871a6e2af8cec2f684e4ba3af34f6e1a391`. Exact probe timing and diagnostics are in the fresh JSON evidence.

Basic code inspection and local negative probes were performed. A dedicated security audit/scanner was not run. Lint/typecheck/build/integration/E2E suites were not run because none are configured; the 16 pytest checks must not be described as those suites passing. The 10-round replay is I/O verification, not benchmark performance.

## Findings relevant to implementation

1. **B01 input validation gap, reproduced:** constructing `OfflineBandit` directly with a `NaN` reward allows `step` to return `NaN`. The normalizer rejects this upstream, and the stored prepared dataset passes the fresh finite-value checks, so this does not invalidate the historical smoke. A general simulator must validate its own boundary or consume a validated dataset type.
2. **B01 catalog contract gap:** arm choices are derived separately from each query's available rows. A probe with one arm for one query and 11 for the others is accepted. Variable pools are not inherently a bug, but the planned fixed-pool experiment needs an explicit catalog and an error for a missing pair. Do not hardcode 11 into the reusable contract; toy and binary experiments need smaller declared pools.
3. **B01 no-leakage boundary currently holds for the supplied runner:** `select` receives query metadata and arm IDs; `update` receives one chosen reward/cost after selection. Tests check this path and reject a second reward request for the same query. Python private attributes are conventions, not a sandbox against hostile policy code. B01 should pass only context to policies and reserve costs/oracle observations for the evaluator.
4. **B02 reuse opportunity:** split generation and saved IDs already exist and pass verification. Add validation/load behavior and explicit subset semantics; do not recreate or silently reshuffle the full dataset.
5. **B03 is missing, not broken:** there is no metrics evaluator. Implement total/mean observed reward, accumulated selected-arm cost, action counts, and empirical oracle regret with analytic toy expectations. Oracle access belongs to evaluation after action selection.
6. **Research gates still open:** preference identity, prompt/multi-turn semantics, task aggregation, embedding choice, and estimated cost remain unresolved. They do not block synthetic-context contracts/metrics; they do block claims of a complete faithful reproduction.

## Scoped next change for B01–B03

Use the existing Python modules and environment. Preserve preparation logic and public smoke entry points. No package reshuffle is justified for this stage.

| Ticket | Smallest proposed change | Acceptance criteria |
| --- | --- | --- |
| B01 | Add explicit policy/context and fixed arm-catalog contracts around the current environment; validate reward/cost/IDs/shapes; hand policies only numeric context and selected reward. | Reject NaN/Inf, missing/extra pairs, duplicate IDs, unknown arms and invalid vector dimensions. Spy policy never receives outcomes, oracle actions or costs before selection. Other-arm outcomes cannot affect a fixed policy's action. |
| B02 | Load and verify the existing split manifest; support clearly labeled synthetic fixtures and deterministic small subsets without changing full split membership. | Exact stored full-split counts, disjoint IDs/prompt groups, stable stream order and checksum; same seed gives the same subset. MMLU remains a separate future track. |
| B03 | Add an evaluator and a learning-only runner around the contracts; log reward, cost, action counts and oracle regret. | Hand-calculated multi-arm fixture matches per-round and cumulative metrics; oracle computation occurs after selection; policy cannot inspect the evaluator; same seed yields the same trace. |

The requested contracts, data separation and acceptance criteria are already described in the user's supplied roadmap. This table narrows them to the inspected repository. It is not a second specification system. If OpenSpec is approved, record this next change there as the single implementation source of truth, referencing the roadmap and this audit.

Synthetic numeric contexts will test the contracts without choosing/downloading an embedding model or implying paper-equivalent features. B01–B03 does not include LinUCB, prior training, cost-policy deployment, full benchmark runs, plots of paper results, or publishing code. CPU unit/small fixture checks should remain small; any meaningful expansion requires a measured resource plan.

## Workflow decision (resolved after audit)

At audit time, OpenSpec CLI **1.15.0** was available and this repository had no OpenSpec root. On 2026-10-28 the user explicitly approved repo-local initialization and B01–B03 implementation. The active source of truth is [baseline-infrastructure](../openspec/changes/baseline-infrastructure/tasks.md). No global configuration, unrelated repository, source upload or remote Git action is included. The findings above describe the preimplementation G0 state; subsequent fixes are documented in [baseline-infrastructure.md](baseline-infrastructure.md).

## Paper fidelity and handoff

Reference: [published PILOT PDF](https://aclanthology.org/2025.findings-emnlp.1301.pdf), Sections 2.1, 3.1–3.2 and Appendix C.1. The paper gives 1,000 tuning samples and a 10:1 learning/deployment ratio. Appendix C.1 defines cumulative regret against the best arm for each query. Using the fixed offline outcome table is an explicitly recorded empirical interpretation; do not call it expected-reward pseudo-regret without an expectation model.

Decision details and paper-version differences are recorded in [ambiguities.md](ambiguities.md). The audited artifact checks are PASS; overall reproduction readiness is PARTIAL. Recommended next implementation ticket is B01, followed by B02–B03 using the agreed contracts. Preference training remains a later gated stage.
