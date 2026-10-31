# Implementation and research readiness — 2026-10-29

> **TÀI LIỆU LỊCH SỬ — BGE / mốc 29–30/10/2026.** Các số liệu, trạng thái “hiện tại/current”, nhánh Git và việc còn chờ bên dưới chỉ đúng tại mốc ghi nhận. Không dùng làm kết quả OpenAI hiện hành. Đọc [trạng thái OpenAI](openai-followup-status.md), [bản giao việc hiện hành cho Claude](CLAUDE_UPDATE_OPENAI_2026-10-31.md) và [khôi phục từ checkout sạch](openai-reproduce.md). Giữ nội dung cũ để truy vết, không ghi đè bằng kết quả mới.

The current branch implements a runnable offline numerical pipeline. No Figure 3/4
reproduction or eleven-arm preference training is claimed. OpenSpec change files
are the source of truth for implemented units; this table maps them to the supplied
handoff's broader tickets.

| Tickets | Status | Evidence / remaining prerequisite |
| --- | --- | --- |
| G0 | Conditional | Pinned RouterBench 0-shot prepared; release/task/model ambiguities remain. [Audit](data-gate-audit.md). |
| B01–B07 | Implemented | Contracts, splits, metrics, five policy families and comparison CLI; toy and 64-row real-data plumbing runs with explicitly synthetic contexts. [Baselines](baselines.md). |
| P01/P04 | Evidence-aware preparation complete; full pool blocked | Exact-ID: 374 pairs; verified aliases: 987 pairs / five arms; identity matches: 1,558 pairs / eight arms. Three provider bridges unresolved. [Local preparation](local-research-pipeline.md). |
| P02 | Complete real GPU caches verified | All 36,497 routing and 1,558 preference vectors generated with the audited 2,048-token recipe. All checksums, identities, norms and resume checks pass. No routing truncation; 13 preference inputs truncated. [Evidence](../reports/full-embedding-prior-2026-10-29.json). |
| P03/P05–P07 | All declared eight-arm priors verified | Six research fits cover 64D and 8D at seeds 7/42/2026; all holdout coefficients pass independent checks. The full eleven-arm pool remains blocked. [Evidence](../reports/replicate-priors-2026-10-29.json). |
| A01–A07 | Implemented mathematical contracts | Prior-informed cosine UCB, selected-only updates, checkpointing and standalone cost policy with analytic/negative tests. [PILOT](pilot-core.md), [cost](cost-policy.md). |
| A08 | Complete real local deployment verified | Common frozen adapter supports all 13 primary methods. Every condition uses identical declared data, cost estimates and budget rules. [Results](../reports/full-experiments-2026-10-29/README.md). |
| E01–E06 / V01–V06 | Complete local grid independently verified | All 48 tuning candidates, 858 primary runs and 108 total primary/ablation groups independently replayed; complete seed/budget/prefix tables published. [Execution](full-experiments.md). |
| C01–C03 | Complete local ablations verified | All five controlled variants and 240 deployment runs completed with paired seed comparisons, including unfavorable outcomes. [Results](../reports/full-experiments-2026-10-29/README.md). |
| F01/F02 | Current clean-source environment verified | New locked environment: 441 passed, one Windows symlink skip; toy/baseline smokes and guarded real-group replay pass. Historical HEAD baseline: 310 passed. [Current evidence](../reports/full-experiments-2026-10-29/reproduction-post-review/report.json). |
| F03 | Local research report/poster verified | Six figures, nine-page report and A1 poster derived from the completed real grid. All 912 plotted values match source tables; all pages visually checked. Paper equivalence remains unresolved. |
| F04 | Scoped repository hygiene | Raw datasets stay external; explicit toy JSON and aggregate evidence are tracked. File-size/type checks are recorded in the log; no dedicated security audit implied. |

## Inputs needed to advance research

1. Resolve Arena release equivalence and the three remaining provider checkpoint
   bridges (Claude-v2, Mistral and Mixtral). Five/eight-model development pools now
   have explicit opt-in evidence policies; they do not establish full paper equivalence.
2. The frozen local plan, replicate priors, all budgets and controlled ablations
   are complete. Do not replace these results when studying a different protocol.
3. Figures, report, poster, clean-source reproduction and whole-branch review are
   complete. The review's optimization bypass was reproduced and fixed. The
   remaining CSV link-label minor was corrected during GitHub packaging on
   2026-10-30. [Final handoff](final-research-handoff-2026-10-29.md).
4. Continue resolving source-release/shot selection and 86 raw task labels versus
   the paper's 64 for the separate equivalence target. The current local experiment
   protocol freezes these as explicit limitations, not invented resolutions.

Completed engineering and development runs do not establish paper equivalence.
No paid API calls were performed. Full local embeddings, six research priors and
all declared routing experiments are verified. The sections below retain the
chronological verification history; earlier pending statuses are superseded by
the current table and final handoff.

## Local pipeline verification update

The uncommitted `local-real-data-pipeline` implementation passed 338 CPU tests
(baseline 310), with one actual symlink probe skipped because Windows denied
symlink creation. The path-check regression still ran using an injected path
classification. The full toy pipeline and 15-method / 64-row RouterBench smoke
passed; both use synthetic vectors. All three preparation policies ran on the
real pinned data and were independently read back, including grouped holdout
checks and all seven unchanged source-file checksums.

The review found four issues, all corrected: empty cost provenance, normalized
hashes hiding changes in verbatim text, unsafe writable resume paths, and partial
completion manifests. A reproduced Windows venv-launcher false conflict was also
fixed. Full evidence and outstanding runtime limits:
[verification report](../reports/local-pipeline-2026-10-29.json) and
[handoff](local-pipeline-handoff-2026-10-29.md).

After physical RAM became available, the optional environment installed with 31
compatible packages. PyTorch 2.8.0+cu128 passed an actual FP16 matrix operation on
RTX 5070 (compute capability 12.0). BGE-M3 produced 128 × 1,024 float32 unit vectors,
with zero truncated queries (maximum 493 tokens), approximately 1.09 GiB peak
allocated VRAM and 53.99 seconds inclusive of initial model setup/download.
Readback and resume passed without loading the model again. The 512-token cap
was used only for that smoke.

## Complete input audit update

The next change, `offline-research-experiments`, now tracks the remaining research
stages. Ten new tests bring the regression suite to **348 passed, one skipped**.
The full token audit covers 36,497 routing and 1,558 eligible preference inputs,
preserves frozen splits and source identities, and selects a 2,048-token cap from
learning/tuning and preference-training text only. Independent AutoTokenizer
checks and the earlier 128-query token totals agree. No routing input exceeds the
cap; 13 preference inputs do. There is no cross-corpus exact normalized prompt
overlap. The 16-query GPU feasibility probe stopped before model loading when
available RAM fell to 7.95 GiB after importing libraries, so no full cache or new
GPU result is claimed. [Evidence](../reports/input-audit-2026-10-29.json).

## Cost input update

The base environment now has an optional `cost` extra pinned to tiktoken 0.5.2.
The offline exporter produces all 36,497 query input counts and tuning-only
response counts for all 11 RouterBench arms. The public code's literal cost
formula is a factor of ten below recorded costs; an explicitly configured
multiplier reconstructs all 11,000 tuning query-arm costs within float tolerance.
It never silently rescales. The declared triplet cost order uses a local 1:1
input/output reference-token mix and supports the existing 2/5/8-arm evidence
policies. The full regression suite passes 360 tests with one Windows symlink
skip. Nine independent/integration probes passed, including a real wrong-scale
failure before artifact publication. [Protocol](cost-inputs.md),
[evidence](../reports/cost-inputs-2026-10-29.json).

## Full embedding and real prior update

After RAM became available, the feasibility probe and both complete GPU jobs
finished sequentially. Cache readback checks all 38,055 input identities, finite
unit vectors, 1,350 saved chunk hashes and exact agreement with the independent
length audit. Both completed caches resume without importing encoder libraries;
changed text/recipe and overwrite attempts leave them unchanged. The first real
eight-arm prior passes independent holdout and training-only triplet checks.
The full regression suite now passes **362 tests, one skipped** in 6.19 seconds.
This supersedes earlier RAM/cache/prior blockers without changing old immutable
run manifests. [Evidence](../reports/full-embedding-prior-2026-10-29.json),
[handoff and remaining work](full-embedding-prior-handoff-2026-10-29.md).

## Common deployment and real integration update

The new explicit adapter preserves frozen learned state and baseline semantics:
PILOT cosine, unclipped ridge means, random tie selection among eligible arms and
fixed-arm-only routing. All share cost estimation, binning, threshold/fallback,
estimated affordability and actual-cost accounting. Independent replay verifies
1,664 real learning actions/updates and 1,360 served deployment rounds across all
39 smoke runs. Complete regression: **390 passed, one skipped**. Final tuning,
budget/seed grids and ablations are still pending; no algorithm superiority is
claimed from the smoke. [Evidence](../reports/common-deployment-2026-10-29.json),
[handoff](common-deployment-handoff-2026-10-29.md).

## Tuning implementation in progress

The next stage has a declared paper alpha/window grid, complete candidate trace
and snapshot exports, tuning-only outcome isolation, and empirical budget-bound
calibration. Seventeen tests failed before the module existed. Post-implementation
tests have **not run**: preflights stopped at 7.92, 7.57 and 5.58 GiB; the latest
snapshot showed 5.91 GiB and no known heavy job. The prior-64 / 30+40-epoch fit
completed and reports valid coefficients for all eight arms, but its independent
readback remains pending. Task 3.2 stays unchecked. The 390-test result above
applies to the previous verified stage, not these new edits.
[Progress and restart instructions](tuning-progress-2026-10-29.md),
[evidence](../reports/tuning-progress-2026-10-29.json).

## Ablation design and verification preparation

Five controlled PILOT variants are now declared, with exact controls, paired
reporting rules and missing-run handling. They add 240 deployment conditions to
the 858 primary conditions. Static inventory arithmetic matches both plans, and
OpenSpec strict validation passes. An independent 64D-prior verifier has been
prepared, but has not executed. The latest targeted-test attempt was refused
before pytest at 5.65 GiB available RAM. This is design/preparation progress,
not a new passing runtime result; tasks 2.3 and 3.2 remain unchecked.
[Ablation protocol](ablation-protocol.md),
[progress evidence](../reports/ablation-plan-2026-10-29.json).

## Verified tuning and frozen protocol

After the user authorized closing Zalo/Foxit, RAM passed the unchanged resource
gate. Targeted tuning tests: 17 passed; complete regression: **407 passed, one
skipped**. Independent prior readback reproduces 188/313 holdout pairs and both
final training objectives. Both complete tuning searches pass independent replay
of all 33,552 actions/updates and their predictions, final states, scores and
calibration. In-memory held-out mutations leave 98 search files byte-identical;
invalid tuning rewards and repeated output paths fail without modifying original
data or artifacts. Tasks 2.3 and 3.2 are complete; 10/14 research tasks are now
complete. Full experiment execution, ablations, figures/report/poster and final
reproduction remain. [Evidence](../reports/tuning-2026-10-29.json),
[handoff](tuning-handoff-2026-10-29.md).
