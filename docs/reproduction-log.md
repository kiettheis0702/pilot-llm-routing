# Reproduction log

## 2026-10-24 — Repository setup

Connected the local checkout to `kiettheis0702/pilot-llm-routing`, initialized `main`, and added project context and Git hygiene. No dataset or result was assumed complete.

## 2026-10-24 — First Data Gate audit

Evidence: [audit report](data-gate.md), [machine-readable measurements](../reports/data-gate.json), [source pins](../configs/data_sources.json).

- Downloaded and hash-verified both processed RouterBench variants and the public Arena 55K candidate.
- Selected 0-shot provisionally because it has unique IDs, complete reward/cost data, and 36,497 rows. The paper's exact file selection remains unknown.
- Found 86 raw task labels versus the paper's 64; retained the labels without inventing a mapping.
- Rejected 5-shot for strict preparation because of duplicate IDs and missing rewards. No silent imputation or merge.
- The older Arena 33K file was gated (HTTP 401); recorded metadata only. Arena 55K is a separately identified candidate, not silently substituted as the confirmed original.
- Counted exact-ID and proposed-alias preference coverage separately. No preference priors were trained.
- Preserved fractional scores; grouped duplicate prompts across a seed-42 split; recorded deterministic output hashes.
- Exported query and outcome tables separately. Kept estimated cost unknown and response-dependent cost out of preselection observations.
- Passed a one-step selected-arm feedback/update smoke check and the focused data-contract tests. These are infrastructure checks, not research results.

Status: **conditional Data Gate**. Offline baseline development can proceed under the provisional protocol; exact dataset provenance, alias versions, preference preprocessing, and deployment cost estimation still need resolution.

## 2026-10-28 — G0 re-audit and B01–B03 infrastructure

G0 inspected HEAD `f07ac1cee551fda203083aa3f160f3a09aa932fd` and verified the three
source files, prepared artifacts and a repeated ten-round legacy smoke. The
existing suite passed: `16 passed in 5.18s`, exit 0. Evidence remains in
[data-gate-audit.md](data-gate-audit.md) and
[g0-evidence-2026-10-28.json](../reports/audits/g0-evidence-2026-10-28.json).
G0's constructor-gap observations describe the original code, before the fixes below.

The user explicitly approved repo-local OpenSpec setup and B01–B03 implementation.
Initialized CLI 1.15.0 for Codex; the authoritative change is
[baseline-infrastructure](../openspec/changes/baseline-infrastructure/tasks.md).
Worked in the existing `codex/data-gate` checkout to retain the audited files;
no commits, push, PR, dataset writes, dependency changes or external model calls.

- B01: simulator validates complete fixed catalogs, string identifiers and finite
  reward/cost boundaries; copies input frames. New numeric policy contract passes
  only context and selected reward, with independent context copies at select/update.
- B02: checks pinned manifest bytes and semantic assignments against query metadata
  and the declared seed; records deterministic subset provenance without rewriting splits.
- B03: learning-only lifecycle, per-round/cumulative reward and actual cost,
  all-arm action counts, empirical oracle regret, and CSV/JSON run logs with fingerprints.

TDD evidence: constructor/context tests initially had 23 failures (missing validation/API)
and 1 existing behavior pass; the 17 split tests and 11 evaluator tests failed before
their modules existed. These failures were expected development checks, not suppressed
assertions. Each implementation was followed by a passing full suite: 40, 57, then 68 tests.

An independent read-only reviewer inspected the implementation and ran the 68-test
suite successfully. It found a valid-input compatibility issue: pandas `string`/Arrow
string columns compared unequal solely due to dtype. The implementer treated this as
a functional contract defect worth fixing now (reviewer rated it minor), reproduced
all six dtype/column combinations, then normalized already-validated comparison values
without mutating inputs. All six regression tests went RED → GREEN.

Final commands and observed results:

| Command | Result |
| --- | --- |
| `uv run --locked --offline --no-sync python -B -m pytest -q -p no:cacheprovider` | Exit 0; **74 passed in 1.98s**. |
| `openspec validate baseline-infrastructure --strict` | Exit 0; change is valid. |
| `git diff --check` | Exit 0. New untracked text files were additionally checked for whitespace/syntax/JSON validity before handoff. |
| `uv run --locked --offline --no-sync python -B -m reports.audits.baseline_checks_2026_10_28 --data-dir 'D:\DOWNLOAD GG\PILOT_DATA' --output-dir 'reports/runs/b01-b03-2026-10-28-reviewed'` | Exit 0; probe 3.013 s, command wall 5.045 s. |

The final probe validates all 36,497 queries / 401,467 outcomes and the unchanged
1,000 tuning / 32,270 learning / 3,227 deployment split. Two ten-query runs match
exactly. Source and prepared SHA-256 hashes still match the original pins.
[Final evidence](../reports/runs/b01-b03-2026-10-28-reviewed/verification.json),
[summary](../reports/runs/b01-b03-2026-10-28-reviewed/summary.json), and
[trace](../reports/runs/b01-b03-2026-10-28-reviewed/trace.csv) are saved. The earlier
`b01-b03-2026-10-28` run is retained as pre-review evidence; use the `-reviewed` run
for current implementation hashes.

Hand-derived two-arm test: reward 1.25, mean 0.625, selected cost 0.30, regret 0.75.
RouterBench plumbing probe: reward 1.75, mean 0.175, historical selected cost
0.0006609, empirical regret 7.75 over ten rows. The fixed-arm probe uses synthetic
contexts; these figures measure contract execution, **not research performance**.

Review boundaries retained: DTO provenance and Python object isolation are trusted
local interfaces, not protection from a malicious caller; deployment, embeddings,
research policies and preference training remain outside B01–B03. Dataset/version
ambiguities remain in [ambiguities.md](ambiguities.md). These boundaries would need
revisiting before untrusted execution or paper-result claims. There are no deferred
minor implementation findings from the review.

Only pytest unit/regression tests and the explicitly described local-data probe ran.
No lint/typecheck/build/E2E suite or CI is configured; none is claimed passing.
Basic relevant code review covered validation and leakage; no dedicated security
audit or scanner was run. B01–B03 is complete; overall research Data Gate remains
conditional. Next implementation work is B04–B07 baselines, then the gated preference work.

## 2026-10-28 — Incremental continuation authorized

The user authorized all currently feasible work and commit/push after each completed
unit. B01–B03 was rechecked (74 tests, strict OpenSpec validation) and published as
`5e8f499` on `origin/codex/data-gate`.

B04–B06 adds five validated baseline policies, with disjoint linear UCB and explicit
PAPER_INFERRED exploration schedules. Independent review found finite but numerically
singular updates at context [1e8,1e8]; the new regression failed, then passed after
candidate Cholesky validation before state mutation. Policy tests: 17 passing.
The entire concurrent working tree suite also passed (151 tests at this checkpoint);
preference/cost-policy changes are separately reviewed units and are not part of this
policy commit. No full experiments or embedding calls were run.

P01/P04 preparation adds strict exact-ID parsing, grouped preference holdout and
aggregate readiness reporting. Parent reviewed the implementation and exclusion
precedence. The local source yields 374 eligible single-turn pairs, 299 train and
75 holdout, with no normalized prompt-group overlap. Nine arms have no eligible
coverage and the scoped cache inventory finds no embeddings. Unknown alias/version
statuses are unchanged; `all_arm_prior_ready=false`. Focused tests: 28 passing;
strict OpenSpec preference-readiness valid. The report is deterministic across two
measured executions. See [readiness details](preference-readiness.md). This is a
completed preparation unit with explicit blocked real-training prerequisites.

A05–A07 now provides a tuning-only token-cost estimator, Algorithm 3 binning and
rollover policy, plus a separate actual-cost ledger. Parent reviewed the threshold,
fallback, allocation arithmetic and separation of actual costs. The explicit
affordability guard is a documented DEVIATION; other unresolved edge interpretations
are listed in [cost-policy.md](cost-policy.md). 83 focused tests pass; six targeted
in-memory mutations were caught. Strict OpenSpec deployment-cost-policy passes.
Synthetic measured evidence is [cost-policy-toy.json](../reports/cost-policy-toy.json).
Real estimation still requires verified tokenizer/counts/prices and bounds; no
actual deployment cost or output is used as an estimated-cost shortcut.

A01–A04 mathematical core adds normalized cosine-UCB, explicit per-arm priors and
positive lambdas, selected-only ridge updates and validated JSON model checkpoints.
Parent reviewed numerical normalization, initialization, loading and evaluator use.
37 focused tests pass, including analytic matrices/scores, extreme finite vector
scales, corrupt checkpoints and continuation equivalence. Strict OpenSpec pilot-core
passes. A two-query synthetic integration obtains reward 2 and reloads identical
scores. Checkpoints resume model state, not a stream cursor; run_learning still
resets by contract. Zero query vectors fail, zero arm estimates score zero. Priors
are caller supplied with explicit provenance; no real trained prior is implied.

B07 now runs bounded baseline comparisons and produces standard Matplotlib PNG/SVG
plots from saved traces. Matplotlib 3.11.2 and its dependencies were resolved into
uv.lock (about 18 MiB of principal wheel downloads); no model/download/API was used.
The plan was updated from custom dependency-free SVG to a standard plotting library.
Toy: 7 methods × 100 rows, 2.227 s. RouterBench: 15 methods × 64 rows, 11.877 s.
Independent review recomputed all artifact metrics and checked fingerprints/order;
3 CLI regression tests pass. Source/prepared data remain read-only, and all contexts
are explicitly synthetic. See [baselines.md](baselines.md) and saved reports.

A08 connects a fixed learned PILOT to held-out deployment and separate estimated/
actual cost accounting. Parent reviewed the implementation and its declared model
lineage limitation. Preflight rejects split overlap, catalog/context/count gaps
before any reservation; no deployment feedback updates the learned router.
The focused deployment plus end-to-end toy tests passed: 29 tests in 4.23 s,
exit 0. Strict OpenSpec heldout-deployment is valid. Full-suite evidence and
analytic examples are in [deployment.md](deployment.md). This completes the
deployment API, while real deployment inputs remain gated.

P02/P03/P05–P07 now have explicit embedding-cache contracts, strict triplet
sampling, NumPy cosine losses/gradients, two-stage training with frozen projection,
held-out inverse-accuracy diagnostics and validated JSON interchange. This is a
small-input numerical core; real embedding generation and eleven-arm training
remain blocked. Independent review compared gradients to separately constructed
objectives (maximum error 3.78e-10) and found artifact-size/type validation defects;
regression tests preceded their fixes. Final focused suite: 39 passed. Reviewer
full suite: 310 passed in 47.10 s, exit 0. Strict OpenSpec pretraining-core passes.
Parent inspected the implementation and documented inferred sampling/SGD choices.
See [pretraining-core.md](pretraining-core.md) and its fingerprinted toy evidence.

The complete synthetic CLI now fits a prior, learns on 50 queries, reloads and
serves 6 held-out queries without router updates. Independent review confirmed
arm/lambda alignment, split isolation, fixed cosine deployment and provenance.
Parent found a Windows CSV newline issue: Git's CRLF-to-LF normalization changed
the persisted split checksum. A regression failed before explicit LF serialization
and then passed; evidence was regenerated in 0.217 s. The pre-fix generated output
is retained only in ignored local outputs, not committed. Source/lock and persisted
manifest hashes are checked. See [toy-pipeline.md](toy-pipeline.md).

After the newline correction, parent full-suite command
`uv run --locked --offline --no-sync python -B -m pytest -q -p no:cacheprovider`
passed **310 tests in 14.24 s**, exit 0. `openspec validate --all --strict`
passed all eight changes. Git-staged source, uv.lock and split bytes match the toy
report hashes, and `git diff --cached --check` passes. Inventory of 177 tracked or
proposed files found no file over 5 MiB and no raw dataset/model/private-key/env
file extensions; this is a scoped file inventory, not a secret-content/security
audit. Dataset licensing remains unresolved where recorded in the G0 audit.

## 2026-10-29 — Fresh checkout and environment verification

Verified published code revision `5de84dd2dbb1c24f392c82603cb264b433be5d39` in a
new temporary local clone (`git clone --no-hardlinks --single-branch --branch
codex/data-gate`) with a separate new `.venv`, using CPython 3.11.0.

The initial `uv sync --locked --offline --python <base Python 3.11>` exited 1
because dependency wheels were absent from the machine cache. Retried
`uv sync --locked --python <base Python 3.11>` and it exited 0, installing 20
locked packages. Reported downloads: PyArrow 27.2 MiB, NumPy 12.0 MiB, pandas
10.8 MiB and Pygments 1.2 MiB. These are ordinary Python dependencies; no model,
benchmark dataset or paid API was used. All subsequent execution used offline mode.

| Command inside the new clone | Observed result |
| --- | --- |
| `uv run --locked --offline --no-sync python -B -m pytest -q -p no:cacheprovider` | Exit 0; 310 passed in 23.59 s. |
| `uv run --locked --offline --no-sync python -B -m scripts.run_toy_pipeline --output-dir outputs/toy` | Exit 0; 50 learning rounds, 6 deployment queries, 0.195 s internal time. |
| `uv run --locked --offline --no-sync python -B -m scripts.run_baselines --source toy --rows 100 --output-dir outputs/baselines` | Exit 0; 7 methods × 100 rows, 2.908 s internal time; trace-derived PNG/SVG written. |
| Compare saved/new source, lock, split fingerprints, checkpoints, traces and numerical metrics | Exit 0; all match. Prior/learned/deployed checkpoint bytes match exactly. |
| `git status --porcelain` | Exit 0; empty, both before execution and after ignored outputs were generated. |

[Machine-readable evidence](../reports/reproducibility-2026-10-29.json) records
the revision, environment, commands, actual setup failure/recovery and results.
No lint/typecheck/build/E2E suite or CI exists in this repository; those checks
were not run or represented as passing. Relevant input/leakage/artifact validation
was reviewed; no dedicated security audit or scanner was run. Real-data research
prerequisites remain in [implementation-status.md](implementation-status.md).
