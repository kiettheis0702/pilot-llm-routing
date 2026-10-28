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
