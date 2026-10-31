# Reproduce the reported follow-up studies

Use a separate checkout of `openai-followup-reproduction-2026-10-31` for the
supplementary entrypoints. This later tag preserves the tested source and guides.
See the [completed fresh-machine validation](followup-reproduction-validation-2026-10-31.md)
for the exact exercised groups, 714 passing tests and remaining CUDA/full-grid limits.
The original `openai-reproduction-2026-10-31` tag preserves the earlier core
release and does not include these later additions. First restore the checksum-verified inputs using
[openai-reproduce.md](openai-reproduce.md). The core archive supplies all paid
OpenAI embeddings; none of the commands below calls a paid model API.
Run commands sequentially with at least 8 GiB available RAM and no competing
heavy job. Keep the three BLAS thread variables at 1 as in the core guide.

If the core guide left this clean checkout detached at the old release tag,
switch to the follow-up tag before using the new entrypoints:

```powershell
git fetch origin --tags
git switch --detach openai-followup-reproduction-2026-10-31
```

## Previously missing preparation

Before offline regression, restore the small public tokenizer vocabulary. A new
Python environment on an existing computer can silently reuse the global
tiktoken cache; a genuinely new machine cannot. This is a static public file,
not an inference API. `HistoricalCounter` checks its pinned SHA256 and version.

```powershell
$env:TIKTOKEN_CACHE_DIR=Join-Path (Get-Location) '.cache/tiktoken'
.venv/Scripts/python.exe -c "import tiktoken; from scripts.cost_inputs import HistoricalCounter; tiktoken.get_encoding('cl100k_base'); print(HistoricalCounter().metadata)"
```

```powershell
.venv/Scripts/python.exe -m tools.openai_study_reproduce prepare
.venv/Scripts/python.exe -m tools.openai_followup_assets --sources
```

Preparation reconstructs the prompt-grouped 847/211/500 preference split from
the restored original inputs and checks its historical SHA256. It also installs
the original hardware-only batch decision record at the path expected by the
immutable Hybrid adapter. Existing conflicting files are rejected.

The packaged batch record is **historical evidence from the original machine**,
not a new hardware benchmark. It preserves why the binary recipe used 8/16;
the actual `probe` below must still succeed on the reproduction machine before
declaring or training that recipe. Neither benchmark outcomes nor paid calls
were used to choose the batch size. The record cannot establish that batch8 is
faster on a different machine.

## Dependency map

| Report family | Inputs beyond restored core archive | How to produce them |
|---|---|---|
| Core | None | `tools.openai_study run` |
| Ablations | 8D priors; six main PILOT learning groups | `fit8`; `openai_study_reproduce parents` |
| Preference500 | Three-way split | `openai_study_reproduce prepare` |
| Budget policies | Six main PILOT learning groups | `openai_study_reproduce parents` |
| Cost shocks | Published core/ablation report tables | In Git; verified before analysis |
| RouteLLM MF | Three-way split; pinned upstream source; neural environment | Preparation; public source helper |
| Feedback noise | None | `freeze`, then `run` |
| Binary routing | Fresh two-arm tuning | `declare`, `tune`, `freeze`, `run` |
| Instructor | Public local weights; real compatibility probe; locally encoded vectors | Asset helper, probe, encoder |
| Online task shift | Hybrid drift cohorts and classifier declaration | Original Hybrid probe/audit/declaration |
| Native Hybrid binary | Public DeBERTa weights, source, batch decision record, actual probe/audit | Preparation, assets, default-batch adapter |

The [recorded course requirements](course-requirements-audit-2026-10-30.md)
require runs to be repeatable and all reported experiments to have documented
inputs/commands. They permit a justified core reproduction and do not explicitly
require executing every completed grid a second time in another checkout. A
small check proves the exercised path only, not completion of every condition.

## Core-dependent numerical studies

On a restored checkout `run` resumes a matching completed group. It regenerates
raw learning/deployment artifacts; those large artifacts are not in the archive.
For follow-ups alone, `openai_study_reproduce parents` produces just the six
required PILOT groups, preserving the exact core protocol and runner; it does
not claim the other72 core baseline groups are complete. Use the original core
`run` command instead when reproducing the full core grid.

```powershell
.venv/Scripts/python.exe -m tools.openai_study_reproduce parents
.venv/Scripts/python.exe -m tools.openai_study_followup fit8
.venv/Scripts/python.exe -m tools.openai_study_followup freeze
.venv/Scripts/python.exe -m tools.openai_study_followup run
.venv/Scripts/python.exe -m tools.openai_study_budget freeze
.venv/Scripts/python.exe -m tools.openai_study_budget run
.venv/Scripts/python.exe -m tools.openai_study_noise freeze
.venv/Scripts/python.exe -m tools.openai_study_noise run
.venv/Scripts/python.exe -m tools.openai_study_binary_run declare
.venv/Scripts/python.exe -m tools.openai_study_binary_run tune
.venv/Scripts/python.exe -m tools.openai_study_binary_run freeze
.venv/Scripts/python.exe -m tools.openai_study_binary_run run
```

For smoke coverage, the core and ablation runners accept `--limit-groups 1`;
budget/noise/binary accept `--limit 1`. A limited core run does **not** supply all
six learning groups needed by budget `freeze` or all single-bin ablations.
Full inventory verifiers intentionally refuse partial inventories. Do not mark
a smoke output as a completed report. Native frozen report builders refuse to
replace the reports already present in Git; retain fresh outputs separately.

The binary runner orders methods from its frozen protocol: `--limit 1` produces
epoch-greedy, not PILOT. Use `--limit 7` to complete all seven methods at seed7
before the PILOT group reader below, or run without a limit for all three seeds.

Budget and noise readers accept the same smoke limit; omit it after full runs:

```powershell
.venv/Scripts/python.exe -m tools.openai_study_budget_verify --limit 1
.venv/Scripts/python.exe -m tools.openai_study_noise_verify --limit 1
.venv/Scripts/python.exe -m tools.openai_study_binary_verify tuning
.venv/Scripts/python.exe -m tools.openai_study_binary_verify --domain routerbench --seed 7 --method pilot --output outputs/reproduction-readback/binary-seed7.json
.venv/Scripts/python.exe tools/research/verify_runs.py --domain routerbench --seed 7 --method pilot --variant zero-prior-direction --base outputs/openai-followup-20261031/ablations/runs --protocol outputs/openai-followup-20261031/ablations/protocol.json --index outputs/openai-followup-20261031/ablations/prior-index.json --cache-dir outputs/openai-standard-20261030/imported/caches/source-0 --output outputs/reproduction-readback/ablation-seed7.json
```

The group reader checks all traces for the requested domain/seed/method/variant;
repeat it for every completed group when verifying a full inventory. For
Instructor groups, use its `outputs/instructor-core-20261031` runs/protocol/index
and `outputs/openai-followup-20261031/instructor/caches/source-0`. Reader outputs
are immutable: choose a fresh output filename for another verification.

## Preference500 and cost analysis

The seed42 run evaluates all three predeclared candidates using calibration
accuracy, breaking ties by fewer epochs. Seeds7/2026 reuse that choice. The
held-out 500 labels never choose a candidate or the per-arm initialization.

```powershell
.venv/Scripts/python.exe -m tools.openai_study_reproduce preference --seed 42
.venv/Scripts/python.exe -m tools.openai_study_reproduce preference --seed 7
.venv/Scripts/python.exe -m tools.openai_study_reproduce preference --seed 2026
.venv/Scripts/python.exe -m tools.openai_study_reproduce mutation
.venv/Scripts/python.exe -m tools.openai_study_reproduce costs
```

The entrypoint checks the original split, cost-size and embedding-cache hashes,
plus the frozen report package, before fitting. Changing a valid input after
preparation does not silently define a new reproduction. It writes operational
verification separately. `mutation`
actually refits the three candidates after flipping only the500 final-test
labels, checks the byte-identical selected checkpoint and independently scores
the changed labels. It does not fabricate the historical
`preference500-verification.json`. Each ordinary fit is independently scored
and compared to the published per-seed values. Cost analysis keeps decisions/rewards fixed, independently
checks overspend counts/excess and compares all 18 published aggregate rows.

## RouteLLM MF

Create the pinned neural environment using [course-reproduce.md](course-reproduce.md).
No paid provider SDK is used for these cache-only fits.

```powershell
.venv/Scripts/python.exe -m tools.openai_study_routellm_train freeze
.venv-embedding/Scripts/python.exe -m tools.openai_study_routellm_train run --seed 7
.venv-embedding/Scripts/python.exe -m tools.openai_study_routellm_train run --seed 42
.venv-embedding/Scripts/python.exe -m tools.openai_study_routellm_train run --seed 2026
.venv-embedding/Scripts/python.exe -m tools.openai_study_routellm_train run --seed 42 --mutation-probe
.venv/Scripts/python.exe -m tools.openai_study_routellm_verify
```

## Instructor sensitivity

```powershell
.venv/Scripts/python.exe -m tools.openai_followup_assets --model instructor
.venv-embedding/Scripts/python.exe -m tools.openai_study_instructor_probe
.venv-embedding/Scripts/python.exe -m tools.openai_study_instructor_encode prepare
.venv-embedding/Scripts/python.exe -m tools.openai_study_instructor_encode encode
.venv-embedding/Scripts/python.exe -m tools.openai_study_instructor_verify
.venv/Scripts/python.exe -m tools.openai_study_instructor_study declare
.venv/Scripts/python.exe -m tools.openai_study_instructor_study fit
.venv/Scripts/python.exe -m tools.openai_study_instructor_study tune
.venv/Scripts/python.exe -m tools.openai_study_instructor_study freeze
.venv/Scripts/python.exe -m tools.openai_study_instructor_study run
```

Encoding accepts `--limit-chunks 1`; a repeat resumes the journal. A partial
encoding cannot enter `declare`. The original tail-recovery adapter is for the
original interrupted journal only; do not transplant that recovery overlay into
a fresh run. Public weights take about5GB; full encoding can take hours.

## Hybrid binary and task shift

```powershell
.venv/Scripts/python.exe -m tools.openai_followup_assets --model deberta
.venv-embedding/Scripts/python.exe -m tools.openai_study_hybrid_default_batch probe
.venv-embedding/Scripts/python.exe -m tools.openai_study_hybrid_default_batch audit
.venv/Scripts/python.exe -m tools.openai_study_hybrid_default_batch declare
.venv-embedding/Scripts/python.exe -m tools.openai_study_hybrid_default_batch train
.venv-embedding/Scripts/python.exe -m tools.openai_study_hybrid_default_batch verify
```

The separate drift fit retains microbatch2/accumulation64 and the original
prompt-disjoint cohort construction. Run all of these for the full comparison:

```powershell
.venv-embedding/Scripts/python.exe -m tools.openai_study_hybrid_train probe
.venv-embedding/Scripts/python.exe -m tools.openai_study_hybrid_train audit --cohort drift
.venv/Scripts/python.exe -m tools.openai_study_hybrid_train declare --cohort drift
.venv-embedding/Scripts/python.exe -m tools.openai_study_hybrid_train train --cohort drift
.venv-embedding/Scripts/python.exe -m tools.openai_study_hybrid_verify --cohort drift
.venv/Scripts/python.exe -m tools.openai_study_drift declare
.venv/Scripts/python.exe -m tools.openai_study_drift run
.venv/Scripts/python.exe -m tools.openai_study_drift_verify
```

Hybrid training requires20GiB free disk for recoverable checkpoints, in addition
to the RAM/CUDA checks. A probe performs real forward/backward/optimizer work,
but does not stand in for a completed five-epoch fit. Protocols include current
probe/environment hashes; fresh protocol bytes need not equal historical ones.
Keep original conclusions separate from any newly measured numerical differences.

## Repeat the CPU checks on a separate machine

The [Offline reproduction workflow](../.github/workflows/reproduction.yml)
uses one standard `windows-2025` runner and one test/BLAS worker. Its job refuses
private repositories. Standard runners in public repositories are
[free without consuming the included compute allowance](https://docs.github.com/en/billing/concepts/product-billing/github-actions).
It disables the Actions dependency cache and uploads no artifacts. Evidence is
printed in the run log as `REPRODUCTION_EVIDENCE_JSON`.

This workflow checks the actual available RAM before installation and each
scientific stage. It restores the public archive, runs preparation twice,
refits the three preference seeds and the real test-label mutation, rebuilds
six PILOT parent groups, exercises five ablations,40 allocation conditions,
two noise conditions and all seven binary methods at seed7 with fresh tuning. Independent
readers replay the traces; the final step compares the core/ablation/binary
metrics against the frozen tables. It also freezes RouteLLM's actual inputs,
without claiming a new neural fit. Existing CUDA recipes remain separate.

To execute the same stages manually, after environment/tokenizer preparation:

```powershell
foreach ($stage in @('inputs','preference','mutation','parents','ablations','budget','noise','binary','summary')) {
    & tools/reproduction-ci.ps1 -Stage $stage
    if (-not $?) { throw "Reproduction failed at $stage" }
}
```

The summary records the exact commit and source hashes. Repeating an identical
summary is safe; a conflicting existing summary is retained and rejected.
Fresh output paths are required for the immutable reader proofs. Start another
checkout for a complete repeat; do not delete or overwrite old scientific runs.
