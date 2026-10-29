# Running the frozen local experiments

The experiment definition remains `configs/frozen_experiment_protocol.json`.
All six research priors are now independently verified and indexed in
`configs/research_prior_index.json`. This is the eight-arm local development
experiment; unresolved source/serving equivalence prevents an eleven-arm paper
reproduction claim.

Run from the repository root in PowerShell, one command at a time:

```powershell
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
$env:OPENBLAS_NUM_THREADS='1'
.\.venv\Scripts\python.exe -m scripts.experiments `
  --output-dir outputs/research-experiments-20261029 `
  --domain routerbench --seed 7 --method all --resume
```

Repeat sequentially for seeds `42`, `2026`, then the same three seeds with
`--domain mmlu`. `--method pilot` limits the command to one complete condition
for diagnostics; it does not complete the experiment inventory. Each primary
method produces four prefix checkpoints and eleven deployment runs. Prefixes
are snapshots of one uninterrupted learning stream, not retraining on shuffled
subsets. The first real full PILOT condition completed in 35.806 seconds on this
workstation, including input validation; this is an observation, not an ETA for
other conditions.

For each domain and seed, run the five declared ablations with `--method pilot`
and `--variant zero-prior-direction`, `unit-initialization-strength`, `prior-only`,
`single-budget-bin`, or `projection-dimension-8`. Single-bin requires the exact
completed main PILOT stream. The 8D variant selects the full 30/40-epoch prior,
never the earlier 2/5-epoch smoke. Prior-only records zero learning rounds and
marks its learning selection as an unused disjoint reference.

The CLI checks at least 8 GiB available physical RAM, no competing known heavy
job, single-thread BLAS settings and at least 16 GiB free disk. It performs a new
resource check before each method group. A failed check prevents the next group;
it does not terminate another application's work. All feedback comes from local
recorded outcomes. No provider request or API key is involved.

Every group first verifies the frozen data/config/source hashes and exact prior,
cache and tuning identities. Artifacts store the runtime source-file hashes,
seed, selected feedback trace, prefix snapshots and budgets. Completion manifests
are written last. `--resume` reads back all completed files and requires exact
request identity. It never appends to or rewrites a completed run. An incomplete
directory is an error requiring inspection; do not delete it or silently replace
it with a reduced run.

Full completion requires 858 primary plus 240 ablation deployment manifests,
independent numerical readback, paired descriptive aggregates and the figures,
report/poster and reproduction gates. A successful command or partial inventory
does not meet those requirements.

## Completed evidence

The complete 1,098-run inventory, all 108 independent replays, six figures and
PDF deliverables are packaged in
[`reports/full-experiments-2026-10-29/`](../reports/full-experiments-2026-10-29/README.md).
The frozen protocol still says execution was incomplete at declaration time;
that historical file is intentionally unchanged. The result manifest records
completion. Do not rewrite a frozen hash just to update a status sentence.

`tools/research/verify_runs.py` independently recomputes a full group. Example:

```powershell
.\.venv\Scripts\python.exe tools/research/verify_runs.py `
  --domain routerbench --seed 7 --method pilot --variant main `
  --output outputs/my-new-verification/routerbench-seed7-pilot-main.json
```

Use the three single-thread variables above. The output must be new. Never use
Python `-O`/`-OO` or `PYTHONOPTIMIZE` with evidence tools; current packaged tools
explicitly refuse optimization. Repeat one group at a time for each domain,
seed, primary method and PILOT variant. The published `verification/` reports
already cover all groups. `aggregate_results.py`, `render_figures.py` and
`build_documents.py` target the recorded run root and refuse existing aggregate,
figure or PDF destinations. Preserve completed evidence when preparing a new run.

The published package can be checked without private input data:

```powershell
$pilotPackage = Resolve-Path 'reports/full-experiments-2026-10-29'
$pilotFiles = Get-Content -Raw -LiteralPath (Join-Path $pilotPackage 'package-files.json') | ConvertFrom-Json
foreach ($pilotItem in $pilotFiles.files.PSObject.Properties) {
  $pilotActual = (Get-FileHash -Algorithm SHA256 -LiteralPath (Join-Path $pilotPackage $pilotItem.Name)).Hash.ToLowerInvariant()
  if ($pilotActual -ne $pilotItem.Value) { throw "Package checksum mismatch: $($pilotItem.Name)" }
}
Write-Output "PASS: $(@($pilotFiles.files.PSObject.Properties).Count) package files verified."
```

## Clean CPU environment

Install with `uv sync --locked --extra cost`. Set the single-thread variables,
check physical RAM and competing heavy jobs, then run these commands sequentially:

```powershell
.\.venv\Scripts\python.exe -c "from scripts.resources import require_resources; print(require_resources())"
.\.venv\Scripts\python.exe -m pytest -q --basetemp outputs/new-pytest-temp
.\.venv\Scripts\python.exe -m scripts.run_toy_pipeline --output-dir outputs/new-toy
.\.venv\Scripts\python.exe -m scripts.run_baselines --source toy --rows 100 --output-dir outputs/new-baselines
```

Choose unused output/temp names each time, and stop if any command fails. The
final reproduction evidence uses byte-preserving **source snapshots**, new
Python 3.11.18 environments, and offline installation from the pinned lockfile.
They are not new Git clones: these snapshots were taken before committing. The
smokes are explicitly synthetic; complete real-grid evidence is separate.

A clone alone cannot run the real commands because raw data, embeddings, priors,
tuning artifacts and large traces are excluded from Git. Transfer/prepare the
inputs referenced by `configs/frozen_experiment_protocol.json` and
`configs/research_prior_index.json`, retain relative paths, and verify their
checksums. See the [PC guide](PC_HANDOFF_2026-10-29.md),
[embedding pipeline](local-research-pipeline.md), [full-cache stage](full-embedding-prior-handoff-2026-10-29.md)
and [prior/tuning stage](tuning-handoff-2026-10-29.md). PDF authorship additionally
uses ReportLab; independent PDF checks use pdfplumber. They ran in the bundled
Codex Python, with versions recorded alongside the results.
