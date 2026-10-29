# Research evidence tools

These scripts read the fixed run directory
`outputs/research-experiments-20261029`. They use existing local inputs and never
call a model provider. The experiment CLI and protocol are documented in
`docs/full-experiments.md`; all commands run sequentially with the 8 GiB RAM gate.

```powershell
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
$env:OPENBLAS_NUM_THREADS='1'
.\.venv\Scripts\python.exe tools/research/verify_runs.py `
  --domain routerbench --seed 7 --method pilot --variant main `
  --output outputs/research-experiments-20261029/verification-reproduction/routerbench-seed7-pilot-main.json
```

`verify_runs.py` independently reconstructs actions, RNG, ridge/cosine updates,
prefix states, threshold/fallback decisions and cost/metric arithmetic. It does
not import the production numerical implementation. Repeat for every method,
domain, seed and variant. Canonical group reports use
`verification/<domain>-seed<seed>-<method>-<variant>.json` under the run directory.

`aggregate_results.py` requires all 1,098 manifests and all 108 independent group
reports. It verifies file hashes, refuses missing or extra conditions, and writes
per-seed tables, descriptive seed ranges and paired deltas. It never silently
drops a failed condition. `render_figures.py` uses those tables only;
`build_documents.py` uses their verified figures and requires ReportLab. On this
workstation the bundled Codex Python provides ReportLab; the project environment
provides Matplotlib. Both rendering steps are headless.

All scripts require assertions enabled and reject `-O`, `-OO` or an active
`PYTHONOPTIMIZE` before reading inputs or creating outputs. This prevents a
verification run from silently bypassing its checks.

All scripts refuse existing output destinations. Keep completed evidence intact;
use a fresh output path for a repeat verifier, or a separate full run folder after
intentionally revising a protocol. Do not delete old evidence to make a rerun fit.

`verify_replicate.py` checks one declared preference prior using `--fit-name`,
`--dimension` and `--seed`; it retains the fixed preference/prepared/cache paths.
`probe_cli.py` exercises real command failures on disposable copies and verifies
that the original experiment files remain unchanged.

The original execution sources are preserved byte-for-byte in
`reports/full-experiments-2026-10-29/evidence-scripts/`. Packaged analysis scripts
initially changed only the run-directory lookup so they can live here. Their
numeric logic is identical; `source-packaging.json` records those original
packaging hashes. A subsequent final-review fix adds an early optimization guard
to all seven current tools. `review/guard-changes.json` in the results package
records the before/after hashes; original execution sources and reports remain
unchanged. This distinction is explicit because a script's own hash appears in
its verification report. `verify_deliverables.py` independently recomputes all
plotted values from per-seed CSV rows and reads PDF contents using pdfplumber.
