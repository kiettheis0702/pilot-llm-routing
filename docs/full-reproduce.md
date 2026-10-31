# Complete offline replay inventories

This extends the [bounded follow-up checks](followup-reproduce.md). It does
not replace frozen scientific reports or add W&B records. See the validation
record linked in README for what has actually finished on a named commit.

## Execution boundaries

- Use a **separate fresh checkout for each scope** below. Do not point the
  orchestration at an old scientific output directory. Core refits preserve
  restored original inputs in `outputs/full-reproduction/original-core-inputs`
  and then fit at the original relative paths. Non-restored files, including
  completed runs, must be rejected before that move.
- Windows, Python3.11.18 and the committed `uv.lock` are the CPU recipe.
  Keep all BLAS/test workers at one; check >=8GiB available physical RAM and
  no other heavy local job before starting. Wait if the check refuses.
- OpenAI embeddings are restored from the checksum-pinned public archive.
  These stages make no paid model requests. Public source/model/tokenizer
  downloads are inputs, not inference calls.
- Preserve `reports/`, the original139-file ZIP and the existing reproduction
  tags. A numeric disagreement is evidence to investigate, not permission to
  rewrite the published reference or relax comparisons silently.

## Prepare a fresh scope

After cloning the repository and selecting the verified commit, use PowerShell:

```powershell
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
$env:OPENBLAS_NUM_THREADS='1'
$env:TOKENIZERS_PARALLELISM='false'
$env:UV_CONCURRENT_DOWNLOADS='1'
$env:UV_CONCURRENT_INSTALLS='1'
$env:UV_CONCURRENT_BUILDS='1'
$env:TIKTOKEN_CACHE_DIR=Join-Path (Get-Location) '.cache/tiktoken'
if ((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory/1MB -lt 8) {
    throw 'Wait for at least 8 GiB available physical RAM'
}
# Also inspect active test/build/training jobs before installing.
uv sync --locked --python 3.11.18 --extra cost --extra tracking --extra openai
if ($LASTEXITCODE -ne 0) { throw 'Locked installation failed' }
.venv/Scripts/python.exe -c "import tiktoken; from scripts.cost_inputs import HistoricalCounter; tiktoken.get_encoding('cl100k_base'); print(HistoricalCounter().metadata)"
if ($LASTEXITCODE -ne 0) { throw 'Tokenizer preparation failed' }
& tools/reproduction-ci.ps1 -Stage inputs
if (-not $?) { throw 'Input preparation failed' }
```

`inputs` downloads/checks the139-file archive, restores/prepares twice, checks
34 pinned public source files and freezes the RouteLLM inputs. It does not
train a neural model. Check the commands' exit status before continuing.

## Run exactly one scope in this checkout

```powershell
& tools/reproduction-full.ps1 -Scope core
# OR, in a different fresh checkout:
& tools/reproduction-full.ps1 -Scope followups
# OR, in a different fresh checkout:
& tools/reproduction-full.ps1 -Scope mf
```

| Scope | Work performed | Completion check |
|---|---|---|
| `core` | Three new prior fits; fresh RouterBench/MMLU tuning; all858 core conditions |78 independent group readers; exact condition inventory and published served/reward/recorded-cost/estimated-cost comparison |
| `followups` | Six PILOT parents; all240 ablations;240 budget policies;12 noise conditions; fresh binary tuning and231 binary conditions |30 ablation and21 binary readers, complete budget/noise readers, full inventory/metric comparison |
| `mf` | A new pinned neural environment; three CPU fits and a real seed42 final-test-label mutation fit |Independent NumPy scoring; byte-identical mutation weights; actual CPU results and differences from original CUDA results |

MF is deliberately a **CPU-port replay**, including on a machine that has
CUDA. Its child processes hide CUDA, then restore the caller's device selection.
This uses `CUDA_VISIBLE_DEVICES=-1`, as specified by the
[pinned PyTorch2.8 documentation](https://docs.pytorch.org/docs/2.8/cuda_environment_variables.html).
CPU and CUDA random streams can produce different checkpoints/scores; the
summary records the difference and does not relabel CPU evidence as a repeated
CUDA fit. Fresh local CUDA runs use the native recipes in
[followup-reproduce.md](followup-reproduce.md) and a separate output tree.

The240 budget simulations exclude48 one-bin controls already covered by the
full ablation inventory. The original budget CSV contains both populations.
Do not count those48 controls twice as newly executed policy simulations.

## Evidence and hosted execution

The helper writes immutable summaries under `outputs/full-reproduction/` and
prints `FULL_REPRODUCTION_EVIDENCE_JSON` in the log. Scientific completion
requires all native commands and independent readers to succeed. The summary
rejects missing, duplicate or unknown condition IDs, unverified proofs,
nonfinite values and metric disagreements. It verifies all frozen report
packages and records source hashes. Core also records whether fresh input
artifacts are byte-identical to their preserved originals.

The [Full offline reproduction workflow](../.github/workflows/full-reproduction.yml)
uses standard public Windows runners, refuses private repositories, disables
Actions dependency caching and uploads no artifacts. Scientific scopes run
one at a time. A `gate` dispatch runs regression only on a separate remote
runner; it never certifies a scientific scope or consumes local workstation RAM.
This permits a guard fix to be tested while a long unchanged replay continues.

Instructor encoding/full-grid, both Hybrid fits and task shift remain separate
native CUDA recipes. Model downloads or a successful compatibility probe do
not establish that their full fits/readbacks finished. Keep pending work and
completed evidence distinct in the validation record.
