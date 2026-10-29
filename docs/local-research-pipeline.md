# Local research pipeline

This change advances the verified numerical core to real local embeddings. It is a development experiment, not a claim of reproducing the paper's figures. Design and tracked work: `openspec/changes/local-real-data-pipeline/`.

## Model identity policies

The default `exact_id` still includes only the two matching GPT snapshots. `verified_alias` additionally includes the two historical Claude renames and WizardLM V1.2, each with pinned evidence in `configs/model_mapping.json`. `identity_match` also includes Llama 2, Code Llama and Yi based on canonical model/variant identity; serving configuration equality is not established. Claude-v2, Mistral and Mixtral remain unresolved and excluded. Full eleven-arm requests must fail until those sources are established.

The evidence and candidate counts do not silently change the old strict report. A new run records the selected policy and its own checksums. Source conversations and generated vectors remain in ignored local output folders.

Actual preparation on 2026-10-29: `verified_alias` admits five arms / 987 eligible single-turn pairs; `identity_match` admits eight arms / 1,558 pairs. The exact-ID baseline remains two arms / 374 pairs. Each preparation preserves grouped holdout and records per-arm coverage. The full-pool flag remains false.

```powershell
.venv\Scripts\python.exe -B -m scripts.research_pipeline prepare --data-dir data --output-dir outputs/my-preparation --mapping-policy identity_match --rows 128
```

Use a fresh output directory. `--require-all-arms` rejects the current incomplete pool. `--rows` selects only the saved learning split; it does not redefine train/deployment membership. These inputs do not yet authorize a paper-equivalent claim.

## Vector storage and training memory

The new disk cache stores little-endian float32 NPY vectors plus a checksum-protected JSON manifest. Loading verifies the file checksum and finite values in chunks, then uses a read-only memory map. The old small JSON cache remains supported. Incomplete generation has no completion manifest and cannot enter training. Projection reads bounded chunks; triplet selection preserves the same ordered uniform pair distribution without building its Cartesian product, and training reads vectors as needed.

## Optional Windows GPU environment

Keep `.venv`/`uv.lock` for CPU work. The encoder uses a separate `.venv-embedding` with a hash-pinned lock. This is a Windows CPython 3.11 environment; the CUDA 12.8 wheel supports Blackwell. The runtime must still pass an actual GPU operation before a real run. It downloads several GB of packages/model weights and requires disk space, but no inference credits.

```powershell
# Run from the repository root, with uv on PATH.
.venv\Scripts\python.exe -B -c "from scripts.resources import require_resources; print(require_resources())"
uv venv .venv-embedding --python .venv\Scripts\python.exe
uv pip sync --python .venv-embedding\Scripts\python.exe --require-hashes --only-binary=:all: requirements-embedding.lock
.venv-embedding\Scripts\python.exe -B -c "import torch; print(torch.__version__, torch.cuda.get_device_name(0)); print((torch.ones(2,device='cuda')*2).cpu())"
```

Do not proceed past a failed preflight. Use one worker; set `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS` to `1`. No system CUDA toolkit change is required by these wheels.

## Real local encoding

```powershell
.venv-embedding\Scripts\python.exe -B -m scripts.local_embeddings --inputs outputs/my-preparation/routing-inputs.json --output-dir outputs/my-routing-cache --download --batch-size 2
# An interrupted run can continue only with identical inputs and recipe:
.venv-embedding\Scripts\python.exe -B -m scripts.local_embeddings --inputs outputs/my-preparation/routing-inputs.json --output-dir outputs/my-routing-cache --resume --batch-size 2
```

`--download` permits only the pinned public BGE-M3 files. Without it, files must already be cached. CLS pooling, normalization, precision and the exact max-length cap are fingerprinted. The default 512-token cap is for the first probe; the manifest reports the number of truncated inputs, original token totals and maximum length. A full-data decision must inspect these numbers. Encode `preference-inputs.json` separately with the same recipe before fitting. No automatic CPU or paid-provider fallback is used.

## Fit and learning comparison

`fit` requires a cost-size JSON with exactly `sizes` (positive values for all admitted RouterBench arms), `provenance` (nonempty description of historical token-price interpretation), `source_urls` (nonempty HTTPS sources), and `synthetic: false`. The [cost-input exporter](cost-inputs.md) now supplies calibrated historical rates with a declared 1:1 input/output reference-token mix for cost ordering. Do not replace it with mean realized query costs. A syntactically valid provenance string is a declaration, not independent source verification.

```powershell
.venv\Scripts\python.exe -B -m scripts.research_pipeline fit --prepared-dir outputs/my-preparation --cache-dir outputs/my-preference-cache --cost-sizes path/to/reviewed-cost-sizes.json --output-dir outputs/my-fit
.venv\Scripts\python.exe -B -m scripts.research_pipeline learn --prepared-dir outputs/my-preparation --data-dir data --cache-dir outputs/my-routing-cache --fit-dir outputs/my-fit --output-dir outputs/my-learning
```

The CLI rejects synthetic inputs and requires sufficient resources. The first fit uses bounded 8-dimensional projection, 2 projection epochs and 5 prior epochs as explicit development defaults. These are not tuned paper hyperparameters. A zero-accuracy or unsupported prior is saved as a diagnostic and blocks learning. PILOT, Random, LinUCB, ExploreOnly, EpochGreedy and each fixed-model policy share the selected stream and projected contexts. This command reports learning feedback and historical actual-cost metrics, not budgeted deployment figures.

## Remaining research work

Verified on 2026-10-29: the isolated environment installed from the hash lock;
CUDA FP16 computation and BGE-M3 generation passed on RTX 5070. The 128-query
learning-split probe produced 1,024-dimensional float32 unit vectors with zero
truncation (maximum 493 tokens, 23,664 tokens total). Peak allocated/reserved VRAM
was 1,167,440,384 / 1,191,182,336 bytes. Recorded encoding time was 53.99 seconds,
including first initialization and any download, not isolated inference time.
Completed-cache resume succeeded using the CPU environment without Torch;
changing the recipe on resume was rejected. Weight SHA-256 was checked against
the pinned Hugging Face ETag. See `reports/local-pipeline-2026-10-29.json`.

Full caches and the first real eight-arm prior are now verified; see the
[current evidence](../reports/full-embedding-prior-2026-10-29.json). The original
128-row smoke remains separate. `--checkpoint-batches 32` reduces progress-file
writes during full encoding while retaining a GPU batch size of two. On
interruption, rows after the last verified prefix are re-encoded. The default is
one checkpoint per batch; valid intervals are 1–64.

`configs/research_protocol.json` records unresolved source/release and task
grouping, plus null cost bounds and baseline deployment score protocol. Three
model bridges remain unresolved. Historical price/count inputs are now available.
Next: freeze the evaluation protocol, tune on the tuning split, implement common
deployment behavior, run held-out budget sweeps and ablations, then produce figures
and the poster. Never select hyperparameters from deployment outcomes.
