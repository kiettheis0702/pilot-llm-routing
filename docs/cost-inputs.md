# Offline cost inputs

The frozen tuning set now establishes the cost scale of the local RouterBench
release. All **11,000 query-arm costs** agree with the historical cl100k counting
semantics and the public table's numerical constants expressed per **one million**
tokens. The public repository divides those constants by ten million, so using
that formula unchanged underestimates the stored costs by exactly 10x.
`configs/routerbench_costs.json` preserves both the original divisor and an
explicit dataset multiplier of 10. No deployment response or realized cost is
used to choose this multiplier. These are benchmark accounting values, not a
current API price quotation or proof of provider invoices.

Source identities and unresolved model/data questions are in
[the evidence record](source-cost-evidence-2026-10-29.md). The price file was last
changed in the repository's initial commit
`30e0925ac07a45108b9ee82086040a5664ea83e6` on 2024-03-18; its bytes agree with the
later pinned revision.

## Reproduction

Install the optional historical tokenizer into the base Python 3.11 environment:

```powershell
uv sync --locked --extra cost
```

Prepare its small public vocabulary once. This downloads a public tokenizer
asset, not a generated response or a paid API request:

```powershell
$env:TIKTOKEN_CACHE_DIR=Join-Path (Get-Location) '.cache/tiktoken'
& .venv/Scripts/python.exe -c "import tiktoken; tiktoken.get_encoding('cl100k_base')"
```

The actual export requires the local vocabulary with the pinned SHA-256
`223921b76ee99bde995b7ff738513eef100fb51d18c93597a113bcffe865b2a7` and
`tiktoken==0.5.2`. It has no network fallback. It counts original prompt/response
segments separately, preserving the historical converter behavior. Non-OpenAI
model names use the original benchmark's cl100k fallback; native provider
tokenizers are not implied.

```powershell
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
& .venv/Scripts/python.exe -m scripts.cost_inputs `
  --data-dir data --output-dir outputs/research-cost-inputs-20261029
```

The command checks RAM/jobs before loading heavy dependencies or data, verifies
the raw pickle's pinned size/hash before deserialization, validates the frozen
prepared inputs/splits and refuses existing output directories. It writes:

- `input-counts.json`: 36,497 query-only counts covering all frozen phases.
- `tuning-output-counts.json`: only the 1,000 tuning queries, with 11 output counts each.
- `estimator.json`: explicit prices and mean response-token counts learned only from tuning.
- `cost-size-{exact_id,verified_alias,identity_match}.json`: inputs for prior training under each evidence policy.
- `manifest.json`: source/config identities, file checksums, calibration results and limitations, published atomically last.

The triplet ordering uses input rate + output rate, equivalent to a declared 1:1
reference-token mix. PILOT does not specify this mix, so it remains a local
assumption. It matters for crossed rates such as GPT-3.5 versus Claude Instant;
it is not a universal claim that one is cheaper for every query. None of these
cost files admits an unresolved Arena model mapping.

## Verification and limits

The regression suite passes 360 tests, with the same one Windows symlink skip.
Independent checks statically read the original source constants without executing
them, reproduce the saved estimator, confirm native-tokenizer agreement on 183
queries and compare all 11 tuning means with a separate raw-data probe. A real
CLI run with multiplier 1 rejects all 11,000 mismatched pairs before creating any
output directory. A repeat against existing outputs fails without rewriting them.

The known 86-versus-64 task discrepancy, exact shot/release selection and three
Arena identity bridges remain unresolved. Cost bounds, final tuning settings and
fair baseline deployment behavior still need to be frozen. Calibrated costs alone
do not establish a completed budget experiment.
