# Offline research experiments

The full remaining research plan is tracked in
`openspec/changes/offline-research-experiments/tasks.md`. A completed input audit
does not establish paper-equivalent data, model identity or budget results.

## Complete input audit

Run from the project root with the already installed optional environment:

```powershell
$env:TOKENIZERS_PARALLELISM='false'
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
& .venv-embedding/Scripts/python.exe -m scripts.input_audit `
  --data-dir data `
  --prepared-dir outputs/local-pipeline-verification-20261029/prepared-identity `
  --output-dir outputs/research-input-audit-20261029 --batch-size 32
```

The command requires at least 8 GiB available physical RAM and no competing heavy
job. It uses the cached pinned tokenizer without loading PyTorch or making a
network request. Outputs are immutable: use a new output directory for another
audit. `inputs/` exports all frozen routing phases and the prepared preference
rows. `audit/lengths.jsonl` contains IDs, hashes and untruncated lengths, without
prompt text. `audit/manifest.json` contains aggregate statistics, selection rules,
tokenizer hashes and lineage. A completion manifest is atomically published last.

The rule was declared before the audit: choose the smallest cap among
512/1024/2048/4096/8192 covering at least 99% **separately** for routing
learning+tuning and preference training. Deployment and preference holdout lengths
are diagnostics only. Counts include BOS/EOS special tokens and exclude padding.

The 2026-10-29 audit covered 36,497 routing rows and 1,558 preference rows in 9.78
seconds of tokenization/reporting. It selected 2,048 tokens. Routing has no
truncation at that cap; 8/1,245 preference training and 5/313 holdout rows exceed
it. The 512-token smoke recipe remains unchanged. The selected recipe is
`configs/local_embedding_full.json`; GPU feasibility and full cache generation
have now passed independent readback. Preference inputs currently use the explicit
eight-arm `identity_match` development policy, whose model serving equivalence is
limited. The three unresolved arms are not admitted.

For tests on this Windows installation, give pytest a fresh project-local
`--basetemp outputs/<unique-test-directory>`: the default system pytest temporary
directory denies access. Do not change permissions or reuse an existing basetemp
directory that might contain needed files.

## Full caches and real prior

The 16-input longest-training-text feasibility probe passed on RTX 5070. Full
preference and routing caches contain 1,558 and 36,497 real 1,024-dimensional
float32 unit vectors. Their original token totals and truncation counts exactly
match the earlier audit. Readback verifies every vector-file/chunk checksum,
ordered ID/hash, verbatim input and recipe fingerprint. Completed-cache resume
succeeds in the base environment without PyTorch; changed inputs/recipe and
overwrite attempts fail without modifying artifacts.

The first real development prior uses the eight-arm `identity_match` pool,
1,245 training pairs, 313 holdout pairs, dimension 8, two projection epochs, five
prior epochs and seed 42. Every sampled triplet uses only training prompts.
Independent cosine calculations reproduce 186/313 correct holdout pairs (59.42%)
and all eight finite inverse-accuracy initialization coefficients. This is a local
development metric, not a comparison to the paper's differently prepared holdout.
These settings have not been tuned on routing outcomes.

Evidence and exact commands:
[aggregate report](../reports/full-embedding-prior-2026-10-29.json),
[handoff](full-embedding-prior-handoff-2026-10-29.md).

## Remaining work

The complete declared local experiment now contains 1,098 verified deployment
runs, including all five [controlled ablations](ablation-protocol.md), and 108
independently replayed method/variant groups. Six figures, the report and poster
passed table-to-figure and visual checks. The [results package](../reports/full-experiments-2026-10-29/README.md)
contains all seed tables and limitations. Final clean-environment reproduction
passes 441 tests with one skip; both smokes and the final review fix are verified.
See the [final handoff](final-research-handoff-2026-10-29.md).
Source/model ambiguities remain explicit; historical cost inputs are now
[calibrated](cost-inputs.md). No paid model call is needed for the current offline
route. The full eleven-arm reproduction target remains open.
