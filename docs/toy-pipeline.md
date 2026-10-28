# Complete synthetic pipeline

Run from the repository root after `uv sync --locked`:

```powershell
uv run --locked python -B -m scripts.run_toy_pipeline --output-dir outputs/toy --seed 7
```

This is a CPU integration fixture with no external data or provider calls. It fits
a preference prior from eight explicitly constructed two-dimensional vectors and
two arms, checks supported held-out accuracy, trains PILOT on 50 routing queries,
reloads its model and deploys on six separate queries. Ten other queries supply
synthetic tuning counts. Preference and routing identities are separate; routing
split seed is fixed at 42 and the training seed defaults to 7.

The output directory contains supplied embedding caches and comparison records,
the trained prior and its complete optimization settings, routing split CSV,
learning trace/summary, learned and deployed checkpoints, deployment trace/JSON,
and `summary.json` with source/lock hashes, Git revision/dirty status and runtime
versions. All inputs are synthetic and tiny, so the checked-in evidence contains
no real conversation text, embeddings or model weights.

The command refuses an existing output directory. A failure can leave partial
artifacts but does not write the final summary. Different seeds can yield an
unsupported or zero-accuracy prior; the command fails explicitly rather than
manufacturing a lambda. Default seed 7 is an integration fixture, not seed tuning
for a research result. Wall-clock timing and Git metadata vary between executions;
same-seed numerical traces, parameter arrays and metrics are reproducible.

The saved [execution](../reports/toy-pipeline-2026-10-28/summary.json) took 0.217 s
inside the command. Its projection loss decreased 0.198077 → 0 and pair loss
0.691887 → 0.395793. Both synthetic arms had 2/2 held-out pair predictions and
lambda=1. Learning reward was 49/50 with empirical oracle regret 1; deployment
served 6/6 with reward 6 and unchanged learned A,b matrices.

The cost-error fixture deliberately uses actual cost 0.5 per action. Synthetic
input/output counts and prices estimate 0.07 or 0.28 per action. Deployment records
estimated total 1.47 against budget 2, actual total 3, and actual excess 1. The
estimated feasibility flag is false for violation; the actual violation flag is
true. These are arbitrary fixture currency units, not dollars or provider charges.
This verifies cost-error reporting; it is not a statistical critique experiment.

Tests run the pipeline twice, compare numerical traces/metrics, load before/after
deployment checkpoints and assert equal matrices, check provenance fields and
reject output overwrite. Split CSV bytes use explicit LF so Git checkout preserves
the recorded manifest checksum; a Windows regression reproduced the prior CRLF
mismatch before this correction. Full research runs remain gated by
[implementation-status.md](implementation-status.md).
