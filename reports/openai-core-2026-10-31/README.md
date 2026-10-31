# OpenAI core study — 2026-10-31

Completed858 primary deployment conditions,78 learning streams and312 snapshots. All78 groups independently replayed.
No additional paid API calls. Historical BGE results are preserved.

Representation: real text-embedding-3-small,1536D, full original text; projection64D,30/40 epochs; seeds7,42,2026.
Same8 admitted arms, partitions and stream order;1245 preference training and313 calibration pairs. Tuning-only selection.
MMLU is a nested RouterBench subset, so the two domains are not independent datasets.

| Domain | Estimated budget fraction | PILOT quality / GPT-4 | Recorded cost / GPT-4 | Coverage |
|---|---:|---:|---:|---:|
| routerbench | 20% | 77.81% | 19.23% | 100.00% |
| routerbench | 50% | 86.60% | 46.23% | 100.00% |
| routerbench | 100% | 97.94% | 91.57% | 100.00% |
| mmlu | 20% | 1.72% | 1.10% | 2.27% |
| mmlu | 50% | 83.53% | 47.53% | 100.00% |
| mmlu | 100% | 98.44% | 91.81% | 100.00% |

Budget fractions refer to tuning-estimated GPT-4 cost; plotted costs use actual historical cost on all deployment queries.
Quality is total reward divided by GPT-4 total reward on the same ALL requested queries. Unserved queries contribute zero reward.

## Paired representation comparison

OpenAI minus BGE, for PILOT full-learning conditions; all8 budget fractions and3 seeds are included:

- routerbench: mean quality delta -0.334 percentage points; range -5.680 to +4.585.
- mmlu: mean quality delta -2.760 percentage points; range -54.864 to +2.626.

## All-method comparison

Ranking by total reward among all13 methods at full learning;3-seed means. Cost and coverage remain available in the CSV.

| Domain | Estimated budget fraction | Best method | Best quality / GPT-4 | PILOT rank /13 |
|---|---:|---|---:|---:|
| routerbench | 50% | linucb | 91.28% | 4 |
| routerbench | 100% | pilot | 97.94% | 1 |
| mmlu | 50% | linucb | 86.96% | 4 |
| mmlu | 100% | pilot | 98.44% | 1 |

This includes fresh tuning and a different input representation recipe: BGE had a2048-token cap that truncated13 preference inputs; OpenAI uses full text. It does not isolate the encoder alone.

## Limits

- This remains an8-arm local study. Three model/version aliases remain unverified; no11-arm reproduction claim.
-313 calibration pairs set initialization strengths; they are not an independent preference test.
-86 raw task labels versus64 in the paper remain unresolved. No labels were removed to match the count.
-30/40 training epochs and64D are declared local choices, not verified author hyperparameters.
-Budget controller keeps the declared guarded affordability rule. Estimated-budget safety does not guarantee recorded costs never exceed allocation.
-Three seeds give descriptive variability, not confidence intervals. Paper93%/25% is not a directly matched target.
-No new OpenAI ablations,11-arm trials, Instructor-XL, HybridLLM or RouteLLM runs are included.

## Evidence and reproduction

Raw immutable outputs: `outputs/openai-core-20261031`. Protocol, input/cache/prior/code pins, tuning traces, learning snapshots, deployment traces and independent replay proofs are retained.
Commands (PowerShell; one command at a time, one BLAS thread):

```powershell
$env:OMP_NUM_THREADS='1'; $env:MKL_NUM_THREADS='1'; $env:OPENBLAS_NUM_THREADS='1'
.venv/Scripts/python.exe -X utf8 -B -m tools.openai_study fit
.venv/Scripts/python.exe -X utf8 -B -m tools.openai_study tune
.venv/Scripts/python.exe -X utf8 -B -m tools.openai_study freeze
.venv/Scripts/python.exe -X utf8 -B -m tools.openai_study run
```

Completed artifacts are validated on resume; changed/partial artifacts are rejected. Reports require a fresh destination.
