# PILOT — LLM Routing Reproduction

Student research reproduction and experimental evaluation of **Adaptive LLM Routing under Budget Constraints (PILOT)** for Introduction to Machine Learning, Team 8, HCMUS — University of Science, VNU-HCM.

This repository is a student reproduction, not the authors' official implementation.

## Research reference

Pranoy Panda, Raghav Magazine, Chaitanya Devaguptapu, Sho Takemori, and Vishal Sharma. *Adaptive LLM Routing under Budget Constraints*. Findings of EMNLP 2025.

- [Paper and citation](https://aclanthology.org/2025.findings-emnlp.1301/)
- [Paper PDF](https://aclanthology.org/2025.findings-emnlp.1301.pdf)
- [Research site](https://ast-fri.github.io/contextual-bandit-routing-of-llms/)

## Current status

**OpenAI follow-up completed (31 October2026):** real cached
`text-embedding-3-small` vectors now feed858 main conditions and240 controlled
ablations: **1098 OpenAI conditions**, separate from historical1098 BGE.
[Current evidence and limits](docs/openai-followup-status.md) cover preference500,
RouteLLM-MF, budget/allocation and cost shocks, feedback noise, verified binary
routing, Instructor-XL858 conditions and MMLU-to-GSM8K task shift. Both separate
source-based Hybrid classifiers have complete independent neural readback.
All reported adverse results remain; this is an eight-arm local adaptation.

The [new public registry](reports/openai-publication-2026-10-31/README.md) maps
**2498 new condition/summary views and35402 verified metrics** to source CSVs,
producer scripts and public W&B URLs. All1128 old records were verified and
preserved, giving3626 current project records. Anonymous API readback verified
all configs/expected metrics; the separate70-file11MB artifact has exact remote
manifest and anonymous access checks. These counts are not live LLM calls or
independent datasets. The old19-file artifact is preserved.

Final serial regression: **681 passed,1 skipped**; origin/main baseline490 passed
with the same Windows symlink-privilege skip. Fourteen frozen packages and235
pinned references verify; their bytes also survive a clean Git-index export,
which rebuilds the identical2498-record payload without ignored caches.
The [Claude update map](docs/CLAUDE_UPDATE_OPENAI_2026-10-31.md) is ready for the
actual22-slide deck and one-slide poster. All7 presentation/reference hashes
remain unchanged. [QA and validation limits](docs/openai-followup-qa-2026-10-31.md).
No additional paid model API requests, billing/visibility changes or Moodle
submission were made by this follow-up. Three model identities and author
release/task/split/settings details remain unverified; no exact-paper claim.

**OpenAI core completed (31 October 2026):** real `text-embedding-3-small`
1536D caches now feed three retrained 64D priors and fresh tuning. All **858 core
conditions, 78 learning streams and 312 snapshots** completed; all 78 groups
independently replayed. [Vietnamese results and QA](reports/openai-core-2026-10-31/KET_QUA_VI.md)
include four SVG/PNG figures, same-query GPT-4 normalization and paired BGE data.
At 50% estimated allocation, PILOT reaches 86.60% GPT-4 quality at 46.23% recorded
cost on RouterBench; OpenAI does not improve all conditions. Low-budget MMLU
stops early under the declared bin/fallback rule; six cases were diagnosed and
retained. The original core checkpoint made no additional paid requests and retains its
eight-arm scope. Core-checkpoint regression: **596 passed, 1 skipped**, with the transient Windows file-denial
invocation and successful reruns retained. [Frozen execution evidence](reports/openai-core-2026-10-31/assembly.json).

**Offline OpenAI workflow (30 October 2026):** downloaded Batch outputs can now
be imported with partial recovery into verified 1,536D caches and the existing
training pipeline. Separate train/calibration/test fitting is tested, including
final-test mutation invariance. The [core migration protocol](configs/openai_core_protocol.json)
keeps the old preference split; the 500-pair study is supplementary. A bounded
real BGE check scored 326/500 on the new local split; this is not an OpenAI or
matched-paper result. [Workflow and evidence](docs/openai-workflow.md).
This workflow made no provider requests and preserves the frozen BGE study.

**Course package (30 October 2026):** [33-slide editable midterm and A2 portrait
poster](reports/course-submission-2026-10-30/README.md), source-linked paper
critique, two supplementary analyses, hosted-input reconstruction, and **1,128
verified historical runs uploaded to [W&B](https://forge.coreweave.com/wandb/kiettheis-university-of-science-vnu-hcm/pilot-llm-routing-team8)**.
All 13,314 metric values were checked against local evidence. The
[results artifact](https://forge.coreweave.com/wandb/kiettheis-university-of-science-vnu-hcm/pilot-llm-routing-team8/artifacts/evaluation-results/pilot-team8-verified-results/v0)
contains eight figures, the research report/poster and aggregate data (19 files).
Project visibility is **Public**. An anonymous connection verified all 1,128
finished runs, configs and 13,314 metrics; the signed-out browser shows the project.
The complete implementation is on **main**. No Moodle submission has occurred.
[Public experiment → script → run mapping](reports/publication-2026-10-30/experiment-map.csv).
[Claude handoff for slides and poster](docs/CLAUDE_HANDOFF.md) contains the current
results, source links, course requirements and remaining presentation work.
[Upload verification](docs/wandb-upload-2026-10-30.md).
[Reproduce and publish checklist](docs/course-reproduce.md). The frozen course
report records the earlier, pre-login snapshot; this paragraph is the current status.
Latest course regression: **490 passed, one Windows symlink test skipped**;
toy pipeline and 64-row RouterBench baseline smoke passed sequentially.

The paper has been accepted for the course project, according to the project handoff. The first **Data Gate audit is conditional**: pinned RouterBench 0-shot data is prepared for offline development, deterministic splits and a one-step feedback smoke check run, and Arena preference coverage has been measured. The exact paper dataset choices and several model aliases remain unresolved.

Implemented and tested: validated offline evaluation, Random/fixed/LinUCB/
ExploreOnly/EpochGreedy baselines, two-stage NumPy preference pretraining,
prior-informed PILOT, checkpoint readback, tuning-only cost estimation and fixed
router deployment with binning/rollover and separate actual-cost accounting.
A single command runs the complete synthetic pipeline below.

The [implementation status](docs/implementation-status.md) distinguishes completed
local research from unresolved paper equivalence. Read the [Data Gate findings](docs/data-gate.md),
[preference readiness](docs/preference-readiness.md), and [verification log](docs/reproduction-log.md).
Actual eleven-arm training remains blocked by three unresolved model mappings.
The [local experiment protocol](docs/experiment-protocol.md) and tuning-only
parameter selection are verified; the complete declared local budget/regret
experiments are now available in the [results package](reports/full-experiments-2026-10-29/README.md).
Synthetic smoke results and the 64-row RouterBench plumbing comparison are not paper results.

The [local embedding pipeline](docs/local-research-pipeline.md) adds evidence-aware
2/5/8-model preparation, resumable disk caches and staged fit/learn commands.
Previous research regression: **441 tests passed, one Windows symlink test skipped**
in a new locked Python environment. See the [final handoff](docs/final-research-handoff-2026-10-29.md).
The [tuning stage](docs/tuning-handoff-2026-10-29.md) has independently verified all
48 candidates across RouterBench/MMLU, including held-out mutation invariance.
All six declared research priors (64D/8D, seeds 7/42/2026) passed independent
readback. The [full experiment runner](docs/full-experiments.md) now executes
immutable prefix snapshots, budget grids and controlled ablations sequentially;
all **858 primary and 240 ablation runs** completed, with **108/108 groups**
independently replayed. Six figures, a nine-page report and an A1 poster passed
numeric and visual checks. Complete
BGE-M3 caches now cover 36,497 routing queries and 1,558 preference inputs. The
first real eight-arm development prior passed independent readback and holdout
checks; no paid API was called. At half the reference budget, PILOT trails
LinUCB on RouterBench and ExploreOnly on the nested MMLU subset. All budgets,
seed ranges, paired ablations and recorded-cost overruns are reported.
The [full input audit](docs/offline-research-experiments.md) covers 38,055 prompts
and selects the verified 2,048-token recipe: no routing truncation and 13 preference
inputs truncated. [Cache/prior evidence](reports/full-embedding-prior-2026-10-29.json)
and [current handoff](docs/full-embedding-prior-handoff-2026-10-29.md).
The [offline cost inputs](docs/cost-inputs.md) now cover all routing queries and
reconstruct all 11,000 tuning costs with an explicit dataset/source scale factor.
The [common deployment adapter](docs/common-deployment-protocol.md) now supports
PILOT and all baseline families. Real integration passed 13 methods × three smoke
budgets on 128 learning / 64 deployment queries, with independent arithmetic and
state replay. [Integration evidence](reports/common-deployment-2026-10-29.json).

## Scope

- Reproduce preference pretraining, PILOT/LinUCB online learning, and the deployment budget policy.
- Compare baselines using the same evaluator, data splits, seeds, and budgets.
- Prioritize performance versus budget and cumulative regret experiments.
- Record changes from the paper, experimental configurations, limitations, and evidence for the poster.
- Prefer offline simulation from benchmark data to repeated paid LLM calls.

## Get the repository

Moving to another Windows PC: follow the Vietnamese
[PC handoff and setup guide](docs/PC_HANDOFF_2026-10-29.md), including the
seven-file data transfer checksum manifest.

```sh
git clone https://github.com/kiettheis0702/pilot-llm-routing.git
cd pilot-llm-routing
```

Python 3.11–3.13 and `uv` are required. The complete implementation is on `main`:

```sh
git switch main
uv sync --locked
uv run --locked python -m pytest -q
uv run --locked python -m scripts.run_toy_pipeline --output-dir outputs/toy
uv run --locked python -m scripts.run_baselines --source toy --rows 100 --output-dir outputs/baselines
```

Use new output directories for each run; existing results are never overwritten.
These CPU commands generate synthetic data locally and make no model/API calls.
Verified in a fresh clone and new Python environment: 310 tests passed and both
commands completed; [execution evidence](reports/reproducibility-2026-10-29.json).
That is the historical committed baseline. This implementation
was additionally verified before committing from a byte-preserving source snapshot in a new
environment: 441 passed, one skipped and both smokes succeeded. [Current evidence](reports/full-experiments-2026-10-29/reproduction-post-review/report.json).
See [toy outputs and interpretation](docs/toy-pipeline.md), [baseline commands](docs/baselines.md),
[pretraining](docs/pretraining-core.md), [PILOT](docs/pilot-core.md),
[cost policy](docs/cost-policy.md), and [deployment](docs/deployment.md).

Separately, download and audit the public candidate datasets:

```sh
uv sync --locked
uv run --locked python scripts/fetch_data.py --data-dir ./data
uv run --locked python scripts/audit_data.py --data-dir ./data
uv run --locked python -m pytest -q
```

Downloads total about 455 MB and require no paid API calls. See [data/README.md](data/README.md) for external-disk paths and output details. This runs the data audit, not a PILOT training or evaluation pipeline.

## Data and reproducibility

- [RouterBench dataset](https://huggingface.co/datasets/withmartian/routerbench) is the candidate routing benchmark. Its [repository](https://github.com/withmartian/routerbench) is benchmark code, not official PILOT code.
- The legacy [Chatbot Arena conversations](https://huggingface.co/datasets/lmsys/chatbot_arena_conversations) source is gated. The separately audited public [Arena 55K candidate](https://huggingface.co/datasets/lmarena-ai/arena-human-preference-55k) has 57,477 rows; its equivalence to the paper's preference release remains unconfirmed.
- Keep raw datasets, real embeddings/checkpoints, API keys, and environment files out of Git. Small explicitly synthetic fixtures and their evidence are tracked under `reports/`.
- Record dataset revisions/checksums, model mappings, splits, seeds, and configurations.
- During online learning, reveal only the selected arm's reward to the learner.
- Keep tuning data separate from deployment evaluation.

See [CONTEXT.md](CONTEXT.md) for the agreed direction and open questions carried over from the project handoff.
