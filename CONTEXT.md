# Project context

**Current entrypoint (31 October 2026):** [OpenAI evidence and limits](docs/openai-followup-status.md), [current Claude handoff](docs/CLAUDE_UPDATE_OPENAI_2026-10-31.md), [offline restore/replay guide](docs/openai-reproduce.md), and [follow-up reproduction commands](docs/followup-reproduce.md). The measured OpenAI core has 858 conditions plus 240 controlled ablations, separate from the historical 1098 BGE conditions. The expanded registry has 2498 new summary views and preserves 1128 old records; these are not independent datasets or live API calls. The later local presentation update records a 24-slide deck and one A2 poster; the Claude handoff identifies their hashes and supplementary note changes. Moodle submission is unconfirmed.

## Historical context through 30 October 2026

All uses of “current”, “complete”, “main” and “pending” below describe their dated checkpoint, not the current OpenAI state.

Publication update (2026-10-30): GitHub `main` is the complete submission branch.
W&B is Public; anonymous readback verified all 1,128 finished runs and 13,314
metrics. The current [Claude handoff](docs/CLAUDE_HANDOFF.md) supersedes historical
pending-publication notes. The user will have Claude author the final slide/poster;
no Moodle submission has been made. [Public links](reports/publication-2026-10-30/README.md).

Current status (2026-10-30): the complete declared eight-arm local experiment is
implemented and verified: 1,098 deployment runs, 108 independently replayed
groups, 441 passing tests and one Windows symlink skip, with report and poster.
See [current implementation status](docs/implementation-status.md) and
[final research handoff](docs/final-research-handoff-2026-10-29.md).
The 2026-10-24 notes below are the original handoff context; their pending
implementation items are historical. Exact eleven-arm paper equivalence and
course submission requirements remain unresolved where the current handoff says so.

W&B update (2026-10-30): 1,128 historical runs are uploaded and finished;
13,314 metric values were independently read back. Eight figures, the research
report/poster and aggregate data are attached as the verified results artifact.
Visibility was Team at upload; it is now Public. See [upload evidence](docs/wandb-upload-2026-10-30.md).
Latest regression: 490 passed, one Windows symlink skip. No paid model calls.

Windows runtime lesson (2026-10-30): a Python installed under AppData from
packaged Codex can be visible only inside Codex, while an ordinary CMD reports
`No Python at ...`. Both local environments now use Python 3.11.18 under
`%USERPROFILE%\.local\share\uv\python`, with an ignored local `.python-version`
pin. Before handing over CLI commands, verify the interpreter's physical path
and execute from an ordinary desktop process; checking file existence or a child
CMD that inherits Codex's virtualized filesystem is insufficient. See
[repair and verification](docs/windows-python-runtime.md).

Updated: 2026-10-24. This is a concise summary of the supplied project handoff; paper details and dataset choices still require direct verification before implementation.

## Agreed direction

- Course: Introduction to Machine Learning, HCMUS; Team 8.
- Paper: *Adaptive LLM Routing under Budget Constraints* (PILOT), Findings of EMNLP 2025.
- Course registration status: Accepted, as confirmed in the handoff.
- Focus: online learning and contextual bandits, faithful reproduction, fair baseline comparisons, analysis, and a poster.
- Prefer Python and offline benchmark simulation with minimal recurring API costs.
- Current stage: conditional W1 / Data Gate. The [audit](docs/data-gate.md) verifies public files, prepares provisional 0-shot routing data, and checks splits and selected-arm feedback. No PILOT training or experiment is complete.

## Method boundaries

1. Preference pretraining learns a shared representation and model priors.
2. Online bandit learning uses the chosen arm's feedback only. The handoff distinguishes this stage from budget-constrained deployment.
3. Deployment applies the paper's online cost policy. A score/cost heuristic must not be presented as the reproduced policy without evidence from the paper.

Verify Algorithms 1–3, equations, hyperparameters, and preprocessing against the official paper before coding. Label every deliberate change, including a local embedding model replacing the paper's embedding model, as a deviation.

## Data Gate follow-up

- Completed: pinned public RouterBench 0-shot/5-shot and Arena 55K files, full-file statistics, provisional 0-shot schema, seed-42 splits, and one selected-arm feedback/update smoke round.
- Resolve: exact paper variant and the mismatch between 86 raw task labels and 64 tasks reported in the paper.
- Resolve: original preference release and nine proposed model aliases/checkpoints. 3,477 decisive pairs is conditional on the proposed mappings, not a verified training set.
- Resolve: preference ties/multi-turn handling and a prompt-group holdout.
- Implement: tuning-only deployment cost estimation. Prepared `estimated_cost` values are currently null.

## Evaluation priorities

- First: performance versus budget and cumulative regret.
- Then: learning-bucket size, removal of preference priors, and cost-policy comparisons.
- Optional extensions: distribution shift, incorrect priors, or cost-estimation error.
- Share the evaluator and experimental conditions across baselines and PILOT.
- Generate figures from saved runs; distinguish published reference results from the team's measured results.

## Still unresolved

- Exact Moodle deadline and rubric. The handoff's eight-week schedule is an internal proposal.
- Team membership and assignments.
- Availability of complete official PILOT source code.
- Dataset variants, preprocessing, preference-model mappings, and embedding configuration.
- Exact regret reference, metric normalization, and deployment policy details.

## Source documents

- [Official paper](https://aclanthology.org/2025.findings-emnlp.1301/)
- [Paper HTML](https://arxiv.org/html/2508.21141v2)
- [RouterBench data](https://huggingface.co/datasets/withmartian/routerbench)
- [Candidate preference data](https://huggingface.co/datasets/lmsys/chatbot_arena_conversations)
- [Course assignment](https://hcmus-cs-ta.github.io/lab-book/intro-ml/project/statement/)

The original handoff remains a separate local source document. This summary does not imply that its research details have been independently revalidated.
