# PILOT — LLM Routing Reproduction

Student research reproduction and experimental evaluation of **Adaptive LLM Routing under Budget Constraints (PILOT)** for Introduction to Machine Learning, Team 8, HCMUS — University of Science, VNU-HCM.

This repository is a student reproduction, not the authors' official implementation.

## Research reference

Pranoy Panda, Raghav Magazine, Chaitanya Devaguptapu, Sho Takemori, and Vishal Sharma. *Adaptive LLM Routing under Budget Constraints*. Findings of EMNLP 2025.

- [Paper and citation](https://aclanthology.org/2025.findings-emnlp.1301/)
- [Paper PDF](https://aclanthology.org/2025.findings-emnlp.1301.pdf)
- [Research site](https://ast-fri.github.io/contextual-bandit-routing-of-llms/)

## Current status

The paper has been accepted for the course project, according to the project handoff. The first **Data Gate audit is conditional**: pinned RouterBench 0-shot data is prepared for offline development, deterministic splits and a one-step feedback smoke check run, and Arena preference coverage has been measured. The exact paper dataset choices and several model aliases remain unresolved.

Read the [Data Gate findings](docs/data-gate.md) before running experiments. No PILOT implementation, preference training, or research result is claimed yet.

B01–B03 infrastructure is implemented: validated simulator/context contracts,
frozen split verification and learning-only reward/cost/empirical-regret evaluation.
See [usage and limitations](docs/baseline-infrastructure.md), the
[OpenSpec change](openspec/changes/baseline-infrastructure/tasks.md), and the
[verification log](docs/reproduction-log.md). Research baselines and preference
training remain later work.

Baseline policies and a bounded comparison CLI are now available; see
[commands, evidence and feature limitations](docs/baselines.md). Standalone
[cost control](docs/cost-policy.md) and the [PILOT mathematical core](docs/pilot-core.md)
have analytic tests. [Preference readiness](docs/preference-readiness.md) remains
blocked for an eleven-arm real prior. Synthetic smoke results are not paper results.

## Scope

- Reproduce preference pretraining, PILOT/LinUCB online learning, and the deployment budget policy.
- Compare baselines using the same evaluator, data splits, seeds, and budgets.
- Prioritize performance versus budget and cumulative regret experiments.
- Record changes from the paper, experimental configurations, limitations, and evidence for the poster.
- Prefer offline simulation from benchmark data to repeated paid LLM calls.

## Get the repository

```sh
git clone https://github.com/kiettheis0702/pilot-llm-routing.git
cd pilot-llm-routing
```

Python 3.11–3.13 and `uv` are required for the current data-audit workflow:

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
- Keep raw data, embeddings, checkpoints, API keys, and environment files out of Git.
- Record dataset revisions/checksums, model mappings, splits, seeds, and configurations.
- During online learning, reveal only the selected arm's reward to the learner.
- Keep tuning data separate from deployment evaluation.

See [CONTEXT.md](CONTEXT.md) for the agreed direction and open questions carried over from the project handoff.
