# PILOT — LLM Routing Reproduction

Student research reproduction and experimental evaluation of **Adaptive LLM Routing under Budget Constraints (PILOT)** for Introduction to Machine Learning, Team 8, HCMUS — University of Science, VNU-HCM.

This repository is a student reproduction, not the authors' official implementation.

## Research reference

Pranoy Panda, Raghav Magazine, Chaitanya Devaguptapu, Sho Takemori, and Vishal Sharma. *Adaptive LLM Routing under Budget Constraints*. Findings of EMNLP 2025.

- [Paper and citation](https://aclanthology.org/2025.findings-emnlp.1301/)
- [Paper PDF](https://aclanthology.org/2025.findings-emnlp.1301.pdf)
- [Research site](https://ast-fri.github.io/contextual-bandit-routing-of-llms/)

## Current status

The paper has been accepted for the course project, according to the project handoff. Repository setup is the first implementation step. No dataset preparation, algorithm implementation, or experimental results are included yet.

The next milestone is the **Data Gate**: verify the dataset version and schema, map model IDs, define reproducible splits, and demonstrate one bandit interaction with feedback from only the selected arm.

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

Python dependencies and execution commands will be documented after the data and implementation requirements are verified. There is no runnable research pipeline at this stage.

## Data and reproducibility

- [RouterBench dataset](https://huggingface.co/datasets/withmartian/routerbench) is the candidate routing benchmark. Its [repository](https://github.com/withmartian/routerbench) is benchmark code, not official PILOT code.
- [Chatbot Arena conversations](https://huggingface.co/datasets/lmsys/chatbot_arena_conversations) is a candidate preference source; compatibility and preprocessing still need verification against the paper.
- Keep raw data, embeddings, checkpoints, API keys, and environment files out of Git.
- Record dataset revisions/checksums, model mappings, splits, seeds, and configurations.
- During online learning, reveal only the selected arm's reward to the learner.
- Keep tuning data separate from deployment evaluation.

See [CONTEXT.md](CONTEXT.md) for the agreed direction and open questions carried over from the project handoff.
