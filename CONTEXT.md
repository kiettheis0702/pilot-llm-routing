# Project context

Updated: 2026-10-24. This is a concise summary of the supplied project handoff; paper details and dataset choices still require direct verification before implementation.

## Agreed direction

- Course: Introduction to Machine Learning, HCMUS; Team 8.
- Paper: *Adaptive LLM Routing under Budget Constraints* (PILOT), Findings of EMNLP 2025.
- Course registration status: Accepted, as confirmed in the handoff.
- Focus: online learning and contextual bandits, faithful reproduction, fair baseline comparisons, analysis, and a poster.
- Prefer Python and offline benchmark simulation with minimal recurring API costs.
- Current stage: repository setup, followed by W1 / Data Gate. No implementation or experiment is assumed complete.

## Method boundaries

1. Preference pretraining learns a shared representation and model priors.
2. Online bandit learning uses the chosen arm's feedback only. The handoff distinguishes this stage from budget-constrained deployment.
3. Deployment applies the paper's online cost policy. A score/cost heuristic must not be presented as the reproduced policy without evidence from the paper.

Verify Algorithms 1–3, equations, hyperparameters, and preprocessing against the official paper before coding. Label every deliberate change, including a local embedding model replacing the paper's embedding model, as a deviation.

## Next milestone: Data Gate

- Verify which RouterBench variant the paper uses; record revision, checksum, row count, tasks, and arms.
- Inspect query, reward, and cost fields, missing values, and model identifiers.
- Verify the preference dataset, licensing, model overlap, and usable pairs.
- Define and persist tuning, learning, and deployment splits with explicit seeds.
- Demonstrate one interaction: query → select arm → observe selected reward → update/log.

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
