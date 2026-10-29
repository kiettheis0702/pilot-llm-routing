# Source and cost evidence — 2026-10-29

This is an investigation record. The follow-up [cost calibration](cost-inputs.md)
now reconstructs all 11,000 tuning query-arm costs. No generation or paid API was
called. The existing model mapping policy is unchanged.

## Source settings

The original RouterBench paper describes five-shot inference for MMLU,
HellaSwag, GSM8K, ARC and Winogrande, and zero-shot for MBPP/MT-Bench/RAG. It names
Together AI as the open-model provider. [Appendix A](https://arxiv.org/html/2403.12031v1#A1).
The repository's pinned [evaluation config](https://github.com/withmartian/routerbench/blob/cc67d1008bd8f3cf1e8040cc3ba4034d31b93c0c/configs/evaluate_routers.yaml)
selects `routerbench_5shot.pkl` and seven evaluation prefixes.

PILOT describes 36,497 queries, 11 models, 64 tasks, a 1,000-query tuning set and
a 10:1 learning/deployment ratio. Its preference experiment uses an unseen
500-example subset. It does not identify the exact downloaded shot file or Arena
release. [PILOT sections 3.1 and D.3](https://arxiv.org/html/2508.21141v2).
Our eight-arm preparation has a 313-row grouped holdout; that is another explicit
deviation, not the paper's 500-row holdout.

The local raw release has 86 eval-name labels, including custom tasks and three
MT-Bench labels. The pinned source groups benchmark prefixes during evaluation,
but does not establish a faithful 86-to-64 mapping for PILOT. Retain the existing
frozen splits and provisional zero-shot development choice. Do not relabel or
drop data to manufacture the paper's task count.

## Historical converter cost implementation

Pinned repository revision: `cc67d1008bd8f3cf1e8040cc3ba4034d31b93c0c` (2024-06-13).
The [converter cost function](https://github.com/withmartian/routerbench/blob/cc67d1008bd8f3cf1e8040cc3ba4034d31b93c0c/routers/abstract_router.py)
counts each prompt/response list element separately, then multiplies by its
per-token rate. The [price table](https://github.com/withmartian/routerbench/blob/cc67d1008bd8f3cf1e8040cc3ba4034d31b93c0c/routers/common.py)
labels rates USD/token and divides its numerical constants by **10,000,000**.
Do not silently change this divisor or present these repository constants as a
current provider quote. First reproduce recorded dataset costs on tuning data.

| RouterBench arm | Input constant / 10,000,000 | Output constant / 10,000,000 |
| --- | ---: | ---: |
| WizardLM/WizardLM-13B-V1.2 | 0.3 | 0.3 |
| claude-instant-v1 | 0.8 | 2.4 |
| claude-v1 | 8 | 24 |
| claude-v2 | 8 | 24 |
| gpt-3.5-turbo-1106 | 1 | 2 |
| gpt-4-1106-preview | 10 | 30 |
| meta/code-llama-instruct-34b-chat | 0.776 | 0.776 |
| meta/llama-2-70b-chat | 0.9 | 0.9 |
| mistralai/mistral-7b-chat | 0.2 | 0.2 |
| mistralai/mixtral-8x7b-chat | 0.6 | 0.6 |
| zero-one-ai/Yi-34B-Chat | 0.8 | 0.8 |

The pinned [requirements](https://github.com/withmartian/routerbench/blob/cc67d1008bd8f3cf1e8040cc3ba4034d31b93c0c/requirements.txt)
specify `tokencost==0.0.5` and `tiktoken==0.5.2`.
Inspection of the [published tokencost 0.0.5 wheel](https://pypi.org/project/tokencost/0.0.5/)
shows `count_string_tokens` uses tiktoken's model lookup and falls back to
`cl100k_base` on unknown model names. This is a benchmark approximation for
non-OpenAI models, not proof of provider-native billing tokenization. The wheel
was downloaded and read, not installed or executed.

## Remaining identity evidence

The RouterBench price table assigns Mistral and Mixtral canonical Instruct-v0.1
names the same price dictionaries as their short chat aliases. That supports a
pricing relationship but does not prove the served checkpoint. Claude-v2 has no
version-specific alias in that table. All three provider bridges remain
unresolved in `configs/model_mapping.json`.

## Saved source identities and next checks

Public source snapshots are in ignored `outputs/research-sources-20261029/`:

- `routers_common.py`: SHA-256 `27f853b177840036dbe2876c2fd0ac8fa68389f5d294fb73d94e30dd7c7105ec`.
- `routers_abstract_router.py`: `f8f724cac79a09183c223d64e4115cb6d62ed3c0e38d42670a2b445a541eacd0`.
- `requirements.txt`: `4fc8a405ea92ed25320c998fc37f44a18415624f14d9d53c761340faceabcafa`.
- Published wheel, 11,830 bytes: `706ac3162ce44ea105b957c3c49e55229c8a0a69ae66cf4df0f328c360436443`, verified against PyPI metadata.

Follow-up found that the recorded tuning costs are 10 times the literal source
formula for every arm. `configs/routerbench_costs.json` declares that multiplier;
the production exporter rechecks all 11,000 pairs before publishing. The
negative-sampling order now has an explicit local 1:1 input/output mix. Cost bounds
and the final evaluation protocol remain open. Continue provider identity
research independently. Exact serving equivalence remains a limitation even for
a canonical model-name match.
