# Data Gate audit — 2026-10-24

**Result: conditional.** RouterBench 0-shot is usable for offline pipeline development. Exact reproduction of the paper and full preference pretraining are not yet cleared. Counts below come from complete local files, not dataset previews. Machine-readable evidence is in [reports/data-gate.json](../reports/data-gate.json).

## RouterBench: measured results

| Check | 0-shot | 5-shot |
| --- | ---: | ---: |
| Rows | 36,497 | 36,511 |
| Columns | 37 | 37 |
| LLM arms | 11 | 11 |
| Distinct raw `eval_name` labels | 86 | 86 |
| Excess rows with duplicate sample IDs | 0 | 14 |
| Rows participating in duplicate IDs | 0 | 28 |
| Missing reward cells | 0 | 154 |
| Rows with at least one missing reward | 0 | 28 |
| Missing cost cells | 0 | 0 |
| Finite reward range | [0, 1] | [0, 1] |
| Fractional reward cells, excluding missing | 84,905 | 84,905 |
| Cost range, USD | 0.0000038–0.05073 | 0.0000038–0.05073 |

The 5-shot duplicate IDs and incomplete reward rows occur in `arc-challenge`. Its 36,497 distinct IDs overlap all 0-shot IDs. Do not concatenate the variants or silently fill missing scores with zero. The preparation function rejects duplicate IDs and invalid/missing rewards.

0-shot contains 16 excess rows with identical prompt text and 14,042 MMLU rows. It has no missing values in any column. Reward values must remain floating point: converting them to Boolean or thresholding them would change the benchmark.

Complete outer cells do not imply nonempty generated text. Decoding response lists found 230 empty/blank response sequences in 0-shot: Llama 2 (9), Mixtral (155), Code Llama (63), and Mistral (3). These recorded generations and their scores are retained, not silently dropped. Response lists in both variants parsed successfully; 5-shot had no empty response sequences in this check.

The [official paper, Sections 3.1.1–3.1.2](https://aclanthology.org/2025.findings-emnlp.1301.pdf) states 36,497 samples, 64 tasks, 11 LLMs, 1,000 tuning samples, and a 10:1 learning/deployment ratio. It does not name the downloaded 0-shot/5-shot file or revision. The raw task-label count of 86 requires clarification; a different grouping or filtering may explain it. Matching the row count alone does not prove that 0-shot is the authors' artifact.

**Working decision:** use the pinned 0-shot artifact for a provisional pipeline. Preserve all rows and raw task labels. Do not describe this choice as confirmed paper preprocessing.

## Preference data: measured results

The handoff's [33K Chatbot Arena conversations](https://huggingface.co/datasets/lmsys/chatbot_arena_conversations) is gated. Its metadata was readable, but an anonymous request for its Parquet file returned HTTP 401. Its contents and exact overlap were not audited. No access restriction was bypassed.

The public candidate [Arena human preference 55K](https://huggingface.co/datasets/lmarena-ai/arena-human-preference-55k) was downloaded from the publisher's repository. The older `lmsys/lmsys-arena-human-preference-55k` URL resolves to this repository at the same checked revision. The dataset card cites the Chiang et al. paper cited by PILOT, but this does **not** establish that PILOT used this exact release.

| Check | Measured value |
| --- | ---: |
| Rows | 57,477 |
| Distinct model IDs | 64 |
| Duplicate battle IDs | 0 |
| Missing cells | 0 |
| Invalid one-hot winner labels | 0 |
| Invalid JSON prompt arrays | 0 |
| Model A wins / Model B wins / ties | 20,064 / 19,652 / 17,761 |

The pinned CSV has fewer model IDs than the dataset card's broad description; the audit uses file contents as the source of counts.

Its response JSON also contains nested missing values despite zero missing CSV cells: `response_a` has 120 null segments and 12 empty/blank sequences; `response_b` has 126 null segments and 14 empty/blank sequences. The current preference counts use valid prompts and winner labels, not response text. Any later response-based processing must handle these explicitly.

### Model mapping sensitivity

[configs/model_mapping.json](../configs/model_mapping.json) makes every proposed alias explicit. It does not strip arbitrary version suffixes. For example, `gpt-4-0613` is not merged into `gpt-4-1106-preview`, and `claude-2.1` is not merged into `claude-2.0`.

| Mapping scenario | Pairs with both arms mapped | Ties | Decisive pairs |
| --- | ---: | ---: | ---: |
| Literal exact IDs: GPT-3.5-1106 and GPT-4-1106-preview | 576 | 142 | 434 |
| All 11 proposed mappings, including unverified aliases | 4,970 | 1,493 | 3,477 |

The 3,477 figure is conditional on the proposed mappings. It is **not** a verified training-set size. It includes 2,993 single-turn and 484 multi-turn battles, with 87 excess repeated first prompts. The labels describe whole conversations; selecting only the first query for multi-turn battles needs an explicit protocol. Ties were excluded only when calculating decisive-pair counts, not erased from the source data.

| RouterBench arm | Candidate Arena ID | Decisive pairs involving arm |
| --- | --- | ---: |
| `gpt-3.5-turbo-1106` | `gpt-3.5-turbo-1106` | 1,110 |
| `gpt-4-1106-preview` | `gpt-4-1106-preview` | 1,221 |
| `claude-instant-v1` | `claude-instant-1` | 929 |
| `claude-v1` | `claude-1` | 539 |
| `claude-v2` | `claude-2.0` | 648 |
| `meta/llama-2-70b-chat` | `llama-2-70b-chat` | 597 |
| `meta/code-llama-instruct-34b-chat` | `codellama-34b-instruct` | 343 |
| `zero-one-ai/Yi-34B-Chat` | `yi-34b-chat` | 250 |
| `mistralai/mistral-7b-chat` | `mistral-7b-instruct` | 242 |
| `mistralai/mixtral-8x7b-chat` | `mixtral-8x7b-instruct-v0.1` | 824 |
| `WizardLM/WizardLM-13B-V1.2` | `wizardlm-13b` | 251 |

Each battle contributes to two arm counts. Provider aliases need review; the Mistral, Mixtral, and WizardLM mappings additionally lack matching explicit checkpoint versions. No prior-training dataset or fabricated priors are exported.

No decisive mapped preference first prompt exactly matched a whitespace-normalized full RouterBench query or individual RouterBench prompt segment. This is a narrow exact-text check, not proof that semantic contamination is absent. Preference train/validation splitting must group repeated prompts and retain a held-out set once the mapping/protocol is settled.

## Prepared local artifacts

`processed/routerbench_0shot/` under the chosen data directory contains:

- `queries.parquet`: `query_id`, `query`, `query_segments_json`, `task`, `query_hash`.
- `outcomes.parquet`: 401,467 query/arm rows with `query_id`, `arm_id`, `reward`, `actual_cost`, `estimated_cost`.
- `splits.csv`: IDs, prompt groups, task, split, and deterministic stream position.

RouterBench prompts are serialized Python lists. They are decoded using `ast.literal_eval`, never `eval`. Original text segments are retained. For provisional context text, all segments are joined with a blank line: taking only the first segment could discard the actual question after a system instruction. Role boundaries and multi-turn semantics are not reconstructed, and this is not yet a confirmed embedding recipe.

`actual_cost` retains the dataset's response-dependent historical USD estimate. The [upstream cost calculation](https://github.com/withmartian/routerbench/blob/cc67d1008bd8f3cf1e8040cc3ba4034d31b93c0c/routers/abstract_router.py) uses prompt/response token counts and prices. It is not a current price quote or an invoice. `estimated_cost` is intentionally null: using realized response cost as the deployment estimate would leak future information. A tuning-only output-token estimator remains to be built.

The seed-42 split has **1,000 tuning / 32,270 learning / 3,227 deployment** rows. Assignment is ordered by a hash of the seed and whitespace-normalized prompt hash. Whole duplicate-prompt groups stay together; deployment uses `ceil((N−1000)/11)`. This is a documented local implementation of the split sizes, not a recovery of the authors' seed or partition. There is zero query-ID or exact normalized-prompt overlap across splits.

The one-round smoke check observes a query, selects an arm with a seeded random smoke learner, reveals that arm's reward, records one learner update, and logs it. The learner receives no oracle column, response text, unselected reward vector, or realized costs before selection. Repeat reward requests for the same query are rejected. This interface check is not a PILOT experiment or baseline result.

## Provenance, access, and licenses

All three downloaded files use immutable Hugging Face revisions and verified Git LFS SHA-256 hashes in [configs/data_sources.json](../configs/data_sources.json). Downloaded pickles are checked before loading and use an explicit pandas/NumPy global allowlist. This loader is limited to these pinned files, not a general sandbox for arbitrary pickles.

- RouterBench: no license field or license file in the pinned dataset repository. The separate benchmark code repository has an MIT license; do not assume that establishes the license of every dataset component.
- Arena 55K: the pinned publisher dataset card declares `apache-2.0`.
- Legacy Arena 33K: its card distinguishes CC-BY-4.0 prompts and CC-BY-NC-4.0 outputs, with gated access conditions; no data file was obtained.
- Only scripts, pins, aggregate reports, and documentation belong in Git. Downloaded or derived conversations, model outputs, and large tables stay local.

## Remaining decisions before preference training

1. Confirm the RouterBench variant/task aggregation, or explicitly accept the provisional 0-shot protocol as a deviation.
2. Verify the nine proposed aliases/checkpoints and identify the preference release used by the authors, or document use of Arena 55K as a substitution.
3. Specify tie handling, multi-turn eligibility, prompt-group holdout, and per-arm preference coverage after filtering.
4. Implement and verify deployment cost estimation from tuning responses only.

The project can proceed with offline baseline engineering on the provisional routing data while these research choices remain open. It cannot yet claim a faithful, complete PILOT reproduction.
