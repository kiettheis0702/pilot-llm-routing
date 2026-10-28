# Preference readiness — 2026-10-28

**P01 audit and strict preparation are implemented. P04 has a reproducible provisional holdout. P02 is blocked by absent embedding artifacts/provenance; the complete eleven-arm prior is not ready.** Machine-readable evidence: [preference-readiness.json](../reports/preference-readiness.json). The source of truth for this scoped implementation is [the OpenSpec change](../openspec/changes/preference-readiness/proposal.md), with its spec, design and tasks.

## Measured eligibility

The pinned Arena 55K file has 57,477 rows. Its source size and SHA-256 pass verification. This audit extends the earlier aggregate Data Gate without rewriting it, the source CSV, prepared RouterBench tables, mapping configuration, or legacy harness sources.

| Strict processing stage | Rows |
| --- | ---: |
| Both model IDs literally match configured exact IDs | 576 |
| Remove 142 ties | 434 |
| Remove one blank A response and one nonstring/null B response | 432 |
| Remove 58 remaining multi-turn conversations | 374 |

These exclusions use the report's mutually exclusive primary-reason order. Independent reason incidence is also reported and can overlap: across the complete file there are 17,761 ties, 7,538 rows with a decoded multi-turn field, 5 rows with a blank prompt segment, 22/23 rows with a blank A/B response segment, and 111/120 rows with a nonstring A/B response segment. These are **row counts**, unlike the older report's null-segment counts. No CSV cells are missing. There are 56,901 rows outside literal exact-ID pairs. No invalid winner labels were observed. Reasons absent from the JSON counter have zero occurrences.

Only `gpt-3.5-turbo-1106` and `gpt-4-1106-preview` are admitted. The GPT-3.5 arm wins 49 of the 374 eligible comparisons; GPT-4 wins 325. Each battle counts once for its unordered pair and once for each participating arm. Six `proposed_alias` and three `checkpoint_unverified` mappings remain unchanged, unapproved, and excluded. A mapping with unequal source/target IDs cannot pass by simply marking it `exact_id`.

The old descriptive sensitivity scenarios remain separately labeled in the new report: 434 decisive pairs for literal IDs, 3,477 for all proposed mappings. Those broader counts allow multi-turn and incomplete responses and are not the strict training input. No public evidence examined in this change establishes the nine equivalences, so none were promoted.

## Preparation and holdout contract

`prepare_preferences(frame, mappings)` returns an in-memory pair table and aggregate diagnostics. It requires unique battle identities, one-hot numeric winner labels, distinct exact-ID arms, and exactly one nonempty string in each JSON prompt/response array. Malformed JSON, nested nulls/nonstrings, empty/blank content, ties and multi-turn rows are excluded with reasons. Responses are validated and then omitted from the returned pair table. No first-turn extraction from a whole-conversation label occurs.

`split_preferences(pairs, seed=42, holdout_fraction=0.2)` groups prompts by the existing whitespace-normalized SHA-256 hash, orders groups by SHA-256 of `seed:preference:prompt_hash`, and holds out `ceil(0.2 * group_count)` groups. At least one train group remains. This targets the fraction of **groups**, so row counts need not be exactly 80/20. It rejects insufficient groups, forged prompt hashes and invalid parameters. Labels do not determine partition assignment.

| Partition | Rows | Prompt groups | GPT-3.5 wins | GPT-4 wins |
| --- | ---: | ---: | ---: | ---: |
| Train | 299 | 298 | 36 | 263 |
| Holdout | 75 | 75 | 13 | 62 |
| Total | 374 | 373 | 49 | 325 |

Battle-ID and normalized prompt-group overlap are both zero. Assignment fingerprint: `552ea91a09b5638d159df0317c2ed310999fbe38c1f25685d3a1b9fc462a07ee`. Reordered input produces identical assignments and fingerprints in tests. Original prompt text is preserved in memory; canonical fingerprints include the retained data without publishing it. The command writes only aggregate JSON: it exports no conversations or training dataset.

No eligible prompt matches a full normalized RouterBench query or one of its segments. This checks exact text only; it cannot establish absence of semantic contamination. The source release, response/turn filtering, 20% holdout and seed 42 are local preparation decisions, not confirmed author preprocessing.

## Embeddings, triplets and blockers

The scoped filename/extension inventory examined this workspace and `D:\DOWNLOAD GG\PILOT_DATA`, skipping `.git`, `.venv`, `.agents`, `.pytest_cache`, `__pycache__`, `node_modules`, and directory/file links. It found **zero candidate artifacts**, with zero scan errors. No global home/cache/secret directories were searched. Filename inventory may miss unconventional artifact names and does not validate any cache. Local embedding model, version, dimension and provenance therefore remain `null`; no embedding model was downloaded or substituted, and no API was called. Dimension invariants cannot be tested against absent vectors.

The paper's first phase uses preference-conditioned query triplets and a smaller-model condition based on token cost. Its second phase freezes the query projection and learns model vectors from pairwise preferences. The current configuration contains no pinned size/cost ordering for that sampling rule; usable paper triplet count is **unknown**, not a fabricated positive value or an assertion that zero valid triplets exist. [Paper §2.2.1](https://arxiv.org/html/2508.21141v2#S2.SS2.SSS1)

Nine requested arms have zero strict eligible comparisons in both train and holdout: the three Claude arms, Llama 2, Code Llama, Yi, Mistral, Mixtral, and WizardLM. The full model IDs and zero counts are retained in the report. An eleven-arm prior requires verified source/checkpoint equivalences or an explicitly accepted alternative protocol, adequate per-arm train/holdout coverage, a validated embedding cache, a justified triplet size order, and actual trained/validated checkpoints. This change does not generate substitute/random priors or start training.

## Commands and verification

Run from the repository root using the existing CPU environment:

```powershell
.venv\Scripts\python.exe -B -m scripts.preference_data --data-dir 'D:\DOWNLOAD GG\PILOT_DATA' --report reports/preference-readiness.json
.venv\Scripts\python.exe -B -m pytest tests/test_preference_data.py -q -p no:cacheprovider
.venv\Scripts\python.exe -B -m pytest -q -p no:cacheprovider
& 'C:\Users\ASUS\AppData\Roaming\npm\openspec.cmd' validate preference-readiness --strict
git diff --check
```

- Initial TDD run: exit 1, 28 failures because the new helper behavior did not yet exist; no collection errors.
- Implemented fixture run: exit 0, **28 passed in 1.27s**. Covers strict selection, missing/blank/nested content, ties and malformed labels, identity/mapping rejection, deterministic grouped holdout, fingerprint sensitivity, negative split parameters, coverage/overlap/privacy, scoped cache inventory, and rejecting prepared hash mismatch before Parquet reading.
- Full current suite: exit 0, **151 passed in 2.21s**. Other scoped implementations were being developed in the same checkout; this is the suite actually observed at this point.
- Actual data command: exit 0, 9.62s command wall time; 374 eligible pairs, nine missing arms, readiness false. Pinned source and prepared-query hashes verified before parsing; report records source revision, config/code/input fingerprints and Python/pandas versions.
- Identical rerun: exit 0, 8.97s command wall time; report bytes match exactly. Report SHA-256: `e4ee71d0bbd50e1b885f758f674fbfab4fec49b15be82fa82749fedc981b6410`.
- Strict OpenSpec validation and `git diff --check`: exit 0. The new module was reviewed using `git diff --no-index -- NUL scripts/preference_data.py` (exit 1 denotes the expected added-file diff); `git diff --no-index --check` also checked new source, tests and documentation without whitespace findings.

No lint/typecheck/build/E2E/security scanner is configured or claimed by these pytest results. Relevant input validation and data-leakage boundaries were reviewed and tested. No raw writes, conversation exports, training, paid calls, or downloads occurred. Git staging/commit/push is handled separately by the coordinating agent under the user's authorization.
