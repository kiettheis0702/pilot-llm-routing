# Verified binary routing on RouterBench

All231 conditions (seven methods x three seeds x eleven learning/budget settings) independently replayed, including all677670 selected-feedback learning rounds,84 nested snapshots and deployment decisions.
Pool: GPT4-1106-preview and Llama2-70B-chat, a verified pair appearing in PILOT Appendix D.4.
Prior: select their two original directions/lambdas from the shared8-arm preference fit; retain the same64D query projection. This is not a fresh two-arm preference fit or an author-identical prior.
Fresh binary tuning:18 candidates on the original1000 tuning questions, independently replayed. PILOT alpha1, LinUCB alpha1, EpochGreedy window500. Shared positive ratio bounds come only from the selected candidates' tuning predictions.
Three32,270-round learning streams per method, with nested10/25/50/100% snapshots; no restarts for the smaller learning sizes. Same3,227 deployment questions and full-population GPT4 denominators as the primary study.

| Method | Mean quality/GPT4 at50% allocation | Mean recorded cost/GPT4 | Mean coverage |
|---|---:|---:|---:|
|epoch-greedy|70.8087%|47.2456%|100.0000%|
|explore-only|72.0070%|47.2913%|100.0000%|
|fixed GPT4|0.8987%|1.0178%|0.9606%|
|fixed Llama2-70B|42.2032%|6.2321%|100.0000%|
|linucb|72.0902%|46.5581%|100.0000%|
|pilot|72.4530%|46.3153%|100.0000%|
|random|59.9543%|33.0655%|100.0000%|

All compared methods use the same guarded bin controller, unchanged query order and historical prices. Skipped/unserved questions remain in denominators.
Regret is measured against the oracle within these two arms. Do not compare its raw total with an eight-arm oracle total as if the comparator were unchanged.
Original source verifier is preserved. A separate, hash-pinned wrapper selects the two parent theta rows by name and records both reference/wrapper hashes; all reference selection/update/budget equations are reused unchanged.

This is a local D.4 adaptation. It does not reproduce the unresolved GPT4-Mistral/Mixtral pairs or original D.1 HybridLLM comparison. No additional paid API requests.

Commands: tools.openai_study_binary_run declare -> tune -> freeze -> run, tools.openai_study_binary_verify tuning, then the verifier for each of21 seed/method groups. All commands run sequentially with one BLAS thread.
