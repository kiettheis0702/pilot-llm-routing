# Instructor-XL full core sensitivity

All858 primary conditions,78 learning streams and312 nested snapshots independently replayed.
Same8 admitted arms, original1245/313 preference split, routing partitions/order, historical cost inputs and same-query GPT4 denominators as OpenAI core.
Real local hkunlp/instructor-xl pinned checkpoint:768D base representation,512-token cap, FP32; learned64D projection with30/40 epochs and original three seeds. Fresh tuning uses only tuning labels.
No additional paid API calls. BGE and OpenAI packages remain unchanged.

| Domain | Estimated allocation | PILOT quality/GPT4 | Recorded cost/GPT4 | Coverage | Quality change versus OpenAI (pp) |
|---|---:|---:|---:|---:|---:|
|routerbench|20%|81.2277%|19.7145%|100.0000%|+3.4211|
|routerbench|50%|89.1763%|46.9406%|100.0000%|+2.5810|
|routerbench|100%|97.3385%|88.6233%|100.0000%|-0.5978|
|mmlu|20%|6.8093%|2.4193%|9.4902%|+5.0908|
|mmlu|50%|84.0791%|47.5461%|100.0000%|+0.5512|
|mmlu|100%|98.2166%|90.4275%|100.0000%|-0.2270|

Recipe limits: the routing instruction is a declared local assumption, not confirmed author text. Instructor clips214 routing and52 preference inputs, while the OpenAI recipe uses full text; BGE has its own2048-token tokenizer/cap. These are representation-recipe plus retuning comparisons, not an isolated causal encoder substitution.
Production uses dynamic right padding; pinned official fixed512 padding/pooling/mask replay agrees within the recorded float tolerance. Every vector/role mapping/token length is audited; sampled real GPU outputs are independently checked with official source. This is not a second full neural-model implementation.
Eight-versus-eleven arms, release/86-versus64 tasks, author split/order and unpublished training details remain unresolved. MMLU is nested in RouterBench. Do not report matched-paper reproduction error or claim significance from three seeds.
All13 methods remain in CSV, including eight fixed models. The displayed adaptive curves are the five router families. Unserved questions remain in every denominator; early-stop coverage must accompany quality.

Commands: local encoder prepare/encode -> independent cache verifier -> Instructor study declare/fit/tune/freeze/run ->78 generic independent verify_runs.py groups with the Instructor protocol/index/cache paths -> this report.
Raw runs and this report are immutable. Instructor ablations are not included in this858-condition core sensitivity.
