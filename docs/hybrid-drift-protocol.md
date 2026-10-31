# Prospective free HybridLLM and task-shift extension

This extends the accepted follow-up plan. It does not replace completed results.
Task-shift training/evaluation and both independent readers are complete.
The separate full-RouterBench binary classifier also completed five epochs
and full independent neural replay. Real compatibility
and full-input audits pass; each recipe is sealed before its own historical fit. No generation, reward-model scoring,
embedding API, or paid service is used. All supervision is historical RouterBench.

Use the reviewed Microsoft best-route-llm commit
`b13c5bdaaf9a87af7587a9a83afc9b7219c2509d`, its actual NClassReranker,
NClassCollator and pinned LLM-Blender encode_texts. Backbone is public
`microsoft/deberta-v3-large` revision
`64a8c8eab3e352a784c658aef62be1662607476f`. Gather the added `<|source|>` token
representation; preserve dropout0.05/linear2-way head, max128 fixed padding,
det_2cls CE at match_t0 (ties prefer cheap). This is not a CLS/MLP proxy.
Use AutoTokenizer's fast tokenizer, matching the official builder's default;
record the actual tokenizer class/version and full-input token hashes. This
requires the free protobuf conversion dependency, unlike the slow Instructor
tokenizer used earlier. Verify compatibility before freezing either classifier.

The unmodified upstream default CLI has no det_2cls metric mapping at the pinned
commit. Use a declared local wrapper with validation cross-entropy and earliest
minimum epoch selection, five epochs, seed42. Preserve CLI defaults lr1e-5,
Adafactor, weight decay0.1, linear schedule/warmup5%, FP16 autocast with FP32
parameters and gradient scaling. Completed drift uses microbatch2/accumulation64
with effective128, which changes dropout draw order versus upstream8/16. Before
any binary fit, a hardware-only128-synthetic-example comparison showed official
8/16 feasible and faster; binary therefore uses8/16 with eval8 in a separate
additive adapter/protocol. See hybrid-binary-batch-check.md. The original
conservative binary declaration is retained as superseded-before-fit.
Disable DataLoader/BLAS parallel workers. Probe real
forward/backward and memory before protocol freeze; do not silently change
precision, architecture, cap or optimization after seeing outcomes.

Model pair is cheap Llama2-70B-chat / strong GPT4-1106-preview, whose dataset
identities are verified. This is a local binary adaptation. It is not the paper's
unverified Mistral/Mixtral D.1 pair. Supervised labels see both arms' historical
training rewards; PILOT receives only selected-arm feedback and also has its
human-preference prior. Neither this information difference nor raw-text versus
embedding inputs is a controlled single-factor comparison.

Two separately trained cohorts, with disjoint normalized prompt groups:

1. Binary deployment comparison: original routing learning partition for training,
   original1000 tuning queries for checkpoint selection, original deployment
   partition for evaluation. Compare the native Hybrid cheap-match probability
   threshold front (21 predetermined thresholds0..1 inclusive, all queries served)
   with existing seed42 PILOT binary curves. Thresholds are not selected on test.
   Its native threshold front does not share PILOT's guarded-stop controller or
   exact estimated allocation. Report recorded cost/true reward against the same
   GPT4 query population and show coverage explicitly; no same-budget dominance
   claim from mismatched spend. Also report fixed cheap/strong endpoints.
2. Drift: first5000 unique MMLU learning prompts warm training; next5000 distinct
   MMLU prompts before shift; first5000 distinct GSM8K prompts after. Preserve
   order within each task/cohort and reject prompt-group overlap. Checkpoint
   selection uses all original MMLU tuning queries (398), no GSM8K labels.
   The count is checked against the frozen parent protocol, not a handwritten
   assumed count. The earlier prospective199 assumption was corrected before
   any historical-data fit; the prepared dataset/splits were never changed.
   Train a separate static classifier on warm MMLU only; never reuse the binary
   classifier, which has seen GSM8K training labels. Fix threshold0.5 prospectively.
   PILOT seed42 warms on the same5000 prompts, then continues selected-feedback
   updates through before/after with original MMLU alpha and shared8-arm prior
   directions restricted to the verified pair. All10000 evaluation queries served;
   this is online task adaptation, not a budget-controller experiment.

These cohorts reuse an already studied benchmark. The exclusion of GSM8K labels
applies to the newly declared drift classifier/selection procedure; it is not a
claim that the dataset was unseen throughout the project. PILOT retains its
original human-preference prior. Report that prior's provenance separately from
the selected historical routing feedback used in this online stream.

The checked cap128 fast-tokenizer inputs retain every row. Clipped/unknown-token
row counts for drift train/validation/test are1716/14,136/1 and6658/8; partition
sizes5000/398/10000. For binary they are18415/268,568/14 and1846/28, with
sizes32270/1000/3227. These figures describe input information loss, not discarded
examples. The pinned fast SentencePiece conversion warns about missing byte
fallback; both these counts and the versioned recipe must accompany results.

Drift report shows every observation and last1000 before / first1000 after /
last1000 after, with true reward, recorded cost, same-window GPT4 ratios and
PILOT selected-arm uncertainty `sqrt(x^T A^-1 x)` before feedback. No invented uncertainty for
the static classifier. Author cohort sampling/windows and exact model pool are
unresolved, so this is not an exact reproduction of the paper's drift figure.

Before outcomes: seal exact IDs/source hashes/weights/code/configuration in
separate immutable protocols. After training: verify checkpoint hashes, selection,
token/label identity, replay saved logits and actual local inference samples,
independently reconstruct native fronts and online PILOT updates. Keep all
adverse results. Raw prompts/model weights stay outside public publication.

## Verified task-shift outcome

All5 drift epochs completed; validationCE minimum .49625347245 at epoch4.
Independent neural reader reproduced all5x398 validation and all10000 test
logits with maximum difference0. The actual input-extractor test-outcome
mutation preserved training/validation arrays; it is not a full neural refit.
PILOT all15000 actions/scores/uncertainties/final states and6 windows replayed
independently. Report: reports/openai-task-shift-2026-10-31, packageSHA
2856ecb52ea11245f4e02dc7ba282ae78d09e1ec8eeb310941588472e0d19853.
After shift PILOT65.605% true average reward versus Hybrid65.810%, all served.
No claim of PILOT winning after shift. The static model selectsGPT4 on allGSM8K.

The immutable task-shift report records its original producerSHA8f20f802...;
those exact historical bytes are retained in
tools/frozen_sources/hybrid_report_task_shift_20261031.py.txt. The current
report producer later gained the default-batch binary path/metadata handling.
This preserves source provenance without changing any task-shift report bytes.

## Verified native binary outcome

Report reports/openai-hybrid-2026-10-31:7 hashed files; packageSHA
a05b054621b07681d64fb20df37cc6cd0ecafb2708cd1fa76c898295b880f72e.
All5x1000 validation and3227 test logits independently replayed, maximum
difference0. Selected epoch4;1265 attempted updates,1259 applied,6 normal
GradScaler overflow skips. The frozen terminal log's optimizer_steps label
counts attempts; selection.json and the report explicitly separate both counts.
Native0.5 threshold:81.6664% GPT4 quality,23.1174% recorded cost,100%coverage.
This does not share a0.5 estimated-budget constraint with PILOT. All21 native
thresholds and2 separately calculated fixed endpoints remain in the report.
