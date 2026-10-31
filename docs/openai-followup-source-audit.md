# Primary-source followup — 31 October 2026

This is a bounded audit, not proof that undiscovered author code or metadata
does not exist. Preserve all historical data, source pins and admitted arms.

## RouterBench model identities and task taxonomy

The [RouterBench paper, Appendix A.1](https://arxiv.org/pdf/2403.12031)
identifies Together AI as the source of open-model outputs/reference prices.
It gives no checkpoint-version bridge for Mistral-7B-chat, Mixtral-8x7B-chat or
Claude-v2. Appendix A.2 describes 57 MMLU subjects and the other benchmark
families; it does not map the additional raw evaluation labels in our release
to the reported64 tasks. Appendix A.4 describes5-shot inference for the multiple
choice/math families, whereas our current complete release is0-shot. This does
not establish which release PILOT used; release identity remains unresolved.

The paper's405467 inference outcomes differ from the local401467
(36497 queries x11 stored arms). Matching query counts alone cannot resolve the
release/processing uncertainty. No rows or labels have been removed to force a
target count, and no three pending arm aliases have been promoted.

New read-only checks of [issue4](https://github.com/withmartian/routerbench/issues/4),
[issue5](https://github.com/withmartian/routerbench/issues/5),
[issue7](https://github.com/withmartian/routerbench/issues/7) and
[issue10](https://github.com/withmartian/routerbench/issues/10), including their
comments, do not supply the missing checkpoint or64-task mapping. Issue4's
maintainer reply confirms downloading the published pickle directly; issue5's
comments concern tokenizer warnings; issue7 concerns adding independently
generated model data. Issue10 asks about missing labels/score semantics and
provides no confirming answer in the inspected comments.

Evidence: `outputs/openai-followup-20261031/sources-secondary/comments-manifest.json`
records the four exact public request URLs, content hashes and comment counts.
This audit sends no messages or issue comments and calls no paid model API.

## Official baseline sources and local adaptations

- RouteLLM: pinned commit0b64fdafe049e596a3f5657c219329f24af24198,
  [MF training example](https://github.com/lm-sys/RouteLLM/blob/0b64fdafe049e596a3f5657c219329f24af24198/routellm/routers/matrix_factorization/train_matrix_factorization.py).
  Actual cache-only fits and independent verification are complete for the local
  matched847/211/500 split. Official architecture/example settings are retained;
  calibration-only, noise-free checkpoint selection is explicitly adapted.
  This reproduces the algorithm on our inputs, not author-identical results.
  The local comparison matches split/cache/model admission, but not optimization
  budget or latent dimension: PILOT uses its calibration-accuracy-selected
  64D15/20-epoch candidate; MF uses the official128D100-epoch example with
  calibration-BCE checkpoint selection. Report both protocols, rather than
  claiming an isolated architectural effect or equal training effort.
- HybridLLM: the original paper URL's repository
  `m365-core/hybrid_llm_routing` returned HTTP404 from GitHub's direct public API.
  This establishes that endpoint's observed state, not that all original code is
  unavailable. The official Microsoft `best-route-llm` repository includes the
  deterministic HybridLLM implementation, pinned
  b13c5bdaaf9a87af7587a9a83afc9b7219c2509d. It also includes later BEST-Route
  examples; do not treat those example settings as verified PILOT settings.
  Direct inspection also corrects an earlier implementation inference: its
  classifier reads the final hidden state at the added `<|source|>` token,
  not the default CLS token. The official collator adds that token, prepends it
  to the source, replaces embedded separator strings, and encodes only the
  source at source_maxlength128. The backbone must resize its input embeddings.
  Deterministic label0 means the cheaper model matches/exceeds the larger
  model minus match_t; ties go to the cheaper model. Retain this exact direction.
  Do not implement a plain CLS classifier and call it the same baseline.
  The default training CLI declares5epochs, learning rate1e-5 and batch8;
  these are current official-code defaults, not independently verified PILOT
  author settings. Other current CLI defaults are gradient accumulation16
  (effective batch128), Adafactor enabled, weight decay0.1, linear learning-rate
  schedule and5% warmup. These differ from later example snippets; select and
  freeze a declared source-based configuration before local training.
  New public DeBERTa-v3-large metadata is pinned at
  64a8c8eab3e352a784c658aef62be1662607476f, MIT/ungated. Its primary
  pytorch_model.bin is873673253bytes. The pinned public weights/tokenizer/config
  have now been downloaded and all four asset hashes verified locally; actual
  classifier compatibility and full-input audits passed. A separate5000-MMLU
  warm classifier completed all5 epochs and full independent neural replay for
  all5 validation candidates and10000 task-shift test logits (maximum difference0).
  The full-RouterBench binary classifier completed five epochs under separately
  frozen official8/16/eval8 settings, with all5x1000 validation and3227 test
  logits independently replayed (maximum difference0). Completed drift preserves2/64. No exact
  PILOT-author baseline configuration is inferred from these official defaults.
- Instructor-XL: official free checkpoint revision
  ce48b213095e647a6c3536364b9fa00daf57f436 and official encoder code
  9a9742c130cef6e028af9f3bd582204c5bcfe96e have been pinned and downloaded.
  The real offline GPU probe verifies768D output and instruction-masked
  mean/Dense/L2 pooling. The routing instruction is a declared local assumption,
  not an author-confirmed instruction. Full encoding is now complete and
  independently verified:38018unique/38055mapped rows,37 official-source neural
  replay samples (maximum absolute difference8.20e-08). Downstream858-condition
  evaluation is complete with78 independent replay groups and312 snapshots;
  see reports/instructor-core-2026-10-31. SlowT5 tokenization and the routing
  instruction remain declared local choices, not verified author settings.

## Gates still requiring evidence

Three additional arm checkpoint identities; author release/task aggregation;
author split/query order; unreported projection/pretraining details and
Instructor instruction. Continue runnable local experiments while preserving
these limits. A hypothetical alias match, matching headline count or stronger
local score does not close a gate.

## Additional Hybrid/drift source checks during Instructor encoding

The official Microsoft requirements use an unpinned Git LLM-Blender dependency.
We pinned its inspected source to33204d2712944b6b17996f7c079e74cd963ccc7c.
`encode_texts` uses fixed maximum-length padding; the DeBERTa builder uses
AutoModel and declares out_hidden_state_size from hidden_size. The tokenizer
builder adds no model-specific prompt for DeBERTa. Preserve the official
Hybrid<|source|> token/collator behavior rather than guessing a CLS recipe.
Four source URLs/hashes are retained in
`outputs/openai-followup-20261031/sources-secondary/hybrid-dependency-manifest.json`.

At the inspected Microsoft commit, train_router.py defaults to det_2cls but its
loss2metric dictionary lacks that key. A source-only dictionary lookup probe
confirms the missing entry; the unmodified training CLI was not run. The ranker
contains the deterministic CE loss itself. A local training wrapper must state
its validation/checkpoint metric explicitly, preserve that loss/backbone/head
and disclose the difference. Do not silently import later probabilistic metric
logic and call it an exact deterministic baseline. Trainer source/hash and
lookup evidence are in hybrid-trainer-source.json.

The initial drift cohort first5000MMLU warm + next5000MMLU + first5000GSM8K
is disjoint by query ID but has two normalized prompt groups shared between
warm/before, plus one repeated group inside before. These are prospective
cohort findings, not leakage across the verified main routing splits.
A label-free candidate keeps15000 distinct prompt groups, skips three duplicate
query rows when selecting the cohort and preserves source order within each
task/cohort. It does not alter the prepared dataset. Its exact IDs are retained
in drift-prompt-disjoint-candidate.json for a later frozen protocol. This cohort
choice is a local assumption; author sampling/window details remain unpublished.
