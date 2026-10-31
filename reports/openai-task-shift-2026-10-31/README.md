# Verified local MMLU-to-GSM8K task shift



Local verified Llama2-70B-chat/GPT4-1106-preview pair, seed42 only; train=5000, validation=398, test=10000.

Pinned official Microsoft NClassReranker/NClassCollator and DeBERTa-v3-large; actual added source-token head, deterministic cheap-match labels (ties cheap), source cap128 and fixed padding. AutoTokenizer fast implementation and library versions are recorded.

Input audit (rows retained): train: clipped1716/5000, unknown-token rows14; validation: clipped136/398, unknown-token rows1; test: clipped6658/10000, unknown-token rows8. The fast SentencePiece conversion warns that byte fallback is unavailable; unknown-token counts are disclosed rather than filtering questions.

Five epochs; lr1e-5, Adafactor/weight decay0.1, linear scheduler/warmup5%, FP32 parameters with FP16 autocast and gradient scaling. Local microbatch2/accumulation64 preserves effective128; final partial optimizer blocks average over actual examples. These wrapper details are declared local adaptations.

Checkpoint chosen only by minimum validation CE (earliest tie), selected epoch4; attempted optimizer steps200, applied200, AMP overflow skips0.

Independent verification replays validation inference for all five saved candidates and every test logit with a separately constructed source-token/head path, reconstructs all threshold accounting, and mutates held-out outcomes through the actual input extractor. The mutation checks identical training/validation input arrays; it is not a second full neural refit.

Supervised Hybrid training sees both arms' historical training outcomes; PILOT sees selected-arm feedback and retains a separate human-preference prior. Raw-text cap128 versus learned OpenAI vectors also differs. These comparisons do not isolate a single algorithmic factor.

No new LLM answers, reward-model judgments or paid embedding/API calls. Historical model costs are accounting units, not charges incurred by these runs.

Eight-versus-eleven model identity, author release/split/order and author baseline hyperparameters remain unresolved. This is a declared local adaptation, not the paper's exact Mistral/Mixtral D.1 comparison. No multi-seed significance claim from this single-seed baseline.



A separately trained static Hybrid classifier sees only the5000 warm MMLU labels and398 original MMLU tuning labels, then stays frozen. It never reuses the full-RouterBench classifier. Threshold0.5 was declared before outcomes. PILOT warms on the same5000 questions, then continues chosen-arm feedback through5000 before-shift MMLU and5000 after-shift GSM8K questions.

All15000 stream prompts are unique by normalized prompt group. Cohort choice is label-free; original order is retained within each task/cohort. This deliberate task ordering differs from the original mixed global stream. The paper's exact sampling/windows/model pair are unverified.

Both methods serve all10000 evaluation questions; warm examples are excluded from every reported quality/cost denominator. There is no budget controller in this online-adaptation experiment. The six declared windows include all evaluation/all before/all after/last1000 before/first1000 after/last1000 after.

PILOT uncertainty is sqrt(x^T A^-1 x) for the selected arm, recorded before feedback. Static classifier probabilities have a different meaning and are not plotted on that uncertainty axis. Trailing100-query curves are descriptive, not confidence intervals or independent repetitions.

These are previously studied benchmark questions, not a project-wide unseen final test. The restriction on GSM8K labels applies to this newly declared fit/selection; the original human-preference prior is retained. Single-seed adaptation evidence must not be presented as exact paper reproduction or general distribution-shift robustness.



Commands: independent Hybrid drift fit/replay + additive PILOT drift declare/run/full independent replay -> this report. Every adverse window remains in the CSV; no test window/threshold selection.
