# HybridLLM native binary comparison



Local verified Llama2-70B-chat/GPT4-1106-preview pair, seed42 only; train=32270, validation=1000, test=3227.

Pinned official Microsoft NClassReranker/NClassCollator and DeBERTa-v3-large; actual added source-token head, deterministic cheap-match labels (ties cheap), source cap128 and fixed padding. AutoTokenizer fast implementation and library versions are recorded.

Input audit (rows retained): train: clipped18415/32270, unknown-token rows268; validation: clipped568/1000, unknown-token rows14; test: clipped1846/3227, unknown-token rows28. The fast SentencePiece conversion warns that byte fallback is unavailable; unknown-token counts are disclosed rather than filtering questions.

Five epochs; lr1e-5, Adafactor/weight decay0.1, linear scheduler/warmup5%, FP32 parameters with FP16 autocast and gradient scaling. Training microbatch8/accumulation16 preserves effective128; inference batch8. Final partial optimizer blocks average over actual examples. Wrapper details are declared local adaptations.

Checkpoint chosen only by minimum validation CE (earliest tie), selected epoch4; attempted optimizer steps1265, applied1259, AMP overflow skips6.

Independent verification replays validation inference for all five saved candidates and every test logit with a separately constructed source-token/head path, reconstructs all threshold accounting, and mutates held-out outcomes through the actual input extractor. The mutation checks identical training/validation input arrays; it is not a second full neural refit.

Supervised Hybrid training sees both arms' historical training outcomes; PILOT sees selected-arm feedback and retains a separate human-preference prior. Raw-text cap128 versus learned OpenAI vectors also differs. These comparisons do not isolate a single algorithmic factor.

No new LLM answers, reward-model judgments or paid embedding/API calls. Historical model costs are accounting units, not charges incurred by these runs.

Eight-versus-eleven model identity, author release/split/order and author baseline hyperparameters remain unresolved. This is a declared local adaptation, not the paper's exact Mistral/Mixtral D.1 comparison. No multi-seed significance claim from this single-seed baseline.



Binary uses the original CLI8/16 training and8-example evaluation defaults after a synthetic hardware-only feasibility/throughput check, before binary outcomes. An audited AST adapter reuses the immutable local producer and independent reader; original/adapter/source-tree hashes are recorded in the protocol. The conservative2/64 binary preparation was superseded before fitting; the completed separate drift classifier keeps2/64.

The21 predetermined thresholds0..1 inclusive are all retained; no test-based threshold selection. Fixed model endpoints are computed separately, since a probability exactly1 can still choose cheap at threshold1.

Both curves use the same deployment questions and GPT4 reference. Native Hybrid serves every query; PILOT uses its already verified guarded-stop deployment controller. They do not share an exact allocated-budget constraint. Assess actual recorded spend and coverage together; no same-budget dominance claim or fabricated interpolated point.

The matched PILOT file includes only seed42/full learning and eight declared allocation points. Its three other learning-prefix conditions are not mixed into this curve.



Commands: free assets -> Hybrid real compatibility probe -> binary input audit -> declare -> train -> independent Hybrid verifier -> this report. All raw fits/proofs are retained locally; weights/prompts are excluded from publication.
