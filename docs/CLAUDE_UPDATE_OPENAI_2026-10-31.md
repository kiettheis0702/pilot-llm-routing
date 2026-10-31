# Claude: update instructions for the existing slide deck and poster

User request: Codex completes research and provides this update map; Claude
edits presentations. Codex must not edit the PPTX/PDF/preview files.

**Status: final research handoff.** All declared local research extensions,
independent checks and public W&B readbacks are complete. This file directs
Claude's presentation edits; Codex preserved the original PPTX/PDF/preview files.
Author/model/release prerequisites still listed below remain unresolved facts.

## Inspect these exact files

Current deck is **midterm/8.pptx,22 slides**, SHA-256
`38176f6266c5eb7cef5ef64ec7e8fbd8b2f9e8122cd1133a4b45d4f6e3e1f5d5`.
Current poster is **poster/8.pptx,1 slide**, SHA-256
`9c2806ab98863e5c8149f31f2215bc745087ac2bbd4a30d62299acd0bea0c737`.
`deliverables/midterm/8.pptx` is the historical33-slide version; do not apply
the current slide numbers to that file. Preserve the current design, editable
charts/tables and notes. Each slide needs notes, <=500 words;20–40 slides total.
Poster remains exactly one A2 portrait slide,420x594mm.

Read-only source extraction with shape IDs, text, notes and chart cached values:
`outputs/openai-followup-20261031/presentation-current.json`.
All presentation/preview/reference-file baseline hashes:
`outputs/openai-followup-20261031/presentation-baseline.json`.

## Authoritative result packages

| Evidence | Location | Status |
|---|---|---|
| OpenAI858 core, GPT-4-relative metrics, paired BGE comparison | reports/openai-core-2026-10-31/KET_QUA_VI.md and CSV/SVG files | complete;78 replay groups |
| OpenAI240 controlled ablations | reports/openai-ablations-2026-10-31/ | complete;30 replay groups |
| OpenAI supplementary preference500 | reports/openai-preference500-2026-10-31/ | complete;independent scores and actual test-label mutation probe |
| OpenAI fixed-choice cost shocks | reports/openai-cost-sensitivity-2026-10-31/ | complete;Python/PowerShell cross-check |
| Historical BGE1098 and critique30 | reports/full-experiments-2026-10-29/ and reports/course-submission-2026-10-30/ | preserve as historical BGE |
| Budget/allocation comparisons | reports/openai-budget-2026-10-31/ | complete;240 independently replayed conditions plus48 reused controls |
| Verified GPT4/Llama2 binary routing | reports/openai-binary-2026-10-31/ | complete;231 conditions,21 independently replayed learning groups |
| RouteLLM-MF preference500 comparison | reports/openai-routellm-2026-10-31/ | complete;three seeds and actual test-label mutation refit |
| Feedback replacement noise | reports/openai-noise-2026-10-31/ | complete;12 independently replayed conditions |
| Instructor-XL | reports/instructor-core-2026-10-31/ and docs/instructor-sensitivity.md | complete;858 conditions,78 independent replay groups,312 snapshots |
| MMLU-to-GSM8K task shift | reports/openai-task-shift-2026-10-31/ | complete; separate Hybrid classifier, all neural logits and online PILOT replayed |
| Full-RouterBench HybridLLM | reports/openai-hybrid-2026-10-31/ | complete; all5 validation candidates and3227 test logits independently replayed;21 native thresholds plus2 fixed endpoints |

New OpenAI/Instructor/Hybrid evidence has2498 verified public summary URLs in
[the new registry](../reports/openai-publication-2026-10-31/experiment-map.csv).
All35402 expected metrics and configs were read back without credentials;
1128 historical records remain intact. Use the registry's recipe/family/source
CSV columns to select the exact evidence. Do not reuse a BGE run URL for OpenAI.
The separate70-file artifact is linked in the publication README. Anonymous
verification is API/manifest evidence, not a claim of browser UI inspection.

## Map for the current22 slides

| Slide | Existing content needing attention | Required update |
|---:|---|---|
|1–5| Paper introduction, motivation, notation | Retain scientific content; adjust scope/count labels only where present. Eight admitted models remain eight. |
|6| Pipeline footer says BGE-M3 local1024D and substitution for OpenAI | Replace local embedder with real text-embedding-3-small1536D -> learned64D. Keep learning/deployment separation and historical LLM outcomes. Update notes too. |
|7| Preference training/holdout terminology | Main remains1245train/313calibration. Call313 calibration, not an independent final test. Add separate847/211/500 supplementary protocol only if explicitly labelled. |
|8| Online learning equations | Method unchanged; initialization now comes from OpenAI fits. Do not imply the budget applies during learning. |
|9| Bin policy and notes say implemented literally | State guarded estimated-affordability addition and explicit stream-stop rule. At20% MMLU all seeds stop after29/1275; global money remains. New skip/allocation variants must be labelled local extensions once verified. |
|10–11| Conditional theory/proof printing issues | Core theoretical critique unchanged. Do not claim practical cosine inherits the full linear-reward theorem. |
|12| Complexity | Dense implementation remains O(Kd^3) prediction and O(Kd^2) state. No rank-one inverse optimization has been implemented. |
|13| Scope table says BGE substitution,1098 conditions,0 paid API calls | New OpenAI study also has1098=858+240; keep separate from historical BGE1098. Embedder is1536D OpenAI. The old0-paid-call claim is obsolete: the earlier approved embedding run used437 successful requests; this follow-up makes no extra paid requests. Version/release/task/hyperparameter gates remain. |
|14| Tuning/split/metrics/inventory | Splits unchanged. Both domains still select PILOT alpha1,LinUCB alpha5,EpochGreedy window10. Main prior64D30/40. Prefer quality/GPT-4 and recorded-cost/GPT-4 on all requested queries; distinguish allocated estimated fraction from recorded-cost fraction. Main+ablation replay groups total108. |
|15| Old E1 plots and BGE headline utility values | Rebuild from OpenAI core CSV/SVG. At50% allocation: RB PILOT quality86.5953%,cost46.2348%,coverage100%; MMLU83.5279%,47.5276%,100%. LinUCB quality91.28%/86.96% is higher at this allocation. At100% allocation PILOT97.9363%/98.4436% quality,91.5709%/91.8124% cost. Use full precision CSV for charts. |
|16| Old regret4254/1865 and baseline values | Replace all plotted values/notes with OpenAI learning-prefixes.csv. Limit “lowest regret” to the compared adaptive/random baselines shown; do not claim every fixed-arm comparator loses more. |
|17| Heading says more learning helps PILOT most | Re-evaluate this claim from new learning-size curves; it does not describe the OpenAI curves. At half allocation, PILOT changes little across prefixes; EpochGreedy gains much more. Use the four immutable nested snapshots, not four restarts. |
|18| Old five-ablation figures and blanket conclusion about prior | Use OpenAI ablation tables below. Different effects on RB/MMLU; no blanket “prior always helps” or “all preference information removed”. Keep fixed-alpha/shared-bounds interpretation. |
|19| Paper/local comparison with mismatched axes | Both quality and recorded cost can now be normalized to same-query GPT-4, but8-vs11 arms and dataset/split uncertainty still prevent a matched-paper reproduction claim. Do not interpolate a fake measured point at25% cost or compute “reproduction error” against93%/25%. |
|20| Deviation analysis says no paid OpenAI, embedding substitution | That particular model substitution is resolved. Retain unverified three version mappings,86 raw labels/64 reported tasks, author settings/split uncertainty. OpenAI-vs-BGE comparison also changes full text versus13 previously truncated preference inputs and retunes; not isolated encoder causality. |
|21| Takeaways say PILOT gains most from more learning | Replace with qualified current findings. OpenAI is not uniformly better than BGE. Both positive learning-regret and adverse budgeted results must remain. |
|22| Traceability links to1,128 old runs | Retain old links only for BGE historical claims. Use the verified new experiment-map.csv URLs for OpenAI claims; do not manufacture run IDs or replace historical BGE numbers at old URLs. |

## Ablation numbers for slide18

Mean change in quality/GPT-4, **percentage points**, relative to the same seed/
domain/budget main64D OpenAI router at50% estimated allocation. All have100%
coverage at this point.

| Variant | RouterBench | MMLU |
|---|---:|---:|
| Prior only, zero online updates | -6.703 | -13.878 |
| Projection8D | -0.903 | +1.654 |
| One budget bin | -0.355 | -0.616 |
| Set every initialization lambda=1 | +1.550 | +0.324 |
| Zero initial arm-prior directions | +2.142 | -0.454 |

Zero directions still use the preference-trained projection and calibrated
lambdas. This does not remove all preference information. Online updates improve
the pretrained-only router, but direction/strength effects are mixed. One-bin is
not a universal improvement and must not be presented as the fix for stopping.

## Poster section updates

Keep W1's mathematical cosine/magnitude stress:600 versus1.2 regret atT2000,
alpha1 remains the same synthetic result. It is encoder-independent and must
remain labelled synthetic; do not relabel it as a new OpenAI empirical run.

For W2, choose one coherent dataset version throughout graph/caption/notes:

- OpenAI main PILOT/full learning:0/48 overspends at factor1.00; **23/48 at1.10**.
- Historical BGE counterpart remains0/48 and25/48. If both are shown, label both.
- Counts are experiment conditions, not queries or estimated probabilities.
- Cost shocks keep choices/rewards fixed; no rerouting and no provider charge.
- Source CSV gives all factors0.8/1/1.1/1.25/1.5/2 and population definitions.
  Do not reuse an old all1098 overspend count or median budget-utilization number
  with the new OpenAI labels; recompute from the current package.

The prior-calibration mean remains60.70% across three seeds for the main313
protocol (OpenAI186/190/194 correct versus BGE189/188/193). Supplementary500-test
accuracy is63.53% on a different local protocol; never swap these as the same
measurement. New test rows were seen in older project fits, and are not confirmed
to be the authors'500 pairs.

Keep future directions paired to demonstrated limitations. Newly tested budget
variants go in the evidence section only after their final verification; retain
their conditional/local status. Update public URLs/QR captions only from the
final verified publication registry. No changes to theory based on a better curve.

## Embedding payment wording

The approved, already completed Standard embedding run:38018 unique texts,
38055 mapped rows,437 requests,6867235 provider-reported tokens. At the cited
Standard rate the calculated price wasUSD0.1373447 before tax; this is not a
verified billing invoice. No new paid embedding or LLM-response request is made
by the follow-up experiments. Benchmark model costs are historical accounting
units, not charges made by this project.

## Verified budget and noise additions

Budget comparison: `reports/openai-budget-2026-10-31/`,240 independently replayed
conditions plus48 reused one-bin controls. All policies use the same main learned
scores, tuning costs and held-out order. MMLU three-seed mean:

| Estimated allocation | Policy | Quality/GPT4 | Coverage |
|---:|---|---:|---:|
|20%|guarded-stop (main)|1.72%|2.27%|
|20%|guarded-skip|68.90%|97.31%|
|20%|fixed-per-query|68.94%|99.92%|
|30%|guarded-stop (main)|20.17%|27.45%|
|30%|guarded-skip|75.29%|99.87%|

Add this explanation to slide9/20/21 notes or a clearly labelled local extension
panel. It demonstrates the consequence of stopping before future bin money is
released; it does not replace main results. Literal-stop matches guarded-stop
throughout the observed grid. One-bin is not a universal cure. The retrospective
P-lambda*C comparator sees future predicted scores/estimated costs, not true
reward, and is not a guaranteed global optimum. Use allocation-comparison.svg
and the package CSV rather than rebuilding these numbers from rounded text.

Noise sensitivity: `reports/openai-noise-2026-10-31/`,12 conditions, all
independently replayed. MMLU historical budget1,8 arms, selected learning feedback
uniformly replaced at0/1/3/5%. True average deployment rewards are respectively
68.8627%,68.7582%,68.4706%,68.5490%; coverage100% in all conditions.5% replacement
changes mean true reward by-0.3137percentage points. This is replacement
probability, not a forced5% label-flip probability. Noise is not injected into
the final evaluation labels. Zero-noise states/actions exactly match main.
Use noise-sensitivity.svg/per-seed.csv if adding robustness evidence to
slide20/21 or notes; preserve the local-split/model-pool caveat and the mild
nonmonotonic3%-to5% result. No claim of reproducing the paper's exact Table5.

## Verified supplemental additions

Binary routing is now independently verified in
`reports/openai-binary-2026-10-31/`:231 conditions,21 learning groups,
677670 selected-feedback rounds and84 nested snapshots. Pair: verified
GPT4-1106-preview/Llama2-70B-chat (paperD.4). Fresh tuning on1000 questions:
PILOTalpha1,LinUCBalpha1,EpochGreedywindow500. Prior directions/lambdas are
selected from shared8-arm preference fits; the same learned64D projection is
retained. Do not call this newly trained two-arm preference pretraining.

At50%allocation, mean quality/GPT4: PILOT72.4530%,LinUCB72.0902%,
EpochGreedy70.8087%, all100%coverage. PILOT recordedcost/GPT4=46.3153%,
LinUCB46.5581%. Learning regret is878.7333 forPILOT versus729.1 forLinUCB,
731.2167 forEpochGreedy and719.3 forfixedGPT4. Thus there is no universal
PILOT dominance. Regret uses a TWO-arm oracle and must not be compared directly
with eight-arm regret totals. FixedGPT4 at50%allocation has0.9606%coverage
because the declared guarded-stop controller stops early; its tiny deployment
quality is not its model accuracy. Show coverage and retain this limitation.

Use binary-routing.svg, means.csv,per-seed.csv andlearning-prefixes.csv for a
clearly labelled supplemental result/notes forslide15–17/20/21. The exact
GPT4-Mistral/Mixtral andClaude-Mixtral experiments still require model identity
proof; do not relabel this verified pair as those experiments. HybridLLM has
a separate verified native report at reports/openai-hybrid-2026-10-31;
its23 summaries are separate from the231 binary controller conditions.

RouteLLM preference comparison is now independently verified in
`reports/openai-routellm-2026-10-31/`. Same local train847/calibration211/test500,
same cached OpenAI vectors. Seed7/42/2026: MF338/340/343 correct versus
PILOT319/313/321. Mean MF68.0667%, PILOT63.5333%, difference+4.5333percentage
points for MF. This local result reverses the paper's preference-accuracy ranking;
keep it. Three seeds reuse the same500 test pairs. Never call them1500 independent
examples or a budgeted RouteLLM routing experiment. The official128D MF example
uses100epochs; calibration-only checkpoint selection and noise-free evaluation
are explicit local adaptations. Original author500 identity,11-arm pool and
historical test overlap caveats still apply. Actual seed42 test-label mutation
refit keeps all selected weight arrays identical (340/500 becomes160/500).
Use paired-preference500.csv and preference500-comparison.svg for replacement
text/figure if a preference-learning result is added to slide7/20/21 or notes.

Instructor-XL is now complete in `reports/instructor-core-2026-10-31/`:
858 conditions,78 independent replay groups,312 snapshots. Use
recipe-comparison.svg and instructor-routing.svg for a labelled extension on
slide20/21 or notes. At50%estimated allocation, PILOT quality/GPT4 is89.1763%
onRouterBench and84.0791% onMMLU, both100%coverage; recorded cost/GPT4 is46.9406%
and47.5461%. Changes versusOpenAI: +2.5810 and+0.5512percentage points. At100%
allocation Instructor quality is97.3385%/98.2166%, changes-0.5978/-0.2270points.
Do not say Instructor always wins or that PILOT beats all routing baselines.
Keep the local instruction/slowT5 tokenizer/512-token cap and fresh-tuning
caveats; OpenAI uses full text and another tokenizer. This is a recipe comparison.
Instructor clips214 routing and52 preference rows; none are dropped. Recovery
used batch1/fixed512 only for450 vectors; the original37568-vector prefix is
unchanged. Main Instructor calibration still313 pairs; no500-pair test implied.

Use the new public registry for every new URL. Preserve adverse results and
separate unavailable author/version prerequisites. Codex did not change any
slide/poster file; Claude owns those edits.

## Verified task-shift addition for slide20/21 or supplemental notes

Use `reports/openai-task-shift-2026-10-31/task-shift.svg` and
`window-metrics.csv`. This is one seed42 experiment with the verified GPT4/Llama2
pair:5000 warm MMLU queries,5000 before-shift MMLU,5000 after-shift GSM8K.
Both methods serve every evaluation query; no budget controller applies here.

| Window | PILOT mean true reward | Static Hybrid mean true reward |
|---|---:|---:|
| All10000 evaluation queries |73.0625%|72.6950%|
|5000 before shift |80.5200%|79.5800%|
|5000 after shift |65.6050%|65.8100%|
| First1000 after shift |64.7250%|65.4500%|
| Last1000 after shift |65.6500%|65.7750%|

After shift Hybrid chooses GPT4 for every query, therefore has100% GPT4 quality
and cost in that window. PILOT after-shift quality/GPT4 is99.6885% and
recorded-cost/GPT4 is98.7623%. Do not claim PILOT beats Hybrid after the shift,
that static routing fails, or that the raw reward drop alone measures adaptation:
the task and GPT4 reference quality both change. The trailing100-query curves
are descriptive, not confidence intervals. PILOT uncertainty is selected-arm
sqrt(x^T A^-1 x) before feedback, not the classifier probability.

The separate Hybrid model only sees5000 warm MMLU training outcomes plus398
MMLU tuning examples; selected epoch4 by minimum validationCE. PILOT continues
selected-arm feedback and retains its original human-preference prior. Hybrid
uses both arms' training rewards and raw text capped128 tokens; PILOT uses
OpenAI full-text representations. These information/recipe differences and
the single seed prevent a controlled single-factor or broad robustness claim.
All five validation candidates and10000 test logits were independently replayed;
PILOT actions/scores/states were independently reconstructed for all15000 rounds.

## Verified native HybridLLM addition

Use reports/openai-hybrid-2026-10-31/hybrid-binary.svg and its threeCSV tables
for a supplemental slide/panel/notes near15/20/21. All3227 deployment queries
and the GPT4 reference match the seed42/full-learning binary PILOT curve.
This is one source-based DeBERTa-v3-large classifier, not21 separately trained
models. Every predetermined threshold0..1 by0.05 is retained; no test threshold
selection. The pair is verified GPT4/Llama2, not the unresolved D.1 Mistral/Mixtral.

| Predeclared operating point, seed42 only | Quality/GPT4 | Recorded cost/GPT4 | Coverage |
|---|---:|---:|---:|
| Hybrid native probability threshold0.5 |81.6664%|23.1174%|100%|
| PILOT estimated allocation0.5 |72.4896%|46.1573%|100%|
| Fixed Llama2 |42.2032%|6.2321%|100%|
| Fixed GPT4 |100%|100%|100%|

The two0.5 values are DIFFERENT controls: probability threshold versus estimated
allocation fraction. They do not denote a matched allocated budget. Keep the
full measured front and coverage; do not interpolate a fabricated Hybrid point
at50%cost or use these two points to assert universal dominance. The three-seed
PILOT mean72.4530% elsewhere is a different aggregation from this72.4896% seed42
point. Do not swap them. The full curves also do not establish that PILOT loses
at every operating point. Preserve this adverse low-cost comparison honestly.

Recipe: same official source-token head/collator and fixed128 fast-tokenizer
input; actual32270/1000/3227 train/validation/test. Official micro8/accum16
and eval8 were selected by a synthetic hardware-only check BEFORE binary fitting.
Five epochs, selected epoch4 by validationCE0.5797713691. Attempted1265 optimizer
steps, applied1259,6 normal AMP overflow skips; report both counts. Inference
replayed all5x1000 validation and3227 test logits with maximum difference0.
The actual test-label mutation changes6200 outcomes while training/validation
arrays stay identical; this is not a second full neural refit.

Supervised Hybrid uses both-arm training rewards; PILOT uses selected feedback
and human-preference initialization. Raw-text cap128 versus full OpenAI vectors
also differs. These differences, one seed and local splits/settings prevent an
isolated algorithmic or exact-paper reproduction claim. No paid requests.

## Final delivery checks for Claude

Research/code tests:681 passed,1 Windows symlink-privilege skip; this is not a
PowerPoint layout test. Scientific packages14/14 and235 pinned references match.
All7 baseline presentation/reference files remain byte-identical. Do the actual
PPTX edits/render/layout/font/notes checks in Claude's workflow, preserving the
current design and editable objects. Old33-slide deliverable remains historical.
The new public artifact contains no slide/poster derivative. Never label these
local8-arm/single-seed supplements as exact author reproduction.
