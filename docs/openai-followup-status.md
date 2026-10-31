# OpenAI follow-up: evidence and remaining limits

Later reproducibility checks: **714 passing tests** and real CPU follow-up
replays on a fresh Windows runner. See the [validation record](followup-reproduction-validation-2026-10-31.md)
for exact coverage; these repeated checks do not add new scientific/W&B records.

Completed31 October2026: all declared local scientific studies, final review,
public W&B readbacks and Claude research handoff are complete. Completed
scientific packages below retain their exact hashes and adverse results.
Poster, slides, PDFs and previews are outside the editing scope.

**Current reading order:** this status page, the
[OpenAI core results](../reports/openai-core-2026-10-31/KET_QUA_VI.md),
[current Claude update map](CLAUDE_UPDATE_OPENAI_2026-10-31.md), then the
[clean-checkout restore guide](openai-reproduce.md). Dated BGE handoffs and
`CLAUDE_HANDOFF.md` are historical; they do not override the OpenAI evidence.

| Work | Verified evidence | Scope |
|---|---|---|
| OpenAI main | [858 conditions](../reports/openai-core-2026-10-31/KET_QUA_VI.md),78 independently replayed learning groups | Eight admitted models; three seeds; GPT4-normalized quality/cost |
| Controlled ablations | [240 conditions](../reports/openai-ablations-2026-10-31/README.md),30 independently replayed groups | Main+ablations=1098 OpenAI conditions; separate from historical1098 BGE |
| Preference500 | [Three seeds](../reports/openai-preference500-2026-10-31/README.md),actual test-label mutation refit | Local847/211/500; mean63.5333%; not author-identical or project-wide unseen |
| Budget policies | [240 new policy simulations](../reports/openai-budget-2026-10-31/README.md),all independently reconstructed |48 existing one-bin controls reused; predictions/order/outcomes fixed |
| Cost shocks | [18 aggregate cases](../reports/openai-cost-sensitivity-2026-10-31/README.md),independent cross-check | Fixed choices; main PILOT0/48 overspends at1x,23/48 at1.1x |
| RouteLLM MF | [Three cache-only fits](../reports/openai-routellm-2026-10-31/README.md),independent scoring and test-flip refit | Same500 pairs: MF68.0667% versus PILOT63.5333%; different dimension/training/selection |
| Feedback noise | [12 conditions](../reports/openai-noise-2026-10-31/README.md),all independently replayed | Uniform0/1/3/5% selected-feedback replacement; not forced flips |
| Verified binary routing | [231 conditions](../reports/openai-binary-2026-10-31/README.md),21 full learning-group replays | GPT4/Llama2 pair, fresh tuning; shared eight-arm human prior restricted to two arms |
| Instructor-XL | [858 conditions](../reports/instructor-core-2026-10-31/README.md),78 replay groups |38018 unique vectors; slowT5/local instruction/cap512; recipe comparison with fresh tuning |
| Online task shift | [15000 rounds](../reports/openai-task-shift-2026-10-31/README.md),full independent online and neural readers |5000 warm MMLU then5000 MMLU/5000 GSM8K; one seed; every query served |
| Native Hybrid binary | [Five-epoch fit](../reports/openai-hybrid-2026-10-31/README.md),all5x1000 validation and3227 test logits replayed | Official source-token DeBERTa head; original8/16/eval8 batch;21 predeclared thresholds,2 fixed endpoints |

The native Hybrid baseline and task-shift classifier are separate fits.
Supervised Hybrid receives both arms' training rewards; PILOT receives selected
feedback and keeps its human-preference prior. Raw-text cap128 and OpenAI full-text
vectors also differ. These local comparisons do not isolate one algorithmic factor.

At50% estimated allocation, OpenAI PILOT reaches86.5953% quality/GPT4 and
46.2348% recorded-cost/GPT4 on RouterBench, with100%coverage. Instructor's
corresponding quality is89.1763%, but at100%allocation it is slightly below
OpenAI. At20%estimated allocation on MMLU, main guarded-stop serves2.27% of
queries; the separately labelled skip variant serves97.31%. Neither outcome
replaces the preserved main result.

After the task shift, PILOT true reward/query is65.605% versus static Hybrid
65.810%, both100%coverage. Hybrid chooses GPT4 for every GSM8K query. PILOT
has99.6885% GPT4 quality at98.7623% GPT4 cost in that window. Do not claim
PILOT wins after shift, static routing fails, or a one-seed raw reward drop
isolates adaptation rather than task difficulty.

Unresolved prerequisites are documented in the [primary-source audit](openai-followup-source-audit.md):

- RouterBench checkpoint identity for Claude-v2, Mistral-7B and Mixtral-8x7B
  is insufficient to admit three extra arms. Arena naming evidence alone does
  not establish the RouterBench checkpoint. Eight arms remain eight.
- 86 raw task labels versus64 reported tasks, exact release/outcome count,
  author500 preference pairs, split/order and unpublished training details
  remain unverified. No label deletion or guessed alias bridge was used.
- The author Instructor instruction and complete baseline settings are not
  recovered. Declared local recipes are not claimed to be100% equivalent.
- Difficulty labels requiring new LLM judgments were not generated. A paid
  run would require separate explicit approval; no proxy is relabelled as that
  paper experiment.

The already approved OpenAI embedding run is historical:437 successful
requests and6867235 provider-reported tokens. Its calculated Standard price
wasUSD0.1373447 before tax, not a verified invoice. All follow-up fits/replays
reuse caches or free local weights. Recorded benchmark model costs are
accounting values; they are not new provider charges.

Research-checkpoint serial regression:681 passed,1 skipped36.47s. Fresh origin/main
baseline490 passed with the same Windows symlink-privilege skip. Four artifact
CLI interruption cases failed before the phase-aware recovery fix, then passed.
Fresh read-only review found no remaining material issue in that correction.
A clone includes source and report evidence. The separate
[139-file reproduction archive](openai-reproduce.md) now supplies prepared data,
paid OpenAI caches, core priors/tuning and protocol with SHA-256 checks.
A new locked Python environment has exercised restore, a byte-identical
seed-7 prior refit and one complete RouterBench/PILOT group (32,270 learning
rounds, 11 deployment conditions), with independent replay and resume checks.
See [release verification](release-readiness-2026-10-31.md) for exact coverage
and current regression results. The entire core/follow-up grid and a fresh
neural environment have not been rerun; their original results remain frozen.

See [follow-up reproduction commands](followup-reproduce.md) for the missing
preparation steps and the dependency order for every reported study family.
The original batch decision record is now packaged with a checksum; actual
new-machine Hybrid probes remain required. Preference splits are regenerated
from the restored inputs instead of requiring the old course output directory.

[Claude's update map](CLAUDE_UPDATE_OPENAI_2026-10-31.md) targets the actual
22-slide deck and one-slide poster, with exact figures, numeric replacements and
claims to qualify. It is ready for Claude's presentation edits. Old public W&B links support
historical BGE claims; use the new registry for new research claims.

The default Hybrid threshold0.5 yields81.6664% GPT4 quality at23.1174%
recorded cost; the matched seed42 PILOT0.5 estimated allocation yields72.4896%
at46.1573%, both100%coverage. Different controls/training information apply;
retain the whole measured curves, not a universal-dominance claim. Native
threshold sampling does not give a measured point at every cost.

## Verified publication

[Public registry and evidence](../reports/openai-publication-2026-10-31/README.md):
2498 new summary views,35402 expected metrics,1128 historical records preserved.
Project total3626. All were checked through anonymous API pagination; the
70-file artifact manifest was also fetched anonymously and compared entry by
entry. This is not a claim that browser UI or all files were downloaded again.
A real lost-reply probe recovered the exact accepted server identity; repeating
the completed import created no extra runs. No paid model API requests.

The SDK initially rejected resume='must' for a GraphQL-only imported aggregate.
The wrapper now follows the official existing-run artifact recipe using allow,
after proving the exact ID/config/metrics exists, carrying the same canonical
values/labels and verifying completion. A regression test and real upload/readback
passed. Source aggregate still has the same ID; project count remains3626.
