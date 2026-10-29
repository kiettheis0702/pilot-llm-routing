# Execution ledger — frozen local experiment protocol

Plan: `configs/frozen_experiment_protocol.json`; specification:
`openspec/changes/offline-research-experiments/specs/research-experiment-protocol/spec.md`.
Base commit: `47ce94384f9c4a709f9bc85a02c7eb344cce75ae`.

Continue the existing `codex/data-gate` workspace, without committing or pushing.
The frozen numerical choices and pinned documents remain unchanged.

## Interfaces and execution order

1. Complete and independently verify the remaining 64D and 8D priors. Publish
   a separate index of exact checkpoint/verification identities.
2. Test then implement sequential learning with nested-prefix snapshots,
   initial-only and controlled-prior variants, immutable completion manifests,
   exact-identity resume and a complete inventory of 1,098 deployment runs.
3. Test then connect the runner to pinned real inputs, frozen tuning parameters,
   domain-specific estimators, unchanged splits, and all six verified priors.
4. Run a complete first condition, independently replay its actions/metrics and
   measure elapsed time. Run remaining conditions sequentially with RAM gates.
5. Verify inventory, paired ablations and aggregate metrics; then produce figures,
   report/poster and final reproduction evidence. These remain required work.

Interface check: prior artifacts carry seed/dimension/epoch/embedding identity;
learning consumes the matching projection and prior per seed. The earlier 8D
2/5-epoch smoke is never a research input. Tuning states are never learning states.
Deployment consumes frozen prefix checkpoints. The prior-only variant has zero
learning rounds and labels its learning split as an unused disjoint reference.
Single-bin deployment references the exact main full-learning checkpoint.

Ruling: retain the completed shared workspace and record this human-readable
ledger instead of generating another skill workspace — existing artifacts and
uncommitted implementation span earlier stages; duplicating them would consume
disk and complicate provenance. This changes bookkeeping, not experiment choices.

## Progress — 2026-10-29 continuation

- Initial preflight: 10.593 GiB available, no detected heavy jobs; C: free 49.69 GiB.
- All five remaining prior fits completed sequentially. Independent readback
  verified source/cache/checkpoint identity, training-only triplets, projections,
  final losses and every holdout coefficient. 64D seeds 7/42/2026: 189/188/193
  correct of 313; full 8D seeds 7/42/2026: 183/189/197. Seed 42 64D was reused.
- Runner RED/GREEN: 19 new tests; full suite 426 passed, one Windows symlink skip
  (13.46 s). Covers inventory, uninterrupted prefixes, held-out mutation,
  immutable resume/corruption, no-feedback prior-only, matched single-bin reuse,
  real pin validation and actual Windows experiment-process discovery.
- First real RouterBench seed-7 PILOT group: 32,270 learning rows, four snapshots,
  eleven deployment conditions; 35.806 s including validation. Independent math
  replay matched every learning action/update, all snapshot states and 29,076
  served deployment decisions, estimates, costs and metrics.
- All primary groups completed: 78 full learning streams and 858 deployment
  manifests across both domains and all three seeds. Full independent replay,
  ablations and final aggregate conclusions are still required.
- Four real CLI negative probes passed on copies: existing output without resume,
  changed protocol, interrupted artifact and corrupted completed trace. All
  original learning files retained identical hashes. Earlier real resume also
  read back the completed seed-7 PILOT group without changing it.
- The sequential queue is now running 240 ablation conditions, then independent
  replay of all 108 method/variant groups and strict complete-inventory aggregation.
- Runtime source hashes are part of every experiment identity. Do not edit
  `scripts/*.py` while this execution/replay queue is active. The analysis and
  document drivers live separately under the ignored run directory; preserve
  their exact executed sources when packaging final reproduction evidence.
- Prepared drivers, not yet completed: `aggregate_results.py`, `render_figures.py`
  (six figures), `build_documents.py` (report/poster). They reject incomplete
  source inventories and existing output destinations. Document rendering and
  visual inspection, a fresh-source environment check, final whole-branch review
  and the four remaining OpenSpec checkboxes are still pending.
- One ablation resource sample observed 422,891,520 combined private working-set
  bytes for launcher/interpreter, with 10.383 GiB available RAM. This is not a
  measured peak. The earlier main sample landed after completion and contains
  no process memory observation; do not treat its null value as zero usage.
- Reproduction finding: Git's LF clean filter changed the byte hash of the
  already-frozen CRLF mapping/protocol/prior-index files. A real `git hash-object`
  probe reproduced the mismatch before the fix. Scoped `.gitattributes` rules
  now preserve those exact bytes and the published evidence directory, including
  CSV line endings. No frozen input or runtime numerical source was rewritten.

## Completed execution and artifact verification

- The earlier queue status above is superseded: all 858 primary and 240 ablation
  runs completed; all 108 groups passed independent numerical replay. No missing
  condition. There are 102 distinct learning artifacts, 336 prefix snapshots,
  2,142,672 observed learning rounds and 1,828,233 served deployment rounds.
- Six figures and both PDF deliverables were generated from the complete
  aggregate. Independent stdlib recomputation from per-seed rows matched all
  912 plotted points and all ablation ranges. PDF readback found nine report
  pages, one poster page, and no text outside page bounds. Every page was
  rendered with Poppler and visually inspected: legible, no overlap or clipping.
- Packaged `tools/research/verify_runs.py` repeated the seed-7 RouterBench PILOT
  group: all 32,270 learning actions and 29,076 served decisions across eleven
  deployment runs match the original report, excluding script hash/preflight.
- The results retain unfavorable outcomes and all twelve recorded-cost overruns.
  At budget fraction 0.5, PILOT is below LinUCB on RouterBench and ExploreOnly on
  MMLU. No confidence interval, significance or paper-equivalence claim is made.
- Final clean-environment reproduction and whole-branch review remain pending.

## Final review and fix pass

- One fresh-context reviewer read the entire working tree and new files. No
  numerical learning/tuning/deployment defect or Critical issue was found.
- Important: evidence scripts relied on `assert` but did not reject optimized
  Python. A real subprocess under `-O` accepted an empty invalid evidence package
  and published `verified=true` with zero source rows. Fifteen regression tests
  failed before the fix, then all passed after explicit early guards in seven
  packaged tools. Production numerical sources and completed research artifacts
  were not changed. A final full-suite run on the fixed package follows.
- Final: minor (deferred): the results README's paired-delta link says all deltas
  are relative to PILOT; learning-size rows actually compare each method with its
  own full-learning control. The CSV and calculations are correct. This link-label
  polish is retained as a disclosed minor, outside the Important fix pass.
- Final: Ruling: exact paper equivalence and checkpoint identity remain separate
  unresolved scientific targets — preserve exclusions and explicit limitations;
  claiming equivalence would overstate these local results.
- Final: Ruling: a second GPU/full-grid run was outside the read-only review —
  accept the existing complete traces, 108 independent replays and clean CPU
  source verification as the evidence for this local release; cross-machine GPU
  reproducibility is not established by this decision.
- Final: Ruling: the reviewer did not independently inspect PDF appearance — the
  primary agent rendered and visually inspected every page, with independent
  numeric and PDF bounds checks; visual review has one human-model viewpoint.
- Final: Ruling: the final reviewer did not re-investigate upstream authenticity —
  retained pinned source evidence and earlier primary-source investigation;
  remaining source/serving equivalence cannot be inferred from hashes alone.

## Final verified release state

- Final: fixed optimized-verification bypass — all 15
  `test_research_tool_guards.py` cases RED→GREEN; clean-source suite **441 passed,
  one skipped** (14.12 s). No production numerical source changed.
- Two new Python environments installed 27 packages from `uv.lock` offline.
  Historical HEAD baseline: 310 passed. Current source was copied byte-for-byte,
  not described as a Git clone. The final check matches 68 runtime/test/config/
  tool/dependency source files to the post-review snapshot. Seven data files and
  47 protocol/prior file pins pass checksum checks.
- Both final synthetic smokes pass and match earlier numerical artifacts.
  The guarded real verifier still reproduces every action/state/metric of the
  32,270-round RouterBench PILOT group and all eleven deployment conditions.
- Git clean-filter probes match raw bytes for every packaged Python tool and all
  three frozen CRLF config files. Published evidence uses scoped `-text` rules.
- Full review, test output, artifact checks, source identity and limitations are
  in `reports/full-experiments-2026-10-29/`; handoff is
  `docs/final-research-handoff-2026-10-29.md`. Preserve the uncommitted workspace
  and historical evidence. There is no new request to commit, push or archive.
- Final whitespace check uses temporary `git -c core.whitespace=blank-at-eol,blank-at-eof,space-before-tab,cr-at-eol diff --check`:
  ordinary `diff --check` treats the intentionally preserved frozen CRLF mapping
  as trailing whitespace. The CR-aware check passes without modifying frozen
  bytes or changing repository/global Git settings.
