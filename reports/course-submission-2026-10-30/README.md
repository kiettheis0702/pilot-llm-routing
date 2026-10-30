# Course submission package — 30 October 2026

**Reviewable local package, not yet a completed public submission.** The user
does not have a W&B account. No W&B upload, GitHub visibility change or Moodle
submission has occurred. No paid API was called.

| Experiment | Script | Evidence / run mapping | Finding |
|---|---|---|---|
| Budget vs quality | `scripts/experiments.py` | [1,098 source IDs and prepared W&B IDs](experiment-map.csv); public URLs pending | At half local budget PILOT trails LinUCB on RouterBench and ExploreOnly on MMLU |
| Learning prefixes / regret | `scripts/experiments.py` | Same mapping, `learn-0.1/0.25/0.5/1`; [editable chart data](slide-content.json) | PILOT has lower observed learning regret than those two controls on RouterBench; deployment ranking differs |
| Five ablations | `scripts/experiments.py --variant ...` | Lab 2 rows in mapping; original 240 conditions | Unit-strength and prior-only changes have measurable local effects; no universal causal claim |
| Magnitude stress | `tools/course/analyze.py` | [Nine synthetic cases](analysis/analysis.json), 18 policy records | 2,000 rounds, alpha=1: regret 600 vs 1.2; assumption-violating construction |
| Cost sensitivity | `tools/course/analyze.py` | Same JSON, 12 population/factor records | Main full-learning PILOT: 0/48 → 25/48 excesses at 1.00× → 1.10× cost, choices fixed |

The [tracking payload](tracking-payloads.json) contains config, scalar metrics,
provenance and deterministic IDs. The writer adds a native W&B summary plot from
unit-scale metrics. A real SDK offline run with plot and readback receipt passed.
Historical-import tags retain original execution dates. Raw prompts are excluded.

The files [midterm/8.pptx](../../deliverables/midterm/8.pptx) and
[poster/8.pptx](../../deliverables/poster/8.pptx) contain editable text, diagrams,
tables and charts. Midterm: 33 slides, six charts, nine tables, notes on every
slide. Poster: one A2 portrait slide with two charts and adjacent theory/evidence.
No separate references slide; numbered sources appear on relevant slides.

[Paper analysis](../../docs/course-paper-analysis.md),
[reproduction instructions](../../docs/course-reproduce.md),
[Vietnamese defense guide](../../docs/course-defense-guide-vi.md), and
[course audit/deadlines](../../docs/course-requirements-audit-2026-10-30.md).

The original 184-file package is unchanged. The new slide charts are independently
checked against 128 three-seed aggregate points. Chart workbook snapshots use
eight decimal places; JSON retains full precision. Final-PPTX re-import renders
match the authored PNG bytes. This is not a native PowerPoint application check.
The current receipt is [artifact-verification-final.json](artifact-verification-final.json);
the earlier receipt records the pre-review version. Post-review regression:
472 passed, one Windows symlink skip. See [independent review](independent-review.md)
and [verification summary](verification-summary.json), including the transient
Windows rename error seen in one earlier test invocation.

Remaining: create/authenticate W&B locally, approve and verify public aggregate
upload, fill real URLs and rebuild artifacts; approve repository publication and
select the correct default/submission branch; verify on the presentation PC and
submit the correct file to each Moodle assignment. A fresh-machine complete GPU
and grid rebuild is also unverified; the hosted input reconstruction is verified.
