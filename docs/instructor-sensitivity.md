# Instructor-XL sensitivity — verified full core

This supplementary experiment preserves the completed OpenAI/BGE results and
all presentation files. It uses free public weights on the local RTX5070;
no new embedding/generation API request is made.

## Representation and inputs

- Model `hkunlp/instructor-xl`, pinned revision
  `ce48b213095e647a6c3536364b9fa00daf57f436`.
- FP32 T5 encoder1024D -> instruction-masked mean -> unbiased identity Dense768D
  -> L2 normalization. Maximum512 tokens includes the instruction/EOS.
- Local tokenizer is the slow T5 tokenizer (`use_fast=False`) in the recorded
  Transformers environment. The official Instructor loader forwards optional
  tokenizer_args to AutoTokenizer without fixing use_fast. The authors' choice
  is unconfirmed. Official pooling/mask replay below does not establish equality
  to every possible fast-tokenizer/library version or an author-exact recipe.
- Declared local instruction:
  `Represent the user question for routing to a suitable language model:`
  This exact instruction is not verified against the PILOT authors' settings.
- The official Instructor input formatter strips instruction/question and concatenates without
  an extra separator. Pooling excludes instruction tokens. Current computation
  uses dynamic right padding; the independent verifier compares this with the
  pinned official fixed512-padding implementation.
- Original36497 routing and1558 preference rows are retained. Exact UTF8
  deduplication gives38018 unique texts. Compute order is length-sorted for
  batching; role exports restore the original query IDs/order. Router order is
  never replaced by embedding computation order.

The full token audit found214 routing rows and52 preference rows exceeding512
tokens. Maximum full lengths are1127 and3298 respectively. These are capped,
not dropped. The OpenAI recipe used full text; BGE used another tokenizer and
2048-token cap. Any comparison is a representation-recipe/retuning comparison,
not isolated encoder causality.

## Durable local cache

`tools/openai_study_instructor_journal.py` owns an actual OS file lock and
publishes immutable contiguous chunk records with vector checksums after
flush/fsync. It validates all existing commits on resume, overwrites only an
uncommitted tail, rejects input/recipe changes and publishes a cache manifest
only when every declared vector is committed. Existing main caches are untouched.

Nine journal tests and three index/export tests were observed RED then GREEN.
The real64-vector smoke passed independent checksum/unit-norm readback and had
no completed manifest, proving partial work stays partial. Full encoding then
resumed from64. The initial run later stalled at37568; the user explicitly
approved stopping only that job and resuming450 vectors. A separate recovery
overlay records batch1/fixed512 padding, the unchanged prefix hash and new
producer hash. Parent protocol/code/commits are preserved. All450 tail vectors
completed and passed independent NumPy pooling (maxdifference2.82e-08), using
about5.15GB allocated GPU memory. The full cache contains38018 unique vectors,
restores38055 source rows, and preserves the original37568-vector prefix hash.
Independent verification checked every vector/chunk, role mapping and token
length, and replayed37 stratified/boundary samples using the pinned official
pooling/instruction-mask implementation (maximum absolute difference8.20e-08).
Cache completion and the downstream858-condition routing study are verified.

The real four-longest-input GPU probe used about5.60GB allocated GPU memory;
independent NumPy pooling differed by at most1.93e-08. It is a compatibility
probe, not a completed experiment or benchmark performance result.

## Evaluation scope and command order

Keep one heavy command/worker at a time and all BLAS/Torch threads at1. Require
at least8GiB available physical RAM before starting each heavy stage. Commands
below assume the pinned source-data/input bundle and public local model assets
already exist. A clean-clone asset reconstruction helper still belongs to the
final reproducibility/publication work.

```powershell
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
$env:OPENBLAS_NUM_THREADS='1'
.venv-embedding/Scripts/python.exe -m tools.openai_study_instructor_encode prepare
.venv-embedding/Scripts/python.exe -m tools.openai_study_instructor_encode encode --limit-chunks 1
.venv-embedding/Scripts/python.exe -m tools.openai_study_instructor_encode encode
# For the documented approved stalled-run recovery only:
.venv-embedding/Scripts/python.exe -m tools.openai_study_instructor_tail
.venv-embedding/Scripts/python.exe -m tools.openai_study_instructor_verify
.venv/Scripts/python.exe -m tools.openai_study_instructor_study declare
.venv/Scripts/python.exe -m tools.openai_study_instructor_study fit
.venv/Scripts/python.exe -m tools.openai_study_instructor_study tune
.venv/Scripts/python.exe -m tools.openai_study_instructor_study freeze
.venv/Scripts/python.exe -m tools.openai_study_instructor_study run
```

The declared core is858 conditions/78 streams/312 nested snapshots,8 admitted
arms, both RouterBench/MMLU, three seeds and the original64D30/40-epoch prior
configuration. Priors and tuning are refitted on Instructor vectors; calibration
still uses the original313 pairs. No500-pair final-test result is implied.
Generic independent `tools/research/verify_runs.py` replayed all78 groups
with the Instructor protocol/index/cache paths. The report producer checked the
full inventory/current proofs and published11 immutable evidence files in
`reports/instructor-core-2026-10-31/`. Package manifest SHA256:
`985cd6e045def515cd91dda19dae0f02977c6fd523a15875d391d7ff19d9942f`.
Both PNG figures were viewed and are readable.

At50%estimated allocation, Instructor PILOT quality/GPT4 is89.1763% on
RouterBench and84.0791% on MMLU, with100%coverage. Changes versus OpenAI are
+2.5810 and+0.5512percentage points. At100%allocation changes are-0.5978 and
-0.2270points. Do not conclude that one recipe always wins. See the immutable
per-seed/paired-recipes CSVs for all conditions and recorded-cost denominators.

Padding clarification for the report's general dynamic-padding description:
the first37568 vectors used dynamic right padding; the approved450-vector
recovery used fixed512 right padding and batch1. The parent protocol, prefix
hash and recovery overlay remain separate and checked. The cache verifier's
official fixed-padding/pooling/mask replay covers both sides of that boundary.

The separate cache verifier checks every vector checksum, role alignment and
token length, then replays35 stratified real inputs plus two recovery-boundary
samples when an overlay is present, using the pinned official pooling class and
instruction-mask method. This checks sampled neural outputs, not a second
implementation of the entire T5 network. It must finish before fitting starts.

## Current verification caveat

An existing regression test initially failed creating a long Windows temp
filename. The parent directory existed; a223-character test path succeeded, a
normal280-character path failed, and the same extended path succeeded. No core
source was changed. A targeted rerun and full suite with short fresh basetemps
passed649 tests,1 skipped. After the study/recovery additions, the full suite
passed654 tests,1 skipped. Keep experiment-test basetemps short, e.g. a fresh
`outputs/q-ixl-03`, instead of nesting them in the research output directory.
The failed invocation and diagnostic JSON are retained in the followup outputs.

Remaining scientific gates are unchanged: three extra model identities,
author release/task aggregation/split/order and unpublished pretraining details.
Instructor does not close these gates or establish exact paper reproduction.
