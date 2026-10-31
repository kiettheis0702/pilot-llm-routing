# Course evidence and reproducibility

**Current OpenAI restore entrypoint:** [openai-reproduce.md](openai-reproduce.md).
It uses a separately checksummed cache/input archive and tests a clean checkout.
The BGE commands and earlier reconstruction limits below are historical.

**Follow-up prerequisites and complete command order:**
[followup-reproduce.md](followup-reproduce.md). It supplies the missing
preference split preparation, calibration-selected replicate entrypoint,
independent cost replay and the pinned historical Hybrid batch decision record.

Run from repository root, one command at a time. Use a new output directory for every immutable stage. No paid API is needed.

**31 October followup:** commands below reconstruct the historical BGE course
package. Current OpenAI core/ablations, budget, preference500, MF and noise
results have separate immutable packages linked from the repository README.
They reuse previously approved/completed OpenAI caches; they do not regenerate
paid embeddings. New publication is verified in the [2498-view registry](../reports/openai-publication-2026-10-31/README.md);
the1,128-run mapping in the dated section below is historical. See the [followup status](openai-followup-status.md)
and each report README for the current exact command sequence.

Pinned free follow-up assets can now be reconstructed with the tracked helper:

```powershell
.venv/Scripts/python.exe -m tools.openai_followup_assets --sources
.venv/Scripts/python.exe -m tools.openai_followup_assets --model deberta
.venv/Scripts/python.exe -m tools.openai_followup_assets --model instructor
```

The neural stages used a separate Python3.11.18 environment. Preserve an existing
verified environment. On a fresh Windows/CUDA128 checkout only:

```powershell
if (Test-Path -LiteralPath .venv-embedding) { throw 'Use the existing verified environment; do not replace it.' }
$env:UV_CONCURRENT_DOWNLOADS='1'
$env:UV_CONCURRENT_INSTALLS='1'
$env:UV_CONCURRENT_BUILDS='1'
uv venv .venv-embedding --python 3.11.18
uv pip install --python .venv-embedding/Scripts/python.exe torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv-embedding/Scripts/python.exe -e '.[cost]' -r configs/followup-neural-requirements.txt
```

The requirements record observed versions; no environment upgrade is needed for
the completed run. This is a reconstruction recipe, not a claim that a second
fresh environment installation has been tested. Check local CUDA compatibility
with the real offline probe before starting a long neural job.

These commands perform public HTTPS downloads only, one file at a time, with
expected SHA256 checks. Instructor weights require about5GB download/storage;
DeBERTa weights about0.87GB. Existing mismatched or interrupted assets are
retained for inspection. The tracked source manifest is a historical audit
record; dynamic GitHub/HF API responses are not claimed to be reproducible byte
for byte. This reconstructs external source/model assets, not ignored paid
OpenAI vectors, raw study artifacts, or the entire frozen experiment lineage.
After downloading Instructor assets, run the real offline compatibility probe
with `.venv-embedding/Scripts/python.exe -m tools.openai_study_instructor_probe`
before encoder preparation. This creates the checked input required by the
encoder; a downloaded checkpoint alone is insufficient.
See [Instructor recipe/status](instructor-sensitivity.md) and the prospective
[Hybrid/drift protocol](hybrid-drift-protocol.md). Follow workstation sequencing
and RAM guards before starting the neural stages.

The completed drift recipe and the full-RouterBench binary recipe are separate.
For a fresh lineage with the required existing parent inputs, run the following
stages serially, stopping on any failed command. They refuse completed output
paths; these are not commands to rerun over the saved evidence:

```powershell
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
$env:OPENBLAS_NUM_THREADS='1'
$env:TOKENIZERS_PARALLELISM='false'
.venv-embedding/Scripts/python.exe -m tools.openai_study_hybrid_default_batch probe
.venv-embedding/Scripts/python.exe -m tools.openai_study_hybrid_default_batch audit
.venv/Scripts/python.exe -m tools.openai_study_hybrid_default_batch declare
.venv-embedding/Scripts/python.exe -m tools.openai_study_hybrid_default_batch train
.venv-embedding/Scripts/python.exe -m tools.openai_study_hybrid_default_batch verify
.venv/Scripts/python.exe -m tools.openai_study_hybrid_report binary
```

The default-batch adapter requires its pinned hardware-only feasibility record,
immutable parent producers, source downloads, model verification and parent core
protocol/partitions. Run `python -m tools.openai_study_reproduce prepare` first to
install the original checksummed hardware decision record. It is historical
evidence, not a new-machine throughput measurement; run the real compatibility
probe and input audit in the new checkout. Do not bypass these guards or copy
a protocol while changing inputs. The restored archive supplies paid caches;
raw learning lineage must be regenerated in the documented order. Drift uses
the original2/64 producer; binary uses8/16
with eval8. See [declared recipes and limitations](hybrid-drift-protocol.md).

Offline follow-up publication preparation requires all11 verified report packages:

```powershell
.venv/Scripts/python.exe -m tools.openai_followup_tracking
.venv/Scripts/python.exe -m tools.openai_followup_publish
.venv/Scripts/python.exe -m tools.openai_followup_artifact
```

These defaults perform no external writes. Online actions require the separate
explicit upload flag and current authorization for the existing project. The
publisher checks included storage with automatic upgrade disabled, protects the
old1128 runs, persists accepted identities before metric validation and uses new
recipe/family IDs. After a network interruption, rerun only after inspecting
receipts and fresh remote state; do not recreate unknown accepted requests.

```powershell
uv sync --locked --extra cost --extra tracking
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
$env:OPENBLAS_NUM_THREADS='1'
.venv/Scripts/python.exe -c "from scripts.resources import require_resources; print(require_resources())"
.venv/Scripts/python.exe -m tools.course.prepare_inputs --output-dir outputs/my-course-inputs
.venv/Scripts/python.exe -m tools.course.analyze --output-dir outputs/my-course-analysis
.venv/Scripts/python.exe -m tools.course.tracking --analysis outputs/my-course-analysis --output outputs/my-tracking-payloads.json
```

Preparation downloads pinned RouterBench 0-shot and Arena 55K from Hugging Face, or checks cached bytes, reconstructs normalized tables/splits, validates exact hashes, and writes all routing/preference embedding inputs. The completed local run reconstructed 36,497 queries, 401,467 outcomes and 1,558 eligible pairs. It reused verified downloaded files; this is not a test of a new network download.

The analysis command verifies all 184 files in the frozen research package before using any aggregate; it then executes the declared synthetic and counterfactual supplement. Tracking exports 1,128 sanitized conditions with deterministic IDs and original execution dates. These are historical imports, not 1,128 new live LLM runs.

## W&B — uploaded, verified and public

On 30 October 2026, the user authorized uploading the existing experiment
results. [The project](https://forge.coreweave.com/wandb/kiettheis-university-of-science-vnu-hcm/pilot-llm-routing-team8)
contains 1,128 finished runs: 1,098 historical deployment conditions, 18 synthetic
magnitude cases and 12 cost-accounting cases. All 13,314 metric values were
checked after upload. The [verified artifact](https://forge.coreweave.com/wandb/kiettheis-university-of-science-vnu-hcm/pilot-llm-routing-team8/artifacts/evaluation-results/pilot-team8-verified-results/v0)
contains 19 files, including eight figures and the research report/poster.
Visibility is Public. Anonymous readback verified all 1,128 configs and 13,314
metrics without credentials, and a signed-out browser displayed the project.
Use the [current mapping and evidence](../reports/publication-2026-10-30/README.md).
See [verification and limits](wandb-upload-2026-10-30.md).

### Local SDK login on Windows

Run from the repository root. In **Command Prompt (CMD)**:

```cmd
.venv\Scripts\wandb.exe login --cloud --verify
```

In **PowerShell**:

```powershell
& .\.venv\Scripts\wandb.exe login --cloud --verify
```

Paste the API key only into the CLI's interactive prompt, never into chat,
scripts, or Git. A command containing spaces needs double quotes in CMD;
PowerShell's leading `&` is not valid CMD syntax. If the launcher reports
`No Python at ...`, follow the [Windows runtime repair notes](windows-python-runtime.md).
On this workstation the Python 3.11.18 repair was verified from outside Codex;
the user subsequently completed SDK login successfully.

The authenticated account is `kiettheis`, with team
`kiettheis-university-of-science-vnu-hcm`. No payment method or pay-as-you-go
was enabled for this upload; the reviewed storage allowance covered the small
evidence package. Recheck limits before any future upload that could exceed
the allowance. The immutable report package retains the earlier pre-login
snapshot, so its historical upload status is not the current project status.

The real SDK offline smoke has succeeded for one actual aggregate. Default execution makes no online W&B run:

```powershell
.venv/Scripts/python.exe -m tools.course.tracking --record --limit 1
```

The approved bulk import uses the pinned W&B SDK's authenticated transport,
serial batches of at most 20 records, and W&B's file stream completion protocol.
It records one aggregate history row per condition; it does not reconstruct
training trajectories. Source checksums and deterministic IDs prevent silent
replacement or duplicate imports. Finished matching runs are verified and skipped.

```powershell
.venv/Scripts/python.exe -m tools.course.import_summaries --entity kiettheis-university-of-science-vnu-hcm --approve-upload --verification outputs/course-20261030/wandb-publication/upload-verification.json
```

Receipts retain destination, payload hash and server identity. The final readback
checks run state, every expected config field and every metric. Float configs
allow one representable floating-point step for JSON serialization rounding;
other scalar types/values remain exact. The regular SDK writer uses a shorter
local working directory to avoid Windows path-length errors while keeping
receipts organized by entity/project/run.

Public sharing was separately authorized and enabled on 30 October 2026.
`reports/publication-2026-10-30/public-verification.json` records the actual
anonymous GraphQL check; `experiment-map.csv` joins those IDs to the frozen
experiment map. A local ID or SDK URL alone is not proof of public access.
The frozen course package is intentionally unchanged. Its old map has empty
URLs and its publication flag is historical; use the new publication folder.

The original `tools.course.publication` reader accepts only `wandb.ai` URLs
and a per-run signed-out-browser proof schema. It does not understand this
new Forge-domain anonymous-API evidence. Do not pass the new manifest into
that legacy reader or relabel API checks as browser checks. The current
presentation task is handed to Claude using [CLAUDE_UPDATE_OPENAI_2026-10-31.md](CLAUDE_UPDATE_OPENAI_2026-10-31.md),
the verified numeric sources and the new public mapping. Existing PPTX files
are references and retain their earlier publication footer until revised.

## Rebuilding embeddings, fitting and tuning

The new preparation entrypoint removes manual raw-data preparation. The remaining GPU stages use the existing pinned local embedding environment and 2,048-token recipe. Run preference encoding and routing encoding separately with `scripts.local_embeddings --inputs ... --output-dir ... --recipe configs/local_embedding_full.json --batch-size 2 --checkpoint-batches 32 --download`. Do not use the older 512-token default.

See [full embedding/prior handoff](full-embedding-prior-handoff-2026-10-29.md), [cost export](cost-inputs.md), [tuning handoff](tuning-handoff-2026-10-29.md), and [frozen full runner](full-experiments.md) for recorded commands and pins. Production prior settings are 64D (8D only for its ablation), 30 projection epochs, 40 prior epochs, seeds 7/42/2026. Run each serially.

**Limit:** the historical full runner validates dated cache, prior and tuning paths/hashes. A newly generated pipeline must create and independently verify a new protocol/index before running its grid; changing paths or bypassing hashes is not an end-to-end reproduction. This turn verified hosted reconstruction and analysis, not a fresh complete GPU rebuild. The original complete evidence and 108 group replays remain valid and unchanged. A clone alone does not contain the ignored large caches/checkpoints.

## Course artifacts

`tools/course/prepare_presentation.py` derives editable chart data and experiment mappings from verified results. `tools/course/build_slides.mjs` authors the deck and poster with the bundled Artifact Tool runtime. Native text, diagrams, tables and charts remain editable. The verified build used Artifact Tool 2.8.106 and the Codex presentations skill 26.1026.10807; Python requirements alone do not install that runtime. The two final files are in separate folders because both must be named `8.pptx`.

For a rebuild, set `COURSE_BUILD_DIR` and `COURSE_DELIVERABLE_DIR` to new folders
inside this repository. The finalizer refuses existing output/receipt files.
Set `CODEX_PRESENTATION_RUNTIME` and `CODEX_PRESENTATION_SKILL` if the bundled
runtime is installed elsewhere. The runtime directory must contain
`node/node_modules/@oai/artifact-tool/dist/artifact_tool.mjs` and
`python/python.exe`; the skill directory must contain `container_tools/` with
`artifact_tool_utils.mjs` and the package/layout validators. Run the builder with its bundled Node executable,
then `python -m tools.course.verify_artifacts --folder NEW_OUTPUT --output NEW_REPORT.json`
and `node tools/course/render_slides.mjs NEW_OUTPUT`. Inspect the final renders
before replacing a reviewed deliverable. Chart snapshots intentionally round to
eight decimals; `slide-content.json` retains the unrounded evidence.

## Final follow-up validation and recovery

A clean export of the reviewed Git index verifies all14 scientific report
packages and rebuilds byte-identical2498 tracking payloads without ignored paid
caches. This validates derived-evidence reconstruction; it does not claim a
second clean neural environment or a new end-to-end GPU fit. The artifact
metadata and every expected run value were checked anonymously after upload.

For Windows QA in a new checkout, create the parent before a short, fresh
pytest basetemp (the parent is not created by pytest):

```powershell
New-Item -ItemType Directory -Path outputs -Force | Out-Null
.venv/Scripts/python.exe -m pytest -q -rs --basetemp=outputs/qa-NEW
```

Use a new temp name for each invocation. Current681pass/1skip and baseline
490pass/1skip share the OS symlink-privilege limit; no admin/registry change was
made. A first baseline invocation with a missing parent is retained in the
operational logs, separate from the successful baseline.

The artifact wrapper follows W&B's [existing-run artifact recipe](https://docs.wandb.ai/support/models/articles/how-do-i-log-an-artifact-to-an-existing-)
with resume='allow', after proving that the exact aggregate ID/config/metrics
already exists. GraphQL imports can lack SDK initialization metadata, so must
can fail despite the public API run existing. The wrapper carries the same
canonical values/labels and verifies the source run is finished afterwards.
Schema2 receipts distinguish no submission from uncertain/accepted submission;
a failed/unknown artifact lookup never authorizes another artifact generation.
The new70-file artifact and all2498 summary views have actual readback proofs
in the publication folder. A real post-acceptance lost-reply probe resumed the
same server ID; an idempotent repeat added no run. No model API charges.
