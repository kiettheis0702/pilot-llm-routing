# Course evidence and reproducibility

Run from repository root, one command at a time. Use a new output directory for every immutable stage. No paid API is needed.

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

## W&B — prepared, not published

The logged-in W&B account is `kiettheis`, with team `kiettheis-university-of-science-vnu-hcm`, confirmed in Chrome on 30 October 2026. The home page shows an Academic trial with 29 days left and an unfinished Academic application; no project is listed. The intended project name is `pilot-llm-routing-team8`. Check the current plan limits before upload; do not enable paid features. Browser login does not establish SDK authentication. Authenticate locally with `wandb login`; never paste a key into chat or Git. Inspect `reports/course-submission-2026-10-30/tracking-payloads.json` first. Public visibility and upload require the user's final approval. The immutable report package retains the earlier pre-login snapshot.

The real SDK offline smoke has succeeded for one actual aggregate. Default execution makes no online W&B run:

```powershell
.venv/Scripts/python.exe -m tools.course.tracking --record --limit 1
```

Only after approval, substitute the real entity and use `--mode online --entity NAME --approve-public-upload`. Receipts are stored immediately with a content-derived run ID. A finished current-version receipt is skipped on continuation. After upload, verify public run access while signed out, then record the actual URLs in `reports/course-submission-2026-10-30/public-run-links.json`. A local ID or SDK URL is not proof of public access.

Publication evidence has `version: 1` and a `runs` array. Each row requires exactly
`id`, `url`, `public_verified: true`, `verified_at` (actual ISO timestamp with
timezone), and `verification_method: "signed-out-browser"`. Use IDs from the
prepared payload and URLs of the form `https://wandb.ai/ENTITY/PROJECT/runs/ID`.
Do not create this evidence before checking access. The reader validates supplied
records; it does not perform the signed-out check. It rejects unknown IDs, wrong
destinations, duplicates and missing verification. Never manually edit the derived
CSV: `prepare_presentation` joins this source by ID and preserves verified links
on every rebuild. Partial publication retains known URLs and pending rows; only
complete verified coverage changes the deck/poster to public-project references.

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
