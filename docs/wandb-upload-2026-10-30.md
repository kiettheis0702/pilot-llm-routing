# W&B upload verification — 30 October 2026

> Publication follow-up: the user subsequently authorized public sharing.
> The project is now **Public**, with all 1,128 runs and 13,314 metrics verified
> anonymously. See [public evidence and current links](../reports/publication-2026-10-30/README.md).
> This document preserves the earlier Team-only upload receipt and its QA history.

The user's request to upload the existing experiment results has been completed.
[Project](https://forge.coreweave.com/wandb/kiettheis-university-of-science-vnu-hcm/pilot-llm-routing-team8)
visibility is **Team**, confirmed in the browser. No visibility change, paid
model call, paid feature activation or Moodle submission was performed.

| Evidence | Verified result |
| --- | --- |
| Expected / remote / finished runs | 1,128 / 1,128 / 1,128 |
| Historical deployment conditions | 1,098 |
| Synthetic magnitude cases | 18 |
| Cost-accounting cases | 12 |
| Metric values compared with local payload | 13,314 |
| Artifact files / bytes | 19 / 7,167,970 |
| Regression | 490 passed, 1 Windows symlink skip |

The [results artifact, version v0](https://forge.coreweave.com/wandb/kiettheis-university-of-science-vnu-hcm/pilot-llm-routing-team8/artifacts/evaluation-results/pilot-team8-verified-results/v0)
contains six research figures, two supplementary figures, the nine-page research
report, research poster, four aggregate CSVs, figure data, the complete sanitized
tracking payload, explanatory READMEs and a checksum manifest. Every remote file
name, size and content digest matched its local source. It is attached to an
existing run, so it did not create an extra experiment.

## Source and interpretation

The importer verified the 184-file frozen research package at
`reports/full-experiments-2026-10-29` and the supplementary analysis package at
`reports/course-submission-2026-10-30/analysis`. Rebuilt payloads exactly matched
`reports/course-submission-2026-10-30/tracking-payloads.json` before sending.
Original execution dates and source IDs are recorded in run configs.

These are historical results from the declared local eight-arm variant, with
synthetic/counterfactual supplements explicitly identified. Uploading them did
not execute 1,128 new model runs. The bulk importer records one aggregate history
row per condition; full training trajectories were not reconstructed. Recorded
benchmark costs are not charges incurred by this upload. No raw prompts, keys,
source code, machine telemetry, embeddings or model checkpoints were uploaded.

## Verification and repairs

After upload, a fresh W&B API session read every run's finished state, expected
config and summary metrics. There were 108 budget fields serialized as
`7.67374146` instead of `7.6737414600000005`: a difference of one floating-point
step (about 8.88e-16). The config comparison accepts at most one such step;
all other scalar types/values remain exact. A regression reproduced the failure,
then passed after the fix; tests also reject larger differences, NaN and wrong
types. Metric comparisons use relative/absolute tolerance 1e-12.

The first SDK upload exposed a Windows path-length failure. A shorter SDK work
directory fixed the actual offline and online paths without changing the
destination-qualified receipt location. Five runs used the full SDK writer;
the remaining historical summaries used serial GraphQL batches and W&B file
stream completion. Re-running verification found 1,128 existing matching runs
and zero pending imports. The artifact upload then resumed one existing run,
finished it again, and independently rechecked its metrics and artifact digests.

Validation ran serially after the resource guard reported no heavy jobs and
over 10 GiB free RAM. The full suite passed: **490 passed, 1 skipped**. The skip
is the existing Windows symlink test. No paper-equivalence or fresh GPU
reproduction claim is made by these upload checks.

## Local receipts

Operational evidence is in ignored `outputs/course-20261030/wandb-publication/`:

- `upload-verification.json`: all 1,128 run IDs, URLs and source hashes.
- `artifact-upload-receipt.json`: artifact ID/version, manifest hash and remote verification.
- `config-readback-audit.json`: the 108 one-step JSON roundings.
- `summary-regression-final.txt`: complete regression outcome.
- `uploaded-project.jpg`: browser evidence of 1,128 runs and Team visibility.

Per-run receipts are under
`outputs/course-20261030/wandb-receipts/online/kiettheis-university-of-science-vnu-hcm/pilot-llm-routing-team8/`.
Frozen source report packages remain unchanged. Their pre-login publication
status is historical; README and the publication follow-up record the current state.

## QA handoff

```text
Trạng thái: XONG — đã chạy thật trên W&B.
Gate: pytest -q --basetemp <new project cache path> → 490 passed, 1 skipped;
      trước thay đổi importer: 472 passed, 1 skipped. git diff --check → đạt.
E2E: import đủ 1.128 bản ghi, đọc lại 13.314 chỉ số; artifact v0 có 19 tệp.
     Build hiện tại ghi config_float_tolerance trong upload-verification.json.
Probe: 6 nhóm chạy, 6 đạt: nguồn/hash; import/readback; chạy lại không trùng;
       artifact/digest; đường dẫn Windows; dữ liệu/lỗi kết nối giả lập.
Đường hỏng: thiếu duyệt/destination sai/dữ liệu ngoài allowlist → chặn trước gửi;
           kết nối lỗi giả lập → giữ receipt planned; phản hồi sai → không finished;
           ngân sách làm tròn 1 ULP → chấp nhận, sai lớn hơn → chặn.
Người thứ hai: n/a — luồng upload một người.
Lưu trữ: API session mới đọc lại toàn bộ; reload trình duyệt thấy 1.128 run,
         Team visibility và artifact v0, 19 tệp, 7.2 MB.
Chưa chạy: ngắt mạng thật giữa upload chưa ép; chỉ mô phỏng transport failure.
          Baseline 472/1 là kết quả trước sửa tại workspace, không tạo worktree mới.
Bug đã biết: không còn bug cản trở upload trong phạm vi đã kiểm tra.
Quyết định: giữ quyền Team theo dự án hiện có; dùng historical aggregates.
Tôi sẽ lo: không còn việc cần làm trong phạm vi upload này.
Cần bạn: không có.
```
