# Common deployment integration — 2026-10-29

> **TÀI LIỆU LỊCH SỬ — BGE / mốc 29–30/10/2026.** Các số liệu, trạng thái “hiện tại/current”, nhánh Git và việc còn chờ bên dưới chỉ đúng tại mốc ghi nhận. Không dùng làm kết quả OpenAI hiện hành. Đọc [trạng thái OpenAI](openai-followup-status.md), [bản giao việc hiện hành cho Claude](CLAUDE_UPDATE_OPENAI_2026-10-31.md) và [khôi phục từ checkout sạch](openai-reproduce.md). Giữ nội dung cũ để truy vết, không ghi đè bằng kết quả mới.

The common adapter and first real end-to-end integration are verified. They
complete tasks 4.1 and 4.2 of the offline research plan. The overall project remains
unfinished: tuning, the full frozen experiment inventory, MMLU and multiple seeds,
controlled ablations, figures/report/poster and final reproduction review remain.
Full eleven-model paper equivalence still lacks historical source/mapping evidence.

Evidence: [aggregate report](../reports/common-deployment-2026-10-29.json).
Declared behavior: [common protocol](common-deployment-protocol.md).

## What changed

`FrozenDeploymentRouter` copies and validates the learned state of PILOT, LinUCB,
ExploreOnly, EpochGreedy, Random and Fixed policies. Its checksummed snapshots can
be saved/reloaded without learning or consuming the original RNG. Adapted runners
validate all future predictions before making a budget reservation.

`BinnedCostPolicy` keeps its cosine default, adds an explicit finite linear-score
domain, and accepts validated allowed-arm subsets and tie permutations. Thus
linear models retain their raw ranking, Random samples uniformly among eligible
tied utilities, and fixed-model policies never switch arms. Bounds and score
calibration limitations are explicit. All methods use identical bin/threshold/
fallback accounting. Reports expose coverage, served-only means, total reward
per requested query and historical actual overspend.

## Verification

The 28 new tests failed before the interfaces existed. The targeted suite then
passed 139 tests; the complete suite passed 390 tests with one Windows symlink
skip in 6.88 seconds. Gates ran with one worker and more than 8 GiB free RAM.

The real integration uses the verified eight-arm preference prior, complete pinned
BGE-M3 cache and historical cost export. Each of 13 methods learns the same 128
queries with selected-arm feedback, then serves the same 64 deployment queries
under smoke budgets 0, 0.02 and 1 (benchmark accounting units), with bin size 16
and ratio bounds 1/100000. These defaults were declared before the run and are not
the final tuned evaluation configuration. No paid API request was involved.

The saved integration took 4.51 seconds after import/preflight. Independent code
reads the files, recomputes projected contexts, replays all 1,664 learning
actions and updates, and reconstructs every budget branch and all 1,360 served
deployment decisions. It verifies all 39 run inventories, learned matrices/RNG,
costs, rewards and coverage. Every method serves zero at budget zero and all 64
at budget one. A repeated CLI attempt was first stopped at 7.97 GiB by the RAM
guard; after RAM recovered, it correctly refused the existing output directory.
All original result file checksums remained unchanged.

Local probe commands (helpers and raw artifacts remain ignored):

```powershell
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
$env:OPENBLAS_NUM_THREADS='1'
.venv/Scripts/python.exe outputs/common-deployment-20261029/run_real_smoke.py
.venv/Scripts/python.exe outputs/common-deployment-20261029/verify_real_smoke.py
.venv/Scripts/python.exe -m pytest -q --basetemp outputs/pytest-common-deploy-full-20261029-a --junitxml outputs/common-deploy-regression-20261029.xml
```

The recorded output paths already exist. Do not reuse the pytest basetemp or
overwrite completed runs. Helper hashes, source hashes, model/data identities,
commands and preflights are recorded in the aggregate report. A general full
experiment runner is still part of the remaining work.

## Self-QA handoff

Trạng thái: XONG adapter và tích hợp thật có giới hạn; CHƯA XONG toàn bộ nghiên cứu.

Gate: 390 passed, 1 skipped; baseline trước thay đổi 362 passed, 1 skipped. Test mới đã có lượt đỏ trước triển khai.

E2E: 13 phương án × 3 ngân sách, cùng 128 câu học và 64 câu đánh giá thật; nhánh `codex/data-gate`, chưa commit. Báo cáo có `deployment_adapter`, `prediction_diagnostics` và `score_domain` của mã mới.

Probe: 11 nhóm probe đạt, gồm kiểm toán độc lập toàn bộ lượt học/triển khai và CLI từ chối ghi đè; lịch sử lần thử bị chặn RAM được giữ trong report. Không có tuyên bố kiểm thử race/tải.

Đường hỏng: Sai domain/catalog/tie-order, trạng thái không hợp lệ hoặc checksum sai, dự đoán tương lai tràn số, ngân sách bằng 0 và ghi đè output đều bị chặn đúng; không đặt trước chi phí khi preflight lỗi. Thay nhãn/chi phí đánh giá không đổi quyết định; chi phí vượt ngân sách vẫn được báo riêng.

Người thứ hai: n/a — CLI local chạy tuần tự.

Lưu trữ: Snapshot lưu/đọc lại; verifier dùng NumPy và phép tính độc lập để dựng lại trạng thái học và chọn model từ JSON/CSV, đối chiếu dữ liệu gốc cùng checksum.

Chưa chạy: Tuning-only search, toàn bộ learning/deployment, nhiều seed/MMLU, ablation, biểu đồ/báo cáo/poster và final fresh-clone/whole-branch review.

Bug đã biết: Manifest prepare/fit cũ còn metadata trạng thái lịch sử về đầu vào đã hoàn thành; không sửa artifact bất biến. Các báo cáo mới xác định rõ đầu vào đã kiểm chứng. Cần làm sạch cách diễn đạt trạng thái ở runner tổng hợp sắp triển khai.

Quyết định: LinUCB giữ điểm tuyến tính không clipping; Random dùng utility 1/K và hòa điểm ngẫu nhiên theo query/seed; Fixed chỉ cho phép model đã định. Đây là quy ước local có tài liệu, không tuyên bố tác giả paper đã dùng y hệt.

Tôi sẽ lo: Tuning-only search và chốt cấu hình đầy đủ trước khi chạy các thí nghiệm chính, sau đó kiểm chứng và tổng hợp các hạng mục còn lại.

Cần bạn: Chưa cần thêm đầu vào cho các bước offline; không có khoản API tính phí nào được sử dụng.
