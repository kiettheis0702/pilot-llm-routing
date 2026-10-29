# Tuning progress — verification pending RAM

The prior-64 fit completed successfully. The new tuning implementation has not
passed execution tests yet. The 8 GiB resource gate correctly prevented new test
processes when free physical RAM fell below the threshold. No existing process
was stopped, and no paid API request was made. Keep task 3.2 unchecked.

Evidence: [progress record](../reports/tuning-progress-2026-10-29.json).
Design: [tuning protocol](tuning-protocol.md).
Primary experiment choices: `configs/main_experiment_plan.json`; numerical
calibration and measured resource estimates are not frozen yet. The subsequent
[ablation design](ablation-protocol.md) declares all five variants before full
deployment outcomes; none has run yet.

## Completed work

- The existing fit command produced
  `outputs/research-prior-20261029/identity-d64-s42/prior.json`: dimension 64,
  30 projection epochs, 40 prior epochs, seed 42. It reports valid holdout-derived
  initialization coefficients for all eight arms. Session 39838 is terminal with
  exit 0; do not restart or overwrite this fit.
- `scripts/tuning.py` implements complete prequential candidate searches, explicit
  tuning-only input isolation, immutable trace/snapshot outputs, parameter choice,
  source verification for real CLI runs and tuning-only budget calibration.
- `tests/test_tuning.py` adds 17 tests; all failed before the new module existed.
  Python syntax parsing passes, which does not establish runtime correctness.
- The declared primary design retains frozen phases and stream order, filters
  MMLU within each phase, uses three replicate seeds and covers 858 deployment
  runs. MMLU contains 398 tuning / 12,369 learning / 1,275 deployment rows.

## Resume in order

1. Inspect current heavy jobs and free RAM. Wait below 8 GiB; earlier CPU waivers
   were scoped to previous runs and are not reused here.
2. Run the targeted tests with a new basetemp, fix any failures and run the full
   suite. The last completed full suite before these edits was 390 passed / one
   Windows symlink skip.
3. Independently verify the new prior's file/embedding/preparation lineage,
   training-only triplets, finite projected directions and holdout cosine counts.
4. Run real tuning sequentially for RouterBench and MMLU, then independently
   replay scores/configuration selection and cost calibration. Every unselected
   candidate must remain in the result inventory. Check held-out mutation
   invariance without changing original pinned files.
5. Update reports and task 3.2 only after those checks pass. Complete the remaining
   ablation choices/resource estimates before marking protocol task 2.3 complete.

Targeted command, after preflight:

```powershell
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
$env:OPENBLAS_NUM_THREADS='1'
.venv/Scripts/python.exe -m pytest tests/test_tuning.py -q --basetemp outputs/pytest-tuning-green-20261029-d --tb=short
```

Intended real command (not run yet):

```powershell
.venv/Scripts/python.exe -m scripts.tuning --data-dir data --prepared-dir outputs/local-pipeline-verification-20261029/prepared-identity --input-dir outputs/research-input-audit-20261029/inputs --cache-dir outputs/research-embeddings-20261029/routing-full --fit-dir outputs/research-prior-20261029/identity-d64-s42 --cost-dir outputs/research-cost-inputs-20261029 --grid configs/tuning_grid.json --domain routerbench --output-dir outputs/research-tuning-20261029/routerbench-d64-s42
```

After that command finishes, repeat with `--domain mmlu` and fresh output path
`outputs/research-tuning-20261029/mmlu-d64-s42`. Never run the two concurrently.

## Subsequent preparation and resource check

The next attempt (`pytest-tuning-green-20261029-d`) was refused at **5.65 GiB**
before importing pytest. No new test result is claimed. The prior verifier is now
saved at `outputs/research-prior-20261029/verify_d64.py`. It independently reads the
checkpoint and unchanged cache hashes, checks training-only triplet constraints,
recomputes all holdout arm counts/lambdas and final training objectives, and checks
finite nonzero projected directions across all preference/routing inputs. It has
only passed syntax parsing; execution still requires the resource gate:

```powershell
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
$env:OPENBLAS_NUM_THREADS='1'
.venv/Scripts/python.exe outputs/research-prior-20261029/verify_d64.py
```

The verifier refuses an existing `outputs/research-prior-20261029/verification-d64.json`.
Do not rerun the training fit to replace a pending verification result. The five
declared ablations add 240 runs to the 858 primary runs. Static arithmetic and JSON
parsing agree with that 1,098-run inventory; OpenSpec strict validation passes.
Those checks do not establish algorithm correctness or any experiment outcome.
See [design progress evidence](../reports/ablation-plan-2026-10-29.json).

Read-only resource diagnosis found CLIProxyAPI private working set 2,337.3 MiB,
89 established TCP connections, and multiple active Codex chats. Connections do
not prove request activity, and parent-process absence does not prove this daemon
is unused. It was retained; no process was stopped to obtain the RAM threshold.

## Self-QA handoff

Trạng thái: CHƯA XONG bước tuning — còn test thực thi và kiểm chứng dữ liệu thật.

Gate: 17 tests đỏ trước triển khai do thiếu module; sau triển khai bị chặn trước khi pytest chạy. Cú pháp Python đã parse; không thay cho test.

E2E: Fit prior 64 chiều chạy thật xong, exit 0; chưa chạy CLI tuning hoặc kiểm chứng độc lập prior mới.

Probe: 3 lần preflight sau triển khai đều chặn đúng dưới 8 GiB; các probe chức năng mới chưa được thực thi.

Đường hỏng: Các test đã chuẩn bị nhãn held-out bị sửa/hỏng, thiếu outcome/context, sai grid, prior chưa sẵn sàng, checksum đổi và ghi đè. Chưa có kết quả thực thi để kết luận đạt.

Người thứ hai: n/a — CLI local một người, một worker.

Lưu trữ: Fit đã ghi checkpoint; mới đọc metadata, còn kiểm chứng độc lập. Tuning chưa tạo output thật.

Chưa chạy: Targeted/full tests, prior readback, 24 candidates trên 1.000 tuning RouterBench và 398 tuning MMLU, replay kết quả và cấu hình ngân sách. Test thường chỉ vài giây khi đủ RAM; thời lượng tuning 64 chiều chưa đo.

Bug đã biết: Chưa có kết luận runtime; mã mới chưa được xác minh. Manifest prepare/fit cũ vẫn giữ các mô tả trạng thái lịch sử đã được báo cáo mới bổ sung.

Quyết định: Grid alpha/window theo paper; số chiều/epoch, seed, cách tách MMLU và quy tắc ngân sách được khai báo là lựa chọn local trước khi xem kết quả tuning/deployment.

Tôi sẽ lo: Chạy hết các kiểm tra còn thiếu theo thứ tự khi đủ RAM, sửa lỗi nếu có, rồi tiếp tục thí nghiệm đầy đủ.

Cần bạn: Không cần thêm key/API. Điều kiện đang thiếu là ít nhất 8 GiB RAM vật lý trống theo AGENTS.md.
