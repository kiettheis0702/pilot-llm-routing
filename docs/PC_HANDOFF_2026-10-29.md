# PILOT — bàn giao sang PC, 29/10/2026

Tài liệu này dùng cho người dùng và Codex trên PC mới. Đây là trạng thái đã kiểm
chứng và hướng dẫn chuyển máy, không thay thế các proposal/design/spec/tasks trong
`openspec/changes/`.

## 1. Những thứ cần mang sang

| Thành phần | Nguồn | Cách chuyển |
| --- | --- | --- |
| Code, tests, OpenSpec, cấu hình, báo cáo | `https://github.com/kiettheis0702/pilot-llm-routing.git` | Clone nhánh **`codex/data-gate`**. |
| Dữ liệu thật | Máy cũ: `D:\DOWNLOAD GG\PILOT_DATA` | Copy nguyên thư mục qua USB/SSD/LAN, giữ cấu trúc bên trong. |
| Tài liệu bàn giao gốc, tùy chọn | `D:\DOWNLOAD GG\PILOT_PROJECT_CONTEXT_HANDOFF.md` và `D:\DOWNLOAD GG\PILOT_CODEX_FILES_RESENT_2026-10-28` | Copy nếu muốn giữ đầy đủ tài liệu lịch sử. Trạng thái hiện tại nằm trong repo. |

Không cần copy repo cũ trong OneDrive, `.venv`, thư mục tạm, Python/npm cache hoặc
các thư mục `outputs/` thử nghiệm. Chưa có embedding/checkpoint thật cần chuyển.
Các checkpoint tổng hợp nhỏ dùng làm bằng chứng đã có trong `reports/` trên Git.
Không đưa dữ liệu thật, tài khoản hay token vào Git.

Dữ liệu được kiểm tra lúc bàn giao: **7 file, 484.074.124 byte = 461,65 MiB**.
Sáu file nguồn/processed khớp checksum đã pin từ trước; file metadata tải xuống
được chụp checksum hiện tại. Danh sách đầy đủ nằm trong
[`pc-transfer-manifest-2026-10-29.json`](../reports/pc-transfer-manifest-2026-10-29.json).

```text
PILOT_DATA/
  download_manifest.json
  arena_55k/train.csv
  routerbench_0shot/routerbench_0shot.pkl
  routerbench_5shot/routerbench_5shot.pkl
  processed/routerbench_0shot/queries.parquet
  processed/routerbench_0shot/outcomes.parquet
  processed/routerbench_0shot/splits.csv
```

`download_manifest.json` có thể chứa đường dẫn máy cũ; đó là metadata lịch sử,
không phải đường dẫn bắt buộc cho PC. Chạy lệnh với `--data-dir` của máy mới.

## 2. Cài công cụ trên PC Windows

Cấu hình PC người dùng cung cấp: Core Ultra 7 265KF, RAM 32 GB DDR5,
RTX 5070 12 GB. Bước setup này chạy CPU và không cần cài CUDA/PyTorch.

Mở PowerShell. Nếu chưa có Git/uv, cài bằng lệnh theo tài liệu chính thức:

```powershell
winget install --id Git.Git -e --source winget
winget install --id astral-sh.uv -e --source winget
```

Nguồn: [Git for Windows](https://git-scm.com/install/windows),
[uv installation](https://docs.astral.sh/uv/getting-started/installation/).
Nếu không có WinGet, dùng bộ cài trên hai trang này. Sau khi cài, đóng và mở lại
PowerShell để nhận PATH; kiểm tra `git --version` và `uv --version`.

## 3. Clone đúng nhánh và tạo môi trường mới

Ví dụ mặc định đặt dưới thư mục người dùng để không phụ thuộc PC có ổ D hay tên
tài khoản giống máy cũ. Có thể đổi `$pilotBase` sang một thư mục local trên SSD.
Các khối lệnh sau dùng cùng cửa sổ PowerShell; nếu mở cửa sổ mới, khai báo lại biến.

```powershell
$pilotBase = Join-Path $env:USERPROFILE 'PILOT'
$pilotRepo = Join-Path $pilotBase 'pilot-llm-routing'
$pilotData = Join-Path $pilotBase 'PILOT_DATA'
New-Item -ItemType Directory -Path $pilotBase -Force | Out-Null

git clone --branch codex/data-gate https://github.com/kiettheis0702/pilot-llm-routing.git "$pilotRepo"
if ($LASTEXITCODE -ne 0) { throw 'Clone failed; inspect the Git error.' }
Set-Location -LiteralPath $pilotRepo
git status --short --branch

uv python install 3.11
if ($LASTEXITCODE -ne 0) { throw 'Python installation failed.' }
uv sync --locked --python 3.11
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed.' }
```

Nếu `$pilotRepo` đã tồn tại, kiểm tra repo/branch/status trước; không xóa hoặc
reset để chạy lại đoạn clone. Không dùng `--offline` cho lần cài đầu nếu máy chưa
có cache. `uv.lock` cố định dependencies; không chạy nâng cấp toàn bộ dependencies.
`uv python install 3.11` chọn bản vá 3.11 hiện có; lần kiểm chứng cũ dùng 3.11.0.
Ghi phiên bản thực tế trên PC, không mặc định hai môi trường có cùng bản vá.

Repo có sẵn `openspec/` và `.agents/skills/`. OpenSpec CLI và các plugin cài ở máy
cũ không được chuyển qua Git; Codex kiểm tra/cài CLI khi cần triển khai tiếp.
**Không khởi tạo OpenSpec lại.** Git, Python và uv đủ để chạy các bước kiểm tra dưới.
Nếu tiếp tục push, đăng nhập GitHub và cấu hình Git author của người dùng trên PC.

## 4. Copy dữ liệu

Trên máy cũ, copy nguyên `D:\DOWNLOAD GG\PILOT_DATA` sang USB/SSD hoặc thư mục
chia sẻ. Trên PC, copy nó thành thư mục `$pilotData` đã chọn ở trên, tránh tạo
thêm lớp `PILOT_DATA\PILOT_DATA`.

Có thể dùng File Explorer. Nếu dùng PowerShell, ví dụ USB chứa
`E:\PILOT_DATA` — **đổi đường dẫn nguồn cho đúng trước khi chạy**:

```powershell
robocopy 'E:\PILOT_DATA' "$pilotData" /E /Z /R:2 /W:2 /XJ
if ($LASTEXITCODE -ge 8) { throw 'Data copy failed; inspect Robocopy output.' }
```

Chọn thư mục đích mới hoặc đúng bản copy đang dở. Lệnh trên copy cả thư mục con
và có thể tiếp tục khi gián đoạn. Theo [Microsoft Robocopy](https://learn.microsoft.com/en-us/windows-server/administration/windows-commands/robocopy),
mã thoát từ 8 trở lên báo lỗi copy; không áp dụng quy tắc chỉ mã 0 thành công cho
Robocopy. Giữ bản gốc ở máy cũ đến khi kiểm tra checksum bên dưới đạt.

## 5. Kiểm tra 7 file sau khi chuyển — không tải hay ghi lại dữ liệu

Chạy tại repo, với `$pilotData` trỏ đến thư mục vừa copy:

```powershell
$pilotManifestPath = Join-Path $pilotRepo 'reports\pc-transfer-manifest-2026-10-29.json'
$pilotManifest = Get-Content -LiteralPath $pilotManifestPath -Raw -Encoding UTF8 | ConvertFrom-Json
$pilotChecked = 0
foreach ($pilotEntry in $pilotManifest.files) {
    $pilotPath = Join-Path $pilotData $pilotEntry.relative_path
    if (-not (Test-Path -LiteralPath $pilotPath -PathType Leaf)) {
        throw "Missing file: $($pilotEntry.relative_path)"
    }
    $pilotLength = (Get-Item -LiteralPath $pilotPath).Length
    $pilotHash = (Get-FileHash -LiteralPath $pilotPath -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($pilotLength -ne $pilotEntry.size_bytes -or $pilotHash -ne $pilotEntry.sha256) {
        throw "Size/hash mismatch: $($pilotEntry.relative_path)"
    }
    $pilotChecked++
}
if ($pilotChecked -ne $pilotManifest.file_count) { throw 'Manifest count mismatch.' }
Write-Output "PASS: $pilotChecked files match the transfer manifest."
```

Kỳ vọng: `PASS: 7 files match the transfer manifest.` Nếu sai, xác định file copy
thiếu/hỏng và copy lại; không thay checksum mong đợi để làm cho kiểm tra pass.
Chưa cần chạy lại `audit_data.py`: lệnh đó tạo lại processed data và ghi đè báo cáo
audit mặc định, trong khi mục tiêu chuyển máy là giữ nguyên các artifact đã xác minh.

## 6. Chạy kiểm chứng trên PC

Sau khi cài đủ dependencies và checksum đạt, chạy tại `$pilotRepo`:

```powershell
uv run --locked --offline --no-sync python -B -m pytest -q -p no:cacheprovider
if ($LASTEXITCODE -ne 0) { throw 'Tests failed.' }

$pilotRunId = Get-Date -Format 'yyyyMMdd-HHmmss-fff'
uv run --locked --offline --no-sync python -B -m scripts.run_toy_pipeline --output-dir "outputs/pc-toy-$pilotRunId"
if ($LASTEXITCODE -ne 0) { throw 'Toy pipeline failed.' }

uv run --locked --offline --no-sync python -B -m scripts.run_baselines --source routerbench --data-dir "$pilotData" --rows 64 --dimension 8 --seed 42 --output-dir "outputs/pc-routerbench-$pilotRunId"
if ($LASTEXITCODE -ne 0) { throw 'RouterBench smoke failed.' }

git status --short --branch
```

Kỳ vọng với code hiện tại:

- **310 tests pass**. Thời gian sẽ khác máy cũ; chưa benchmark PC.
- Toy: 50 vòng learning, 6 query deployment, router giữ nguyên trong deployment.
  Chi phí ước tính 1,47; chi phí thực tế giả lập 3 trên ngân sách 2; phải báo actual
  overspend. Đây là fixture cố ý, không phải lỗi ngân sách bị che giấu.
- RouterBench smoke: **15 methods × 64 rounds**, tạo trace/JSON và PNG/SVG.
  Features ở lệnh này vẫn là hash vectors tổng hợp; không dùng số này làm kết quả paper.
- Các output mới ở `outputs/` được Git ignore. Repo nên giữ trạng thái sạch nếu
  chưa sửa code/tài liệu. Mỗi lần chạy dùng thư mục output mới.

Lần kiểm chứng trước trên clone/môi trường mới: 310 tests trong 23,59 giây,
toy 0,195 giây, toy baselines 7 methods × 100 rounds trong 2,908 giây. Xem
[`reproducibility-2026-10-29.json`](../reports/reproducibility-2026-10-29.json).
Đó là bằng chứng máy cũ, không phải thời gian dự đoán cho PC.

## 7. Trạng thái kỹ thuật để Codex tiếp quản

Repo: `https://github.com/kiettheis0702/pilot-llm-routing.git`.
Nhánh công việc: `codex/data-gate`. Baseline trước commit tài liệu bàn giao:
`eaca6dabd7786ea6bf792a939fd251a0f0a87809`. Clone tip mới nhất để lấy thêm file này;
không checkout lùi về baseline nếu muốn tiếp tục làm việc.

Các mốc code đã push:

| Commit | Nội dung |
| --- | --- |
| `5e8f499` | B01–B03: simulator, frozen splits, evaluator. |
| `ececbd3` | Random, fixed, LinUCB, ExploreOnly, EpochGreedy. |
| `9e1721f` | Strict preference preparation/readiness. |
| `3dc026b` | Tuning-only cost estimator, binning/rollover, actual ledger. |
| `19de9dd` | Prior-informed cosine PILOT và model checkpoint. |
| `caddb0d` | Bounded baseline CLI và trace-derived plots. |
| `0774eea` | Fixed-router held-out deployment API. |
| `22ba6e4` | NumPy two-stage preference pretraining core. |
| `5de84dd` | Full synthetic preference → learning → deployment CLI. |
| `eaca6da` | Fresh-environment verification và trạng thái research gates. |

Đọc theo thứ tự:

1. File này, `README.md`, `docs/implementation-status.md`.
2. `docs/data-gate-audit.md`, `docs/data-gate.md`, `docs/preference-readiness.md`.
3. `docs/ambiguities.md`, `docs/reproduction-log.md`.
4. OpenSpec change và code/test liên quan đến việc sắp làm; không tạo kế hoạch
   cạnh tranh với OpenSpec. Tám change hiện tại đã hoàn thành phạm vi được mô tả,
   qua strict validation và chưa archive.

Nếu repo được người dùng index và có `.codegraph/`, dùng CodeGraph trước khi tìm
hiểu/định vị code; nếu không có thì bỏ qua, không tự tạo index.

Modules chính: `data_gate.py`, `bandit_contracts.py`, `splits.py`, `evaluation.py`,
`policies.py`, `preference_data.py`, `pretraining.py`, `pilot.py`, `cost_policy.py`,
`deployment.py`, `run_baselines.py`, `run_toy_pipeline.py` trong `scripts/`.

## 8. Những điều chưa được giải quyết

- RouterBench 0-shot: 36.497 queries, 401.467 outcomes, 11 arms;
  1.000 tuning / 32.270 learning / 3.227 deployment. File đã pin nhưng chưa xác nhận
  chính là artifact của tác giả; 86 raw task labels so với 64 tasks trong paper.
- Arena 55K: 57.477 dòng; strict filter chỉ giữ 374 cặp giữa
  `gpt-3.5-turbo-1106` và `gpt-4-1106-preview`, chia 299 train / 75 holdout.
  Chín arm còn lại có mapping ứng viên chưa xác minh checkpoint/version.
  Không được biến chúng thành `exact_id` chỉ để có đủ dữ liệu.
- Chưa có embedding thật hoặc prior 11 arm. Chưa quyết định nguồn embedding;
  lựa chọn local sẽ là sai khác so với embedding của paper.
- Chưa pin token-cost order cho triplets, tokenizer, tuning output-token counts,
  historical input/output prices và real cost bounds.
- NumPy pretraining hiện là small-input correctness core, sampler còn liệt kê
  tổ hợp và JSON cache giới hạn 16 MiB. Cần thiết kế batching/cache/sampling phù
  hợp trước khi chạy lớn; chuyển sang PC không tự sửa hạn chế này.
- Deployment API hiện nhận PILOT cosine predictions. Fair deployment của các
  baseline vẫn cần chốt semantics; không âm thầm clip raw linear scores.
- Chưa chạy Figure 3/4, full-data tuning, multi-seed research, ablations hoặc
  thí nghiệm phản biện có thống kê. Chưa có kết quả thật để làm poster.

## 9. Quyền đã có và điều chưa được người dùng chọn

Người dùng đã yêu cầu làm mọi phần khả thi và **commit/push từng phần hoàn thành**
lên nhánh công việc. Không cần hỏi lại cho commit/push thông thường của phần đã
kiểm chứng. Không suy ra quyền merge vào main, deploy hoặc công bố kết quả.

Người dùng chưa chốt mức độ chấp nhận sai khác với paper, embedding model,
ngân sách API hay giới hạn chạy training thật. Khuyến nghị ưu tiên miễn phí đã
được nêu, nhưng **khuyến nghị không phải sự chấp thuận của người dùng**. Xác minh
nguồn công khai và chuẩn bị kế hoạch có thể tiếp tục; trước tải model lớn, gọi API
trả phí hoặc chạy full training, đưa ước lượng cụ thể và lấy quyết định còn thiếu.

Giữ chosen-arm-only feedback; oracle chỉ ở evaluator. Learning không áp ngân sách;
deployment không cập nhật router. Tuning và deployment không trộn. Không dùng
actual deployment output/cost làm estimated cost trước khi chọn. Phân biệt
PAPER_EXACT / PAPER_INFERRED / DEVIATION / EXTENSION. Không dùng fake/random prior
để tuyên bố đã train thật. Không đổi default sang kiet-harness.

Chưa có lint/typecheck/build/E2E/CI được cấu hình. Không nhận pytest là bằng chứng
cho những check đó. Chưa chạy dedicated security audit. Các bản dữ liệu thật vẫn
để ngoài Git; kiểm tra license trước phân phối lại, xem audit.

## 10. Bước tiếp theo và prompt mở chat trên PC

Mở thư mục repo mới trong Codex. Kiểm tra setup/checksum/tests trước, sau đó ưu tiên
xác minh mapping/source preference và protocol; không cần xây lại các phần đã có.
Chọn embedding sau khi biết model pool và yêu cầu môn học. Nếu chọn GPU, kiểm tra
driver bằng `nvidia-smi`, cài backend hỗ trợ RTX 5070/Blackwell trong một thay đổi
có kiểm thử và chạy probe nhỏ; NumPy hiện tại không dùng CUDA.

Dán prompt này, thay đường dẫn dữ liệu bằng đường dẫn thực tế trên PC:

```text
Đọc docs/PC_HANDOFF_2026-10-29.md và docs/implementation-status.md trước.
Tôi đang tiếp tục PILOT trên PC Ultra 7 265KF / RAM 32 GB / RTX 5070 12 GB.
Repo đang ở nhánh codex/data-gate; dữ liệu nằm tại <đường dẫn PILOT_DATA trên PC>.
Kiểm tra Git, môi trường, checksum 7 file bằng transfer manifest, rồi chạy tests,
toy pipeline và RouterBench smoke nhỏ theo handoff. Không rebuild Data Gate.
Sau đó tiếp tục xác minh preference model mappings và những dữ kiện còn thiếu
để đề xuất protocol/embedding phù hợp. Bảo toàn code, dữ liệu và split đã pin.
Tôi đã cho phép commit/push từng phần hoàn thành; chưa chốt ngân sách API,
download model lớn hoặc full training. Không suy ra quyền merge/deploy.
Ghi rõ kiểm tra thực sự đã chạy và các quyết định còn cần tôi chọn.
```

## 11. Kiểm tra tài liệu bàn giao trên máy cũ

Đã parse thành công cả năm khối PowerShell và chạy nguyên khối kiểm tra checksum
ở mục 5 với dữ liệu nguồn: `PASS: 7 files match the transfer manifest.` Manifest
lưu danh sách file theo đường dẫn tương đối, kích thước, hash và metadata chuyển
máy; không chứa hội thoại thật.
Các bước cài đặt/copy và chạy trên PC chưa được thực thi; cần xác nhận ở máy mới.
Thay đổi bàn giao chỉ gồm tài liệu và metadata, không sửa code Python hay lockfile.
