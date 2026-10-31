# Khôi phục OpenAI offline từ checkout sạch

Đây là hướng dẫn hiện hành cho kết quả OpenAI ngày 31/10/2026. Các hướng dẫn
chuyển PC/BGE ngày 29–30/10 được giữ làm lịch sử. Đọc kết quả và giới hạn tại
[openai-followup-status.md](openai-followup-status.md); không lấy số liệu BGE
trong bản bàn giao cũ để dán nhãn OpenAI.

## Gói đầu vào độc lập với máy cũ

- [GitHub Release](https://github.com/kiettheis0702/pilot-llm-routing/releases/tag/openai-reproduction-2026-10-31)
- [Tải ZIP](https://github.com/kiettheis0702/pilot-llm-routing/releases/download/openai-reproduction-2026-10-31/pilot-openai-offline-inputs-2026-10-31.zip)
- [SHA-256 và địa chỉ tải đã pin](../configs/openai-reproduction-release.json)
- [Manifest 139 file: đường dẫn, kích thước, SHA-256](../configs/openai-reproduction-files.json)

ZIP: **196.159.605 byte**, khoảng **187,07 MiB**; giải nén **330,69 MiB**.
SHA-256: `095d39bec4d0a9c679816b8667d5cd1f952c0c5beb825dbd3fed0fca258f3968`.

Gồm bảng RouterBench đã chuẩn hóa và split, 36.497 vector routing + 1.558 vector
preference OpenAI 1.536D, văn bản/ID/hash đầu vào, dữ liệu chi phí, ba prior 64D,
tuning hai miền và protocol core đã đóng băng. Chỉ đóng gói file được liệt kê;
không lấy toàn bộ `outputs/`. Không có API key, `.env`, credential, SDK receipt,
raw provider response, neural checkpoint hoặc trace learning/deployment có sẵn.
Các query/preference text là đầu vào từ bộ dữ liệu công khai được dẫn trong
`configs/data_sources.json`; giữ nguyên nguồn và điều kiện sử dụng của chúng.

## 1. Clone, cài môi trường và khôi phục

Windows PowerShell, Python 3.11.18 và `uv`; chạy lần lượt, dừng khi lệnh lỗi.
Dành ít nhất 2 GiB đĩa cho clone, môi trường, ZIP và một nhóm replay. Trước test
hoặc chạy thí nghiệm phải còn ít nhất 8 GiB RAM vật lý và không có job nặng khác.
Không đóng/giết công việc khác để chạy thử. Dùng một BLAS thread như bên dưới.

```powershell
git clone https://github.com/kiettheis0702/pilot-llm-routing.git pilot-clean
Set-Location pilot-clean
git checkout --detach openai-reproduction-2026-10-31
$env:UV_CONCURRENT_DOWNLOADS='1'
$env:UV_CONCURRENT_INSTALLS='1'
$env:UV_CONCURRENT_BUILDS='1'
uv sync --locked --python 3.11.18 --extra cost --extra tracking --extra openai
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
$env:OPENBLAS_NUM_THREADS='1'
.venv/Scripts/python.exe -c "from scripts.resources import require_resources; print(require_resources())"
New-Item -ItemType Directory -Force .cache | Out-Null
$bundle = Get-Content configs/openai-reproduction-release.json -Raw | ConvertFrom-Json
$zip = Join-Path .cache $bundle.filename
Invoke-WebRequest -Uri $bundle.url -OutFile $zip
if ((Get-FileHash -Algorithm SHA256 -LiteralPath $zip).Hash.ToLowerInvariant() -ne $bundle.sha256) { throw 'Archive checksum mismatch' }
.venv/Scripts/python.exe -m tools.reproduction_bundle verify --archive $zip
.venv/Scripts/python.exe -m tools.reproduction_bundle restore --archive $zip
```

Cài SDK không gọi model. Không tạo `.env.openai` hoặc điền key cho luồng này.
Tag release cố định cả code lẫn manifest; không chạy protocol đã đóng băng trên
một `main` tương lai đã thay source. `verify_pins` cố ý từ chối code khác byte.
Lệnh restore kiểm toàn bộ archive và các đích trước khi ghi; thiếu/thừa file,
checksum sai, đường dẫn thoát thư mục hoặc file đích khác nội dung đều bị từ chối.
Chạy restore lần nữa chỉ tái sử dụng file đúng checksum, không ghi đè dữ liệu.
Nếu mất kết nối khi tải, giữ file tải dở riêng, tải lại ZIP rồi kiểm SHA-256;
không chạy embedding generation để thay thế cache thiếu.

## 2. Refit thật một prior từ cache

Thư mục đích phải chưa tồn tại. Đây là cấu hình seed 7 thật của nghiên cứu core,
không phải vector giả: 64D, 30 projection epochs và 40 prior epochs.

```powershell
.venv/Scripts/python.exe -m scripts.research_pipeline fit --prepared-dir outputs/local-pipeline-verification-20261029/prepared-identity --cache-dir outputs/openai-standard-20261030/imported/caches/source-1 --cost-sizes outputs/research-cost-inputs-20261029/cost-size-identity_match.json --output-dir outputs/reproduction-refit-seed7 --dimension 64 --projection-epochs 30 --prior-epochs 40 --seed 7
$newHash = (Get-FileHash outputs/reproduction-refit-seed7/prior.json -Algorithm SHA256).Hash
$oldHash = (Get-FileHash outputs/openai-core-20261031/priors/seed-7/prior.json -Algorithm SHA256).Hash
if ($newHash -ne $oldHash) { throw 'Refit differs: inspect environment; do not silently repin' }
```

Exact-byte equality đã được kiểm trên Windows/Python và lockfile nêu trên;
không cam kết mọi thư viện BLAS/hệ điều hành đều tạo byte float giống nhau.

## 3. Một luồng PILOT đầy đủ, không tải model

Chạy RouterBench/seed 7/PILOT với **32.270 vòng learning**, bốn snapshot lồng
nhau và **11 điều kiện deployment** của protocol thật. Dùng prior/tuning đã
khôi phục, không giảm số query rồi gọi đó là thí nghiệm đầy đủ.

```powershell
@'
import socket
def denied(*args, **kwargs):
    raise RuntimeError('Network disabled during offline reproduction')
socket.socket.connect = denied
socket.socket.connect_ex = denied
socket.create_connection = denied
from tools import openai_study as core
from scripts.experiments import verify_pins, execute_group
from tools.paper_preflight.common import sha256
base = core.DEFAULT_BASE
core.require_resources()
protocol = core.sealed(base/'protocol.json')
verify_pins(protocol, core.ROOT)
bundle = core.load_data(base, 'routerbench', 7)
bundle['protocol'] = protocol
bundle['lineage'].update(protocol_sha256=sha256(base/'protocol.json'),
    prior_index_sha256=sha256(base/'prior-index.json'),
    tuning_stage_sha256=protocol['tuning']['routerbench']['stage']['sha256'])
results = execute_group(bundle, base/'runs', method='pilot', resume=True)
print('Completed conditions:', len(results))
'@ | .venv/Scripts/python.exe -

.venv/Scripts/python.exe tools/research/verify_runs.py --domain routerbench --seed 7 --method pilot --base outputs/openai-core-20261031/runs --protocol outputs/openai-core-20261031/protocol.json --index outputs/openai-core-20261031/prior-index.json --cache-dir outputs/openai-standard-20261030/imported/caches/source-0 --output outputs/reproduction-check/independent-replay.json
```

Verifier đọc lại các trace, tự tính trạng thái, action, reward và chi phí; không
import phép toán routing của producer. Output verifier phải là đường dẫn mới.
Chạy lại đoạn Python dùng `resume=True` để kiểm tái sử dụng kết quả; không sửa
protocol hoặc checksum để ép resume. Muốn chạy các nhóm core còn lại, lệnh gốc
`python -m tools.openai_study run` dùng cùng đầu vào, chạy tuần tự và lưu tiến độ.
Đó là một lượt chạy lớn riêng; phép thử nhỏ này không chứng minh đã chạy lại cả grid.

## Phạm vi bằng chứng

- Git giữ code, cấu hình, test và các báo cáo khoa học đã đóng băng. ZIP bổ sung
  đúng những đầu vào core ngoài Git; không phụ thuộc đường dẫn tuyệt đối máy cũ.
- Clone mới có môi trường `.venv` mới từ `uv.lock`; gói tải của uv có thể lấy từ
  cache cài đặt trên máy. Đây không phải chứng minh tải mọi dependency từ Internet.
- Kiểm tra này gồm restore, refit một prior, một nhóm learning/deployment,
  independent replay, resume và regression. [Biên bản](release-readiness-2026-10-31.md).
- Không chạy lại toàn bộ 1.098 OpenAI conditions, 858 Instructor conditions,
  hai neural Hybrid fits hoặc toàn bộ nghiên cứu bổ sung từ môi trường mới.
  Gói này không có 29 GiB neural/raw follow-up lineage. Các báo cáo tương ứng
  vẫn là bằng chứng đã đóng băng; không gọi chúng là vừa retrain từ clone.
- Không có request API model mới. Khôi phục cache không phát sinh token embedding.
- Chưa xác nhận file đã nộp Moodle hoặc mức sẵn sàng bảo vệ của hai thành viên.

## Duy trì gói

Trên máy giữ đủ 139 file đúng hash, có thể tạo một ZIP mới bằng
`python -m tools.reproduction_bundle build --archive <duong-dan-moi.zip>`.
ZIP mới có thể khác hash vì timestamp/compression; không tự thay hash của release
đã công bố. Muốn xuất bản phiên bản mới phải cập nhật manifest/release rõ ràng.
