# Full embedding and first real prior — 2026-10-29

The full local caches and first eight-arm development prior are verified. The
research project is not complete. Three unresolved model bridges, paper dataset
choices, tuning/deployment protocol, full experiments, ablations and research
delivery remain open. Aggregate evidence:
[verification report](../reports/full-embedding-prior-2026-10-29.json).

## Artifacts and validation

| Artifact | Result |
| --- | --- |
| Routing cache | `outputs/research-embeddings-20261029/routing-full`: 36,497 × 1,024 float32 vectors; 1,000 tuning / 32,270 learning / 3,227 deployment; zero truncated inputs |
| Preference cache | `outputs/research-embeddings-20261029/preference-identity`: 1,558 × 1,024 vectors; 1,245 train / 313 holdout; 8 train and 5 holdout inputs truncated at 2,048 tokens |
| Prior | `outputs/research-prior-20261029/identity-d8-s42`: eight arms, dimension 8, projection epochs 2, prior epochs 5, seed 42 |
| Holdout | Independent cosine calculations reproduce 186/313 correct pairs (59.42%); all eight arms have support and finite inverse-accuracy coefficients |
| Regression | 362 passed, one Windows symlink skip; 6.19 seconds; one worker; preflight 8.90 GiB and no competing heavy job |

The 2,048-token cap was selected from training/tuning text before encoding. Prior
settings are development defaults, not a claim of reproducing the paper's setup.
The local 313-pair holdout differs from the paper's preference holdout. RouterBench
deployment outcomes did not select this cap or these prior settings. Prior
holdout labels initialize per-arm coefficients as specified by the existing
protocol; they are not the routing deployment split.

Full routing encoding used GPU batches of two, with a saved checkpoint every 32
batches. All 571 routing and 779 preference chunks independently pass checksums.
Unsaved suffix rows are re-encoded after interruption; the interruption test
deliberately corrupts the unsaved suffix and verifies its replacement. The GPU
batch size and active process were not capped or interrupted to save RAM.

## Commands used

From the repository root, after the RAM/heavy-job preflight:

```powershell
$env:TOKENIZERS_PARALLELISM='false'
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
.venv-embedding/Scripts/python.exe -m scripts.local_embeddings --inputs outputs/research-input-audit-20261029/inputs/preference-inputs.json --output-dir outputs/research-embeddings-20261029/preference-identity --recipe configs/local_embedding_full.json --batch-size 2
.venv/Scripts/python.exe -m scripts.research_pipeline fit --prepared-dir outputs/local-pipeline-verification-20261029/prepared-identity --cache-dir outputs/research-embeddings-20261029/preference-identity --cost-sizes outputs/research-cost-inputs-20261029/cost-size-identity_match.json --output-dir outputs/research-prior-20261029/identity-d8-s42 --dimension 8 --projection-epochs 2 --prior-epochs 5 --seed 42
.venv-embedding/Scripts/python.exe -m scripts.local_embeddings --inputs outputs/research-input-audit-20261029/inputs/routing-inputs.json --output-dir outputs/research-embeddings-20261029/routing-full --recipe configs/local_embedding_full.json --batch-size 2 --checkpoint-batches 32
.venv/Scripts/python.exe outputs/research-embeddings-20261029/verify_full_artifacts.py
.venv/Scripts/python.exe -m pytest -q --basetemp outputs/pytest-full-embedding-20261029-a --junitxml outputs/research-embeddings-20261029/pytest.xml
```

These output paths already exist. For another encoding pass use `--resume` with
identical inputs/recipe; both complete caches were successfully resumed using
the base environment without PyTorch. For another fit or verification report use
a fresh output path. Use a new pytest basetemp rather than reusing the recorded
test directory. The independent verifier is retained locally in ignored outputs;
its content checksum is recorded in the aggregate report.

## Self-QA handoff

Trạng thái: XONG phần cache/prior phát triển; CHƯA XONG toàn bộ nghiên cứu.

Gate: Lệnh pytest ở trên → 362 passed, 1 skipped; baseline đầu dự án 310 passed, giai đoạn cost 360 passed, 1 skipped.

E2E: BGE-M3 chạy thật trên RTX 5070; dữ liệu preference thật huấn luyện prior; nhánh `codex/data-gate`, chưa commit. Report lưu HEAD và hash mã đã chạy. Cache routing có 571 checkpoint, nhận diện thay đổi cadence mới.

Probe: 10 nhóm probe chạy, 10 đạt; danh sách và số lần chạy trong report. Không có tuyên bố kiểm thử race/tải.

Đường hỏng: Đổi recipe, đổi văn bản giữ nguyên hash chuẩn hóa, ghi đè cache → từ chối, file không thay đổi. Test ngắt checkpoint và làm hỏng suffix chưa lưu → resume ghi lại suffix đúng.

Người thứ hai: n/a — CLI local một người, chạy tuần tự.

Lưu trữ: Đọc trực tiếp JSON/NPY, tự tính checksum/norm/cosine, so đủ ID/split; chạy CLI resume lại trên cả hai cache, không nạp encoder.

Chưa chạy: Luồng học–triển khai thật hoàn chỉnh, budget sweep, nhiều seed, MMLU và ablation; cần hoàn thiện adapter/quy trình trước. Rà soát độc lập toàn nhánh và xác minh fresh clone sẽ thực hiện khi hoàn tất kế hoạch.

Bug đã biết: Metadata `research_blockers` trong manifest prepare/fit cũ là ảnh chụp tại thời điểm tạo, còn nhắc một số đầu vào đã hoàn thành. Không sửa artifact bất biến; report mới và status hiện tại nêu bằng chứng thay thế. Cần làm rõ metadata trạng thái trước lần tích hợp mới.

Quyết định: Nhóm 8 model và BGE-M3 là cấu hình phát triển đã ghi rõ; 3 model chưa có bằng chứng không được tự ghép. Tham số prior hiện là mặc định phát triển, chưa dùng để kết luận paper.

Tôi sẽ lo: Chốt quy trình và adapter, kiểm chứng tích hợp thật, chạy tuần tự thí nghiệm/đối chứng, tổng hợp báo cáo và poster; tiếp tục giữ riêng giới hạn tái lập đủ 11 model.

Cần bạn: Chưa cần thêm đầu vào để tiếp tục các bước offline; chưa có API trả phí nào được gọi.
