# Kiểm tra tích hợp OpenAI — 31/10/2026

Phạm vi người dùng yêu cầu: sửa mâu thuẫn tài liệu; đóng gói cache/đầu vào,
thử checkout riêng; review và tích hợp PR #1. Slide/poster chỉ cập nhật hướng
dẫn Markdown cho Claude. Không xác nhận nộp bài hoặc khả năng bảo vệ thay nhóm.

## Bằng chứng khôi phục và chạy thật

Git clone riêng từ remote `codex/paper-preflight` tại research commit
`73918157547397bcaf538b282f17ad3b4e683262`. Môi trường Python 3.11.18 `.venv`
mới được cài bằng `uv sync --locked --extra cost --extra tracking --extra openai`.
Không copy `.venv`, `.env` hoặc ignored lineage từ máy cũ. Bốn file CLI/test/
manifest bổ sung được kiểm trong clone trước khi commit; source thí nghiệm gốc
giữ nguyên fingerprint. Gói cache được đưa vào qua ZIP với manifest 139 file.

| Kiểm tra | Kết quả |
|---|---|
| Restore vào checkout trống | 139 file đúng SHA-256; không có trace learning/deployment cũ |
| Restore lần hai | 0 file ghi mới, 139 file tái sử dụng |
| ZIP lưu trên GitHub | Digest phía GitHub khớp; tải lại toàn bộ 196.159.605 byte từ Release nháp và SHA-256 vẫn trùng |
| Refit prior seed 7 | 64D, 30/40 epochs; `prior.json` trùng byte với prior công bố; 186/313 calibration, 1.193 triplet hợp lệ |
| Luồng core thật | RouterBench/PILOT/seed 7: 32.270 learning rounds, 4 nested snapshots, 11 deployment conditions |
| Chặn mạng trong refit và producer | Python socket connect/create_connection bị từ chối; 0 request API model |
| Resume lần hai | Cả 39 file đầu ra trùng SHA-256 trước/sau |
| Reader độc lập | Tự replay 32.270 vòng học và 29.076 vòng deployment đã phục vụ, cả 11 điều kiện đạt |
| So báo cáo đóng băng | Cả 11 điều kiện khớp served/reward/estimated cost/actual cost trong `per-seed.csv` |
| Báo cáo khoa học | 14 package manifest khớp; không sửa byte bằng chứng |
| File trình bày | Cả 7 hash PPTX/PDF/reference giữ nguyên; draft hiện tại 22 slide và poster 1 slide |

ZIP, SHA-256, manifest và các lệnh khôi phục/replay nằm trong
[hướng dẫn hiện hành](openai-reproduce.md). Prior và tuning có trong ZIP;
phép thử refit chứng minh một prior chạy lại từ cache, còn group replay dùng
đúng protocol/tuning đã đóng băng. Không gọi đây là một lượt retrain mọi nghiên cứu.

## Regression và đường hỏng

Baseline `origin/main` tại `a0551aa`: **490 passed, 1 skipped**.
19 test mới của bundle đã đỏ khi thiếu implementation, rồi **19 passed**.
Full regression trên checkout mới: **700 passed, 1 skipped trong 48,32 giây**;
skip symlink do quyền Windows giống baseline. Lệnh: `python -m pytest -q
--tb=short --basetemp=.cache/release-full-final-20261031` sau khi RAM đạt
8,22 GiB, không có job nặng khác; BLAS chạy một thread.

Reviewer độc lập chỉ đọc code: không còn finding quan trọng trong phạm vi
review. Một P2 đã sửa: hướng dẫn phải checkout tag release cố định trước khi
cài/chạy, thay vì dùng `main` có thể thay đổi. Review không thay thế việc tự
chứng minh toàn bộ toán học bổ sung, chạy lại neural/full-grid, đọc lại mọi
dịch vụ bên ngoài hoặc kiểm bố cục/nộp bài/bảo vệ. Các việc đó được tách khỏi
phạm vi phép thử này; file sửa dở không liên quan của người dùng không đưa vào PR.

Đã thử: archive sai checksum; member sai bytes; thiếu/thừa/trùng member;
đường dẫn thoát thư mục, tên file Windows không hợp lệ hoặc giống credential;
source bị sửa; file đích xung đột; chạy restore hai lần; khôi phục sau khi chỉ
một member đã có. Các lỗi integrity/conflict bị từ chối trước khi ghi dữ liệu
đích. Nội dung file người dùng không bị ghi đè.

Lần baseline đầu gặp lỗi `tmp_path` vì Windows từ chối thư mục temp dùng chung
(323 passed, 168 setup errors); lần thứ hai thiếu thư mục cha của basetemp.
Đã tạo `.cache` và dùng basetemp mới trong checkout; baseline đạt như trên.
Lần định chạy full suite đầu bị resource preflight chặn ở 7,78 GiB RAM, nên
không khởi chạy suite và không dừng tiến trình khác để giải phóng RAM.
Người dùng chọn chờ đủ RAM; suite chỉ được khởi chạy sau khi đạt ngưỡng trên.

## Bàn giao kỹ thuật

Trạng thái: phần code/tài liệu/repro đã chạy thật; trạng thái tích hợp xem
[PR #1](https://github.com/kiettheis0702/pilot-llm-routing/pull/1).
Gate: 700 passed / 1 skipped; baseline 490 passed / cùng 1 skipped.
E2E: checkout remote riêng + môi trường mới + ZIP đã pin; 7 bước restore,
restore lại, refit, producer, resume, independent replay và so CSV đều đạt.
Probe: 19 case bundle đạt; integrity/conflict bị từ chối trước khi ghi.
Người thứ hai: n/a, CLI một người. Lưu trữ: đọc lại 139 file đầu vào và
39 file kết quả bằng checksum. Chưa chạy: full-grid/neural và slide render,
được nêu ngoài phạm vi bên dưới. Bug quan trọng đã biết: không có sau review.
Quyết định: gói chỉ chứa đầu vào core rõ ràng; tag cố định source; giữ nguyên
các báo cáo khoa học và trình bày. Không cần thêm quyết định nội dung từ người dùng.

## Trạng thái và giới hạn

- Tài liệu BGE cũ có nhãn lịch sử, dẫn tới trạng thái/kết quả OpenAI và bản
  hướng dẫn Claude hiện hành. `CONTEXT.md` phân tách mốc cũ với hiện tại.
- [Claude update map](CLAUDE_UPDATE_OPENAI_2026-10-31.md) thêm quy tắc ưu tiên
  nguồn, phạm vi reproducibility, kiểm số liệu/citation/notes/chart data và
  yêu cầu render mọi slide/poster. File trình bày chưa được cập nhật ở lượt này.
- Trạng thái tích hợp xem [PR #1](https://github.com/kiettheis0702/pilot-llm-routing/pull/1).
  [Release cố định](https://github.com/kiettheis0702/pilot-llm-routing/releases/tag/openai-reproduction-2026-10-31)
  cung cấp tag source và ZIP; không dùng archive cùng một revision tùy ý.
- Không chạy lại cả 1.098 điều kiện OpenAI, 858 Instructor, hai neural Hybrid
  fits hoặc cài mới môi trường CUDA/neural. Cache model/neural lineage ngoài
  phạm vi gói core không được tuyên bố đã khôi phục.
- Không gọi API model, đăng lại W&B, thay billing hoặc nộp Moodle.
