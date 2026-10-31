# Bàn giao thiết lập embedding OpenAI — 30/10/2026

Trạng thái: **XONG — thiết lập local đã chạy thật; chưa chạy embedding trả phí.**

- Gate: `.venv/Scripts/python.exe -m pytest tests/test_openai_embedding_setup.py tests/test_paper_preflight.py tests/test_embedding_store.py -q --basetemp outputs/openai-setup-qa/final-reviewed --junitxml outputs/openai-setup-qa/final-reviewed.xml` → **59 passed**. SDK 2.54.0 đã cài; optional dependency đã khóa trong `uv.lock`; diff whitespace sạch.
- E2E: CLI thật dùng snapshot 38.018 văn bản / 6.867.235 token / 298 request. Chạy doctor hai lần, fake key, thiếu approval, approval chưa duyệt, payload hỏng và file key hỏng. Build marker `openai-embedding-setup-v1`; guard socket đã được chứng minh bằng negative control.
- Probe: **7 CLI probe chạy, 7 đạt; 0 lần thử kết nối, 0 request trả phí.** Bằng chứng máy đọc được: `outputs/openai-setup-qa/offline-cli-verification.json`.
- Đường hỏng: SDK thật với MockTransport synthetic ép 401/429/500, timeout sau create, tải file đứt giữa chừng, thoát sau publish, payload đổi sau preflight, replay approval sau reformat/chuyển plan; không retry/gửi job thay thế. Test chặn toàn bộ socket thật.
- Người thứ hai: n/a — CLI một người, chạy tuần tự; không chạy thử tải đồng thời.
- Lưu trữ: đọc lại state/file ID/Batch ID/request ID/checksum; khôi phục đúng Batch bằng metadata; tải lại chỉ GET. Receipt theo approval ID ổn định, dùng chung trong repo.
- Chưa chạy: OpenAI thật, xác thực key/quota/billing, upload, Batch generation và import vector thật — người dùng chưa nhập key hoặc duyệt lần chạy trả phí. Những bước này nằm ngoài phạm vi setup miễn phí; không mô tả MockTransport là kết quả provider thật.
- Bug đã biết: không còn bug material trong phạm vi đã review. Gate khoa học tám/11 model và chi tiết paper vẫn theo báo cáo preflight.
- Quyết định: giữ snapshot và trọng số BGE cũ; CLI riêng chỉ quản lý API/Batch. Mặc định doctor offline; key local không phải approval. Giá Batch ước tính $0.06867235, Standard bảo thủ $0.1373447, chưa gồm thuế; chưa có phí thực tế.
- Tôi sẽ lo: sau khi có phê duyệt rõ ràng cho một lần chạy cụ thể, kiểm lại snapshot/giá/phạm vi trước khi gửi; dùng state để theo dõi, không tạo job thay thế khi mạng lỗi.
- Cần bạn: nhập key vào `.env.openai` và lưu. Chưa cần duyệt chi phí trong bước này.

Review độc lập phát hiện hai lỗi trước bàn giao: receipt theo raw JSON/path có thể bị replay và metadata download ghi sau publish không khôi phục được một cửa sổ crash. Đã sửa, thêm regression; reviewer đọc lại xác nhận cả hai đã xử lý, không còn finding material.

Tuân thủ tài nguyên: mỗi lần chạy test đều preflight RAM và tiến trình; một process test, BLAS một luồng. Có một lượt CLI bị chặn ở 7,90 GiB; đã đợi, không hạ ngưỡng/kill tiến trình. Sau đó toàn bộ CLI probe chạy đạt.

Thiết lập chưa commit/push. [Hướng dẫn sử dụng](../docs/openai-embedding-setup.md).
