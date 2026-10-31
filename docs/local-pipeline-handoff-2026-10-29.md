# Bàn giao giai đoạn pipeline local — 29/10/2026

> **TÀI LIỆU LỊCH SỬ — BGE / mốc 29–30/10/2026.** Các số liệu, trạng thái “hiện tại/current”, nhánh Git và việc còn chờ bên dưới chỉ đúng tại mốc ghi nhận. Không dùng làm kết quả OpenAI hiện hành. Đọc [trạng thái OpenAI](openai-followup-status.md), [bản giao việc hiện hành cho Claude](CLAUDE_UPDATE_OPENAI_2026-10-31.md) và [khôi phục từ checkout sạch](openai-reproduce.md). Giữ nội dung cũ để truy vết, không ghi đè bằng kết quả mới.

Trạng thái: **XONG giai đoạn triển khai pipeline local và smoke GPU**. Code, kiểm thử CPU, môi trường embedding riêng, CUDA và BGE-M3 thật đã được kiểm chứng. Đây chưa phải kết quả tái lập paper hoặc prior thật đầy đủ. Ngoại lệ người dùng duyệt chỉ áp dụng cho test CPU; bước GPU bắt đầu sau khi preflight đo đủ 8 GiB.

Gate: `python -B -m pytest -q -rs` → **338 đạt, 1 bỏ qua**, baseline 310 đạt. Một test symlink thật bị Windows từ chối quyền tạo link; test kiểm tra chặn đường dẫn trước khi ghi vẫn đạt. OpenSpec strict validate và `git diff --check` đạt.

E2E: nhánh `codex/data-gate`, code chưa commit trên HEAD `47ce94384f9c4a709f9bc85a02c7eb344cce75ae`. Toy chạy từ prior đến deployment: 50 vòng learning, 6 query phục vụ; router giữ nguyên khi deployment. RouterBench: 15 phương pháp × 64 vòng, vector synthetic. Chuẩn bị dữ liệu thật bằng CLI theo ba chính sách, có `mapping_evidence` và `manifest_sha256` của bản mới. BGE-M3 chạy thật trên RTX 5070: 128 câu, 1.024 chiều, float32, chuẩn L2; VRAM cấp phát đỉnh khoảng 1,09 GiB.

Probe: **13 chạy, 13 đạt** — tám probe CPU/dữ liệu (đọc lại ba bộ dữ liệu; từ chối ghi đè; từ chối pool 11 model; kiểm bảy file gốc; kết quả toy; 15 trace RouterBench) và năm probe GPU/cache (phép toán FP16; encode thật; đọc lại vector; resume không nạp model; từ chối recipe đã đổi). Luồng fit/learn mới đã chạy xuyên suốt trên fixture trong test, chưa chạy với vector BGE-M3 thật.

Đường hỏng: test ép gián đoạn encoder/ghi manifest, đổi input/recipe, sửa vector đã ghi, NaN, thiếu nguồn giá, RAM thấp và job xung đột. Các trường hợp đều chặn hoặc khôi phục đúng; không tự thay prior/lambda hay gọi nhà cung cấp trả phí.

Người thứ hai: n/a — CLI local một người, chạy tuần tự.

Lưu trữ: đã đọc lại cache/checkpoint/trace từ đĩa; checksum của cả bảy file nguồn giữ nguyên. File nội dung dữ liệu và vector nằm trong thư mục output bị Git bỏ qua. Báo cáo lưu hash của code, lock và log kiểm thử.

Chưa chạy trong phạm vi mở rộng nghiên cứu: full cache preference/routing, prior thật và budget sweeps. Còn thiếu quyết định recipe sau audit toàn bộ độ dài và nguồn giá/tokenizer; không tự bịa thứ tự giá. Trong 128 câu smoke, không câu nào bị cắt, tối đa 493 token; 53,99 giây bao gồm khởi tạo và tải model, không phải phép đo tốc độ inference riêng.

Bug đã biết: không còn lỗi quan trọng đã xác nhận trong phạm vi đã triển khai sau review và smoke thật. Windows không có quyền tạo symlink nên Hugging Face dùng bản sao cache; không bật quyền admin/Developer Mode. Cảnh báo `torch_dtype` deprecated của Transformers chưa ảnh hưởng lượt chạy đã kiểm chứng.

Quyết định: mặc định chỉ hai model exact-ID; tùy chọn năm model có alias có bằng chứng hoặc tám model khớp danh tính. Còn thiếu bridge Claude-v2, Mistral và Mixtral. BGE-M3 là lựa chọn thay thế miễn phí chạy local; cap 512 token chỉ dùng probe, chưa phải cấu hình thí nghiệm cuối. Chưa tự đặt thứ tự giá token lịch sử hoặc gộp 86 task thành 64 task.

Tôi sẽ lo ở giai đoạn nghiên cứu tiếp theo: audit toàn bộ độ dài để chốt recipe, chuẩn bị full cache và prior khi có nguồn cost-size đáng tin cậy. Budget sweep và Figure 3/4 còn cần token counts, giá lịch sử, cost bounds và quy trình deployment chung cho baseline.

Cần bạn: hiện chưa cần key/API trả phí hay thêm thao tác để hoàn tất giai đoạn này. Theo yêu cầu giảm RAM, đã gom rác CLIProxyAPI một lần: RAM riêng đo từ 2.668,9 xuống 1.802,5 MiB; PID giữ nguyên, HTTP kiểm tra 200, cấu hình khôi phục đúng checksum và cổng chẩn đoán đã tắt. REA gom rác chỉ giảm khoảng 1 MiB, nên không tắt các tiến trình còn giữ trạng thái. Không đóng/lưu trữ chat hay ép dừng tác vụ. Lượt đọc profile PowerShell tạm dùng nhiều RAM rồi tự kết thúc; script lịch sử đã được chặn chạy lại. Sự tăng RAM trống còn có ảnh hưởng từ các tác vụ khác kết thúc, không quy toàn bộ cho GC.

Bằng chứng tổng hợp: [JSON report](../reports/local-pipeline-2026-10-29.json). Hướng dẫn chạy: [local research pipeline](local-research-pipeline.md).
