# Ablation design preparation handoff

> **TÀI LIỆU LỊCH SỬ — BGE / mốc 29–30/10/2026.** Các số liệu, trạng thái “hiện tại/current”, nhánh Git và việc còn chờ bên dưới chỉ đúng tại mốc ghi nhận. Không dùng làm kết quả OpenAI hiện hành. Đọc [trạng thái OpenAI](openai-followup-status.md), [bản giao việc hiện hành cho Claude](CLAUDE_UPDATE_OPENAI_2026-10-31.md) và [khôi phục từ checkout sạch](openai-reproduce.md). Giữ nội dung cũ để truy vết, không ghi đè bằng kết quả mới.

Trạng thái: CHƯA XONG thí nghiệm — đã khai báo thiết kế, còn kiểm chứng runtime.

Gate: JSON và cú pháp helper đọc được; số lượng tính lại khớp 858 + 240 = 1.098 lượt deployment. OpenSpec strict validation đạt. Pytest bị chặn trước khi khởi chạy tại 5,65 GiB RAM.

E2E: Chưa chạy tuning hoặc ablation. Fit prior 64 chiều là kết quả lượt trước; helper mới chưa xác nhận lại kết quả đó.

Probe: Kiểm tra tĩnh inventory khớp; không có probe chức năng mới đã chạy. Không cộng các kiểm tra tĩnh thành số test runtime đạt.

Đường hỏng: Cổng RAM từ chối trước khi import pytest; không tạo kết quả test giả hoặc giảm ngưỡng để chạy.

Người thứ hai: n/a — quy trình local một worker.

Lưu trữ: Đã lưu cấu hình ablation, mô tả phép so sánh, helper kiểm chứng prior và báo cáo tiến độ. Helper từ chối ghi đè kết quả kiểm chứng đã có.

Chưa chạy: 17 test tuning sau triển khai, regression, helper prior-64, tuning RouterBench/MMLU, replay tuning, đo tài nguyên, thí nghiệm chính và ablation. Thời lượng thí nghiệm 64 chiều chưa đo nên chưa đưa ETA.

Bug đã biết: Chưa có kết luận runtime cho mã tuning mới. Trạng thái trong manifest lịch sử không được dùng làm trạng thái hiện tại.

Quyết định: Năm biến thể giữ cùng split/seed/ngân sách; không retune ablation; công bố cả kết quả bất lợi; các khác biệt so với paper và ba mapping chưa rõ vẫn được giữ nguyên.

Tôi sẽ lo: Chạy các bước còn thiếu tuần tự khi đủ RAM; lấy số đo thực tế để cập nhật dự toán trước thí nghiệm đầy đủ.

Cần bạn: Không cần key/API; điều kiện chạy tiếp vẫn là ít nhất 8 GiB RAM vật lý trống theo AGENTS.md. CLIProxyAPI có kết nối established và chưa được chứng minh là không dùng, nên giữ nguyên.

Evidence: [ablation-plan-2026-10-29.json](../reports/ablation-plan-2026-10-29.json).
