# Việc hữu ích đã làm tiếp trước khi trả phí

Ngày30/10/2026. Nhánh `codex/paper-preflight`. Phần việc này chạy offline,
không gửi API, không đọc key, không upload, không sửa slide/poster.

- [Quyết định phạm vi và hướng dẫn dùng](../../docs/openai-workflow.md).
- [Protocol main core](../../configs/openai_core_protocol.json).
- [Kiểm tra toàn luồng1536D synthetic](synthetic-e2e.json).
- [Một fit BGE thật](real-bge-training.json), [kết quả500 cặp](real-bge-final-test.json).
- [Verification và source hashes](verification.json), [QA](qa-handoff.md),
  [review](independent-review.md), [log test cuối](tests-final.txt).

## Đã thực hiện

1. Import file OpenAI Batch trả về sai thứ tự bằng `custom_id/index`. Kiểm
   model/chiều/vector/nguồn, lưu chunk/request ID, tiếp tục từ output thiếu hoặc
   đứt giữa file; không tạo cache hoàn tất khi còn thiếu, không tự gửi lại API.
2. Sửa bộ kiểm cache BGE-only để nhận profile OpenAI đã xác minh input chính xác.
   Không nới validation cũ. Text đổi whitespace vẫn bị bắt dù normalized hash bằng nhau.
3. Runner train/calibration/test riêng: final test không chọn cấu hình hoặc lambda.
   Đổi final-test label/vector không làm thay candidate hoặc trọng số đã học.
4. Chạy CLI đầy đủ với network bị chặn: mô phỏng32 input/31 văn bản khác nhau,
   tám request record; nhận3 rồi resume đủ8; học prior1536→2D, cập nhật PILOT tám
   vòng và phục vụ tám query deployment riêng, frozen state không đổi.
5. Chạy một cấu hình64D/30/40 trên cache BGE thật với847/211/500 cặp:
   calibration141/211, final326/500=65,2%, đủ lambda cho tám arm. Không grid search.

## Cần cho core, cần để gần paper, và phần mở rộng

| Mục | Quyết định |
| --- | --- |
| Embedding paper, hai thí nghiệm core, baseline công bằng, metric GPT-4 | Ưu tiên tiếp theo sau duyệt phí; luồng nhập vector/fit đã chuẩn bị |
| 11 model và thông tin tác giả còn thiếu | Tiếp tục là giới hạn fidelity; không giả mapping, không chặn mọi việc độc lập |
| 500 preference test | Phân tích riêng, không phải yêu cầu mới cho core và không thay train chính chỉ để khớp số đếm |
| Binary/HybridLLM/RouteLLM, drift/noise, Instructor-XL | Để sau core, khi có mục tiêu kiểm chứng cụ thể |

Main migration giữ1.245 train/313 calibration và cấu hình cũ, không trộn thay
split với thay embedding. Accuracy calibration không phải final-test accuracy.
Study500 dùng dữ liệu từng có trong lịch sử, không phải đúng tập tác giả và
65,2% không chứng minh tái lập65% paper hay cải thiện so với study313 cũ.

Chưa sinh vector OpenAI thật, chưa chạy main experiment với OpenAI/11 arm.
Công cụ submit/giữ approval thuộc [workflow setup riêng](../../docs/openai-embedding-setup.md);
phần này tương thích với `output.jsonl`/`errors.jsonl` của nó và không gọi transport.
Không sửa các file setup/API xuất hiện đồng thời trong workspace.

Gate cuối: **564 test đạt, một skip**. Review độc lập tìm một vấn đề nhãn synthetic
của cost-size; đã tái hiện, sửa và kiểm lại. Bằng chứng nặng nằm ở
`outputs/openai-workflow-20261030/`; gói này chỉ chép báo cáo/aggregate, không có
raw prompt hoặc key. [package-files.json](package-files.json) liệt kê checksum.
