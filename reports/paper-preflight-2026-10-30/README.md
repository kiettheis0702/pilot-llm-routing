# Phần chuẩn bị miễn phí — PILOT / OpenAI

Ngày 30/10/2026. **Không gọi API có phí.** Gói này bổ sung cho nghiên cứu
BGE-M3 tám model; không đổi nhãn hay ghi đè gói 1.098 điều kiện cũ.

- [Báo cáo và lệnh chạy](../../docs/paper-preflight-2026-10-30.md).
- [Protocol đề xuất](../../configs/paper_preflight_protocol.json).
- [Audit dữ liệu/split](input-audit.json), [nguồn đã kiểm và SHA-256](sources.json).
- [96 điều kiện đối chứng ngân sách](budget-comparison.csv),
  [48 cặp chênh lệch](paired-differences.csv).
- [1.098 điều kiện cũ tính lại theo mốc GPT-4](historical-gpt4-relative.csv).
- [Dự toán và cấu hình embedding chưa gửi](embedding-plan-summary.json).
- [Đọc lại độc lập](verification.json), [review và xử lý](independent-review.md),
  [bàn giao QA](qa-handoff.md), [log test cuối](tests-final.txt).

## Kết quả mới

| Việc | Kết quả |
| --- | --- |
| Routing | 36.497 query; 86 nhãn raw, 57 nhãn MMLU; không có nhóm prompt qua hai split |
| Preference | Đề xuất 847 train / 211 calibration / 500 test, đủ tám arm trong mỗi phần |
| Budget controller | 48 cặp × hai cơ chế; không đổi hành động, coverage, reward/cost trong grid đã thử |
| Đánh giá | Cùng mẫu số GPT-4 trên mọi query yêu cầu, kể cả phần chưa phục vụ |
| Embedding | 38.018 văn bản duy nhất, 6.867.235 token, 298 HTTP request đã chuẩn bị |
| Dự toán Standard | 0,1373447 USD; chưa duyệt, chưa gửi; không giả định giảm giá Batch |
| Gate | 517 test đạt, 1 skip; đọc lại 96 điều kiện và khớp 48 trace cũ |

Tại nửa local reference budget, PILOT/BGE đạt 87,18% chất lượng và 46,29%
recorded cost của GPT-4 trên RouterBench; MMLU subset là 82,78% và 47,48%.
Cả hai coverage 100%. Cùng mẫu số không có nghĩa đã khớp protocol paper.

## Các gate khoa học còn mở

1. Chưa đủ bằng chứng phiên bản phía RouterBench cho Claude-v2, Mistral, Mixtral.
   Mã RouterBench gán bảng giá Mistral/Mixtral chung với tên v0.1; đó chưa phải
   xác nhận checkpoint thực sự đã tạo outcome. Chưa promote mapping.
2. Chưa xác minh release/shot, quan hệ 86/64 task, split và stream tác giả.
3. Chưa tìm được implementation PILOT chính thức hoặc đầy đủ hyperparameter
   pretraining trong các nguồn đã kiểm; các cấu hình mới là giả định được công bố.
4. Tập 500 là chia lại corpus đã dùng: 400 từng thuộc train, 100 thuộc holdout.
   Không gọi là historically unseen hoặc đúng 500 cặp tác giả.
5. OpenAI giữ văn bản đầy đủ, trong khi BGE cũ đã truncate 13 preference input;
   so sánh trực tiếp là so recipe, chưa cô lập riêng model embedding.

**Chưa embedding OpenAI, chưa học lại projection/prior/router theo embedding mới,
chưa có kết quả thí nghiệm OpenAI hay 11 model.** Không cần trả phí để giải quyết
các gate về nguồn; việc tìm nguồn công khai đã được làm và giới hạn còn lại được
ghi rõ, thay vì đoán để đánh dấu hoàn tất.

Input request chứa prompt thật chỉ ở thư mục ignored
`outputs/paper-preflight-20261030/embedding-plan/`; không có trong gói này.
Toàn bộ trace và input số của thí nghiệm mới ở thư mục ignored
`outputs/paper-preflight-20261030/budget-comparison/`. Mỗi file có checksum trong
manifest tương ứng. [package-files.json](package-files.json) kiểm toàn vẹn gói
báo cáo này. Các file vẫn ở local, chưa commit/push hay upload.
