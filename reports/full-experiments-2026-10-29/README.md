# Kết quả thí nghiệm PILOT cục bộ

Đã chạy đủ **1.098 cấu hình đánh giá**: 858 lượt chính và 240 lượt ablation,
trên 8 model, 3 seed (7/42/2026), RouterBench và tập con MMLU. Cả 108 nhóm
phương pháp/biến thể đều đã được tính lại độc lập từ dữ liệu và trace.

- [Báo cáo nghiên cứu, 9 trang](deliverables/research-report.pdf)
- [Poster A1 ngang](deliverables/research-poster.pdf)
- [Số liệu từng seed](aggregate/per-seed.csv)
- [Trung bình và khoảng min/max của 3 seed](aggregate/mean-min-max.csv)
- [Chênh lệch từng cặp so với cấu hình đối chứng](aggregate/paired-deltas.csv)
- [Chỉ số tại các mốc dữ liệu học](aggregate/learning-prefixes.csv)
- [Manifest tổng hợp và danh sách run](aggregate/manifest.json)
- [Giá trị số và checksum của 6 biểu đồ](figures/figure-data.json)
- [Kiểm tra biểu đồ/PDF độc lập](deliverable-checks.json)

Trong bảng chênh lệch, các ablation so với PILOT đầy đủ; các dòng `learning-fraction`
so từng phương pháp với chính phương pháp đó khi dùng toàn bộ dữ liệu học.

Ở ngân sách bằng 0,5 lần mốc tham chiếu, utility trung bình của PILOT là
**0,6764** trên RouterBench và **0,6675** trên MMLU, đều phục vụ đủ truy vấn.
LinUCB đạt **0,7056** trên RouterBench; ExploreOnly đạt **0,6899** trên MMLU.
PILOT không đứng đầu ở cấu hình này. Các bảng và hình giữ đủ ngân sách và
ablation, không chỉ chọn kết quả thuận lợi. Utility là tổng reward chia tổng
truy vấn yêu cầu; truy vấn không được phục vụ đóng góp 0.

Có **12/1.098 lượt** mà chi phí lịch sử ghi trong benchmark vượt ngân sách
ước tính (mức vượt lớn nhất 0,127955416 đơn vị benchmark). Bộ chọn chỉ bảo đảm
chi phí **ước tính** nằm trong ngân sách. Đây là phép tính trên dữ liệu lưu sẵn,
không phải tiền API bị trừ trong lần chạy này; không có API trả phí nào được gọi.

Phạm vi hoàn tất là thí nghiệm cục bộ đã khai báo trước. Chưa thể gọi đây là
tái lập nguyên bản paper: còn ba mapping Claude-v2/Mistral/Mixtral chưa chứng minh,
khác biệt nguồn/shot RouterBench, 86 nhãn task so với 64 trong paper, nguồn và
holdout Arena, embedding BGE-M3 thay thế cùng một số quy ước tính toán. MMLU nằm
trong RouterBench; 3 seed chỉ cho khoảng biến thiên mô tả, không phải khoảng tin cậy.

`verification/` chứa 108 báo cáo đối soát độc lập. `evidence-scripts/` giữ nguyên
source đã chạy; `tools/research/` ở gốc dự án là bản đóng gói có cùng toán học,
khác cách tìm thư mục chạy, được ghi rõ trong `source-packaging.json`.
Raw prompt, embedding, prior và trace lớn vẫn ở thư mục `outputs/` bị Git bỏ qua.
Muốn chuyển máy để tái chạy cần mang cả các input/artifact có hash trong protocol;
chỉ clone repo không mang theo những dữ liệu này.

Lệnh chạy: [hướng dẫn thí nghiệm](../../docs/full-experiments.md).
Trạng thái cuối và giới hạn tái lập: [bàn giao](../../docs/final-research-handoff-2026-10-29.md).
