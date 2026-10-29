# Verified tuning handoff

Trạng thái: XONG bước tuning và đóng băng protocol local — chưa xong thí nghiệm đầy đủ.

Gate: `pytest tests -q --basetemp outputs/pytest-tuning-regression-20261029-a` đạt 407 passed, 1 skipped trong 7,51 giây; 17 test tuning mới đều đạt. JUnit ở `outputs/tuning-regression-20261029.xml`. Một test symlink Windows bị bỏ qua như baseline.

E2E: CLI chạy thật 24 cấu hình trên 1.000 câu RouterBench và 398 câu MMLU, CPU một worker. Prior 64 chiều đã kiểm chứng độc lập: 188/313 cặp holdout đúng, đủ tám hệ số hữu hạn; kiểm lại cả hai objective cuối và mọi hướng chiếu.

Probe: Bảy nhóm kiểm chứng trong báo cáo đều đạt: prior; replay toàn bộ 48 candidate; chọn tham số/calibration; mutation held-out; input tuning hỏng; CLI chạy lại; dữ liệu và kết quả gốc bất biến.

Đường hỏng: Gán NaN/chi phí âm vào mọi outcome ngoài tuning và thay text/task ngoài tuning trong RAM vẫn cho 98 file search giống từng byte. NaN trong tuning bị từ chối trước khi tạo output. CLI trỏ vào output đã có báo FileExistsError và không đổi file.

Người thứ hai: n/a — pipeline offline một worker.

Lưu trữ: Đọc lại và replay độc lập 33.552 lượt chọn/cập nhật cùng mọi dự đoán, RNG, trạng thái cuối, điểm candidate và calibration. File nguồn pinned không đổi. Báo cáo liên kết hash chính xác của script, dữ liệu và protocol.

Chưa chạy: Full learning/deployment theo ba seed, 1.098 điều kiện primary/ablation, biểu đồ, báo cáo, poster và kiểm tra tái lập cuối. Các việc này không được tính hoàn tất từ tuning/smoke.

Bug đã biết: Không còn lỗi đã biết trong phạm vi tuning được kiểm tra. Script kiểm chứng prior ban đầu so sai định dạng provenance; đã sửa để đối chiếu cả mô tả và source_urls, rồi chạy lại đạt. Checkpoint/thuật toán huấn luyện không bị sửa.

Quyết định: PILOT alpha=1, LinUCB alpha=5, EpochGreedy window=10 ở cả hai domain; chọn bằng dữ liệu tuning. Seed, grid, ablation, ngân sách và khác biệt paper được đóng băng trước full deployment. Đây là pool local tám model, chưa phải tái lập đầy đủ 11 model.

Tôi sẽ lo: Chạy các phần còn thiếu theo protocol đã đóng băng; kiểm kê đủ seed/method/budget và kiểm chứng kết quả trước khi làm hình/báo cáo.

Cần bạn: Chưa cần API/key hay chi phí. Ngưỡng RAM 8 GiB và chạy tuần tự giữ nguyên; mọi thao tác hệ thống tiếp theo dùng lệnh theo yêu cầu của bạn.

Evidence: [tuning-2026-10-29.json](../reports/tuning-2026-10-29.json).
Frozen configuration: [frozen_experiment_protocol.json](../configs/frozen_experiment_protocol.json).
Protocol: [experiment-protocol.md](experiment-protocol.md).

The earlier tuning/ablation progress notes are dated preparation history; this
handoff supersedes their RAM and verification blockers. Their original reports
are retained rather than rewritten to imply they were verified at the time.

## Next-stage preflight

The subsequent `identity-d64-s7` fit attempt was refused by `require_resources`
at 7.04 GiB available RAM before training started. It did not create its output
directory. The seed-42 prior and both completed tuning runs remain verified.
Recheck resources before starting this same pending fit; do not rerun completed
fits or tuning stages. This later RAM change does not invalidate their results.
