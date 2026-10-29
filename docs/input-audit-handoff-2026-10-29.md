# Input audit handoff — 2026-10-29

Trạng thái: XONG phần xuất input và kiểm tra độ dài; CHƯA XONG toàn bộ nghiên cứu.

Gate: `pytest -q --tb=short --basetemp outputs/pytest-research-20261029-a
--junitxml outputs/research-input-audit-20261029/pytest.xml` đạt 348 tests, bỏ qua
1 test symlink do Windows; tăng 10 tests so với mốc 338. OpenSpec strict validation
và `git diff --check` đạt. Các bước chạy tuần tự, một worker.

E2E: CLI `scripts.input_audit` chạy hết 38.055 input thật trên nhánh
`codex/data-gate`, thay đổi local chưa commit. Artifact `kind=input-length-audit`
và cap 2.048 xuất hiện trong manifest mới. Tokenizer chạy từ cache, không nạp
PyTorch, không gọi API.

Probe: 6 probe đạt: đọc lại thứ tự/identity toàn bộ input; checksum các file và
manifest; split không thay đổi; đối chiếu tổng token của smoke 128 câu cũ; đối
chiếu AutoTokenizer độc lập trên 36 câu dài/lấy mẫu; chạy CLI lần hai bị từ chối
ghi đè. Test riêng kiểm tra prompt/hash sai, phase thiếu, ID trùng, holdout dài
không đổi cap và ghi manifest dở dang không xuất hiện dưới tên hoàn tất.

Đường hỏng: CLI dùng output đã tồn tại trả FileExistsError. Lần thử GPU 16 câu
dừng tại RAM guard 7,95 GiB trước khi nạp encoder; progress có 0 dòng hoàn tất và
không có manifest cache hoàn tất. Test mô phỏng disk-full giữ artifact chưa hoàn
tất. Lỗi thư mục tạm Windows được xử lý bằng basetemp mới trong workspace; không
đổi quyền thư mục hệ thống.

Người thứ hai: n/a — luồng CLI một người, không có dịch vụ nhiều client.

Lưu trữ: đọc lại input, lengths JSONL và manifest; xác nhận SHA-256, thứ tự, raw
text hash và split hash. Báo cáo tổng hợp không chứa prompt hay response.

Chưa chạy: inference 2.048 token và full GPU caches vì thiếu RAM sau khi import
thư viện. Lần resume sau khi RAM trống tăng lên 8,82 GiB cũng dừng ở kiểm tra
thứ hai (còn 7,40 GiB), không nạp encoder và vẫn giữ 0 dòng hoàn tất.
Prior thật, tuning, deployment so sánh và báo cáo kết quả nghiên cứu
chưa có; không được diễn giải audit này là các thí nghiệm đó.

Bug đã biết: không có lỗi chức năng còn biết trong phần audit đã kiểm. Default
pytest temp của máy bị từ chối quyền truy cập; lệnh kiểm tra đã dùng đường dẫn
thay thế có ghi trong tài liệu.

Quyết định: dùng quy tắc 99% riêng cho routing learning+tuning và preference
training, đã ghi trong design trước khi đếm. Không dùng deployment/holdout để
chọn cap. BGE-M3 và nhóm 8 model là cấu hình phát triển có ghi rõ giới hạn.

Tôi sẽ lo: hoàn tất các bước còn lại trong checklist OpenSpec. Có thể tiếp tục
đối chiếu token/chi phí và nguồn model khi GPU còn chờ RAM. Khi đủ RAM, tiếp tục
feasibility bằng `--resume`, rồi mới tạo đầy đủ caches.

Cần bạn: hiện không có quyết định mới cần xác nhận; không gọi API trả phí.

Evidence: [aggregate report](../reports/input-audit-2026-10-29.json),
[research checklist](../openspec/changes/offline-research-experiments/tasks.md),
[source/cost investigation](source-cost-evidence-2026-10-29.md).
