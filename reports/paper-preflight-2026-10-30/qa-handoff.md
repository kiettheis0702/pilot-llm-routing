# Bàn giao QA phần chuẩn bị miễn phí

Trạng thái: **XONG phần chuẩn bị/đối chứng miễn phí đã mô tả**; chưa có run OpenAI/11 model.

Gate: pytest toàn dự án chạy tuần tự, một BLAS thread → **517 passed, 1 skipped**;
baseline trước thay đổi là 490 passed, 1 skipped. RAM trước lượt cuối 10,63 GiB,
không có job nặng cạnh tranh. `git diff --check` cho hai module runtime đạt.

E2E: audit dữ liệu thật và xuất batch embedding thật chạy hai lần vào hai folder
mới, giống byte; đối chứng lịch sử chạy 96 điều kiện và lặp lại 96 điều kiện,
48 guarded khớp mọi arm trong trace cũ. Mẫu số GPT-4 tính lại cho 1.098 điều kiện.

Probe: 12 kiểm tra đọc lại độc lập đạt; ba probe tiến trình Windows đạt; một
lượt chuẩn bị toàn corpus với socket bị chặn đạt, không thử kết nối mạng.

Đường hỏng: CLI từ chối ghi đè audit/embedding/comparison; malformed JSON bị
từ chối trước tạo output; test bắt duplicate identity, input quá dài, giới hạn
token/request, nhóm prompt không thể chia đủ và số liệu kế toán không hợp lệ.

Người thứ hai: n/a — CLI local một người; review độc lập đã hoàn tất riêng.

Lưu trữ: đọc lại JSONL bằng consumer riêng, phục hồi chính xác mọi văn bản từ
request/index; kiểm hash, token, 96 trace bằng phép tính độc lập và 48 trace cũ.

Chưa chạy: submit embedding/OpenAI response/import vector, học lại với vector
OpenAI, 11-model training, benchmark API thật, cloud GPU hoặc W&B upload.
Các thao tác có phí chưa được duyệt; không dùng request thật làm smoke test.

Bug đã biết trong phần bàn giao: không có bug runtime chưa xử lý được phát hiện.
Thiếu bằng chứng alias/release/hyperparameter được ghi thành gate khoa học.

Quyết định: giữ mapping chưa xác minh ngoài tập train; chia 500 là local resplit;
giữ kết quả 48 cặp bằng nhau; báo riêng ảnh hưởng truncation ở recipe BGE cũ;
không commit/push và không sửa presentation đang làm dở.

Tôi sẽ lo: phần miễn phí trong phạm vi trên đã hoàn tất, không có job nền tiếp tục.

Cần bạn: chưa cần cung cấp key hoặc duyệt phí. Chỉ xin duyệt một lượt có phí sau
khi chốt pool/input cuối, với số request và dự toán đã cập nhật tương ứng.

Lỗi thực thi trong quá trình QA đã xử lý: lần baseline đầu trỏ pytest basetemp
vào parent chưa tồn tại; sửa cách gọi rồi baseline đạt. Consumer JSONL độc lập
ban đầu dùng `str.splitlines()` nên tách nhầm U+2028 trong chuỗi JSON hợp lệ;
đã sửa consumer đọc dòng file thật và kiểm lại, không sửa prompt để né lỗi.
