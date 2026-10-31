# Chạy embedding nhanh bằng API trực tiếp

Ngày 30/10/2026, người dùng đã duyệt chi phí để ưu tiên hoàn tất trong khoảng
10 tiếng và xác nhận phạm vi **chỉ tạo embedding OpenAI**. Đây là phê duyệt mới,
thay cho yêu cầu trước đó chỉ kiểm tra key local. Không mở rộng sang huấn luyện,
thay model hoặc xử lý lại văn bản đã thành công.

Dùng cùng `text-embedding-3-small`, mặc định1536 chiều, nguyên văn38018 văn bản
khác nhau/38055 query ID. Chia437 request, tối đa128 input/16000 token mỗi request.
Tổng6867235 token; giá Standard0,02 USD/triệu token → dự kiến0,1373447 USD chưa
gồm thuế. Trần ước tính local cho run là0,15 USD; không phải hard limit billing.

Nguồn: [model và giá](https://developers.openai.com/api/docs/models/text-embedding-3-small),
[schema embeddings](https://developers.openai.com/api/reference/python/resources/embeddings/methods/create),
[quota và response headers](https://developers.openai.com/api/docs/guides/rate-limits).

Lần chạy đã được duyệt nằm ở `outputs/openai-standard-20261030/run-001`.
Đọc `state.json` để biết tiến độ; đây là file local, không chứa API key.
Không khởi chạy thêm lệnh gửi hoặc dùng `--resume` khi process gốc còn hoạt động.
OS lock chặn hai process cùng quản lý một run; receipt chặn dùng lại approval
ở thư mục chạy khác. SDK retries bằng0, không có paid fallback.

Mỗi intent được ghi trước POST, request ID được ghi khi response headers tới,
vector response được lưu ngay và kiểm trước khi chuyển sang request tiếp theo.
Nếu đã có response bền vững nhưng state ghi dở, resume đọc lại response đó.
Nếu mất response và chưa biết kết quả, dừng; không gọi lại embedding để lấy lại.
API embeddings đồng bộ không cung cấp endpoint tải lại một response đã thất lạc.

Pacing dùng tối đa80% giới hạn TPM/RPM nhận từ provider; mặc định bảo thủ40000TPM
trước response đầu. Request đầu cho biết tài khoản5000000TPM/10000RPM. Đây là
quan sát của lần chạy, không phải cam kết SLA hoặc lý do tăng worker song song.
Chỉ dùng một process, một request tại một thời điểm, một BLAS thread.

Sau đủ437 response, công cụ tạo `output.jsonl` tương thích importer hiện có.
Import offline ra cache routing/preference riêng, rồi đối chiếu từng vector với
response provider đã lưu. `verify_final.py` chỉ kiểm file, không gọi API.
Không gọi toàn bộ nghiên cứu xong chỉ vì embedding đã xong.

Phê duyệt và dữ liệu đều nằm trong `outputs/` bị Git ignore. Hướng dẫn Batch cũ
vẫn là tài liệu của phương án cũ; Batch approval chưa được dùng để gửi job.
