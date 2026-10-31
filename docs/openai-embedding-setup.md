# Thiết lập embedding OpenAI — 30/10/2026

Cập nhật31/10/2026: người dùng đã duyệt chuyển sang API Standard để ưu tiên tốc
độ; **embedding thật đã hoàn tất và kiểm chứng**, không gửi Batch. Xem
[báo cáo hoàn tất và đường dẫn hai cache](../reports/openai-embedding-complete-2026-10-31.md).
Các mục Batch bên dưới ghi lại phương án setup ban đầu.

Phần local đã dùng SDK OpenAI chính thức, khóa phiên bản **2.54.0** trong
`uv.lock`. Chưa gửi request tới OpenAI và chưa kiểm tra quyền tài khoản/billing.

## Dán key

Mở `.env.openai` ở thư mục gốc dự án, dán key ngay sau dấu `=` rồi lưu:

```dotenv
OPENAI_API_KEY=
```

File này đã bị Git ignore. Không dán key vào chat, file Python hoặc file example.
Key trong file local được ưu tiên; file trống không dùng ngầm key của tài khoản
khác trong biến môi trường. Chỉ kiểm tra định dạng local, không gọi API để thử key.
Nhập key không khởi động job, không tạo lịch chạy và không duyệt chi phí.

Kiểm tra miễn phí trong PowerShell/CMD tại thư mục dự án:

```powershell
.venv/Scripts/python.exe -m tools.openai_embeddings doctor
```

Kết quả chỉ báo `key_configured: true/false`, không in key. Nếu tạo lại môi
trường, dùng `uv sync --extra cost --extra tracking --extra openai`.

## Cấu hình đã chuẩn bị

| Mục | Giá trị |
| --- | --- |
| Model | `text-embedding-3-small` |
| Chiều vector | 1.536; không gửi tham số `dimensions` |
| Dữ liệu hiện tại | 38.055 query ID → 38.018 văn bản duy nhất |
| Token | 6.867.235, đếm local bằng `cl100k_base` |
| Cách gửi | Một Batch job gồm 298 embedding request |
| Prompt | Nguyên văn; không thêm instruction, không cắt input |
| Ước tính Batch | **0,06867235 USD**, chưa gồm thuế |
| Ước tính Standard bảo thủ | **0,1373447 USD**, không có fallback tự động |
| Thời gian dịch vụ Batch | Trong 24 giờ; phụ thuộc tài khoản và hàng đợi |

Nguồn: [giá model](https://developers.openai.com/api/docs/models/text-embedding-3-small),
[Batch API: giảm 50%, cửa sổ 24 giờ và giới hạn 50.000 embedding input](https://developers.openai.com/api/docs/guides/batch),
[schema embedding](https://developers.openai.com/api/reference/python/resources/embeddings/methods/create).
Đây là ước tính theo giá đã đọc ngày 30/10/2026, không phải phí thực tế hoặc
hard spending limit phía OpenAI. Chưa xác minh quyền Batch/quota của tài khoản.

Snapshot ở `outputs/paper-preflight-20261030/embedding-plan`. Bộ hiện tại dựa
trên tám model đã được chấp nhận. Thêm prompt preference mới cần dự toán mới;
không dùng việc thiết lập API để tự giải quyết các gate khoa học còn thiếu.
Giữ nguyên kết quả BGE và học lại projection/prior/router sau khi đổi embedding.

## Chốt duyệt trước khi gửi

Đã tạo `outputs/openai-embedding/approval.json` với **`approved: false`**.
Không sửa thành true khi người dùng chỉ thông báo đã nhập key. Trước lần chạy,
trình bày model, snapshot, số request/job, chi phí và phạm vi cho người dùng;
chỉ ghi nhận phê duyệt sau khi họ đồng ý rõ ràng cho chính lần đó.

Approval gắn với SHA256 manifest/payload, model, số văn bản, token và request,
một Batch job, người duyệt và mức trần ước tính Standard bảo thủ. Đây là chốt
local theo quy tắc của người dùng; không thay thế giới hạn billing phía provider.
Công cụ kiểm lại toàn bộ hash, token, giới hạn Batch và đóng băng payload trong
bộ nhớ trước khi upload. Một approval đã dùng không gửi lại được, kể cả khi
đổi tên file approval hoặc dùng thư mục chạy khác. Không retry/fallback trả phí.

Lệnh dành cho **sau khi đã duyệt**, không chạy trong quá trình setup:

```powershell
.venv/Scripts/python.exe -m tools.openai_embeddings submit --approval outputs/openai-embedding/approval.json --run-dir outputs/openai-embedding/run-001
```

Intent được ghi trước khi upload; receipt dùng một lần và state không chứa key.
File ID, Batch ID và request ID được lưu ngay khi nhận. SDK chỉ dùng endpoint
`https://api.openai.com/v1`; bỏ qua `OPENAI_BASE_URL` và proxy môi trường của các
công cụ khác, đặt `max_retries=0`.

## Theo dõi và khôi phục, không tạo generation mới

```powershell
.venv/Scripts/python.exe -m tools.openai_embeddings status --run-dir outputs/openai-embedding/run-001
.venv/Scripts/python.exe -m tools.openai_embeddings recover --run-dir outputs/openai-embedding/run-001
.venv/Scripts/python.exe -m tools.openai_embeddings download --run-dir outputs/openai-embedding/run-001
```

`recover` dùng Batch ID đã lưu hoặc tìm job bằng metadata riêng + file ID + hash
plan. Không gọi `create` để tìm lại kết quả. Nếu chưa tìm được duy nhất một job,
giữ state và kiểm tra dashboard; không mặc định tạo job thay thế.

`download` stream dữ liệu xuống đĩa, lưu checksum và cả output/error file của
job hoàn thành một phần hoặc hết hạn. Download gián đoạn chỉ để lại `.part`;
có thể tải lại file đó mà không tạo job. Job `expired/failed` không được gọi là
embedding hoàn tất. Các lệnh này chỉ đọc kết quả đã tạo, không gửi embedding mới.
Chuyển output thành cache và huấn luyện là bước tiếp theo; công cụ này không
tự chạy huấn luyện hoặc sửa cache cũ.

## Kiểm chứng

Xem `reports/openai-embedding-setup-2026-10-30.md`. Các lỗi provider được ép tại
ranh giới transport của SDK bằng dữ liệu synthetic; chưa xác minh API thật vì
chưa có phê duyệt chạy trả phí.
