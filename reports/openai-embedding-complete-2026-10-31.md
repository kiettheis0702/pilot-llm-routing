# Embedding OpenAI đã hoàn tất — 31/10/2026

Đã tạo embedding thật bằng `text-embedding-3-small`, mặc định1.536 chiều, cho
toàn bộ snapshot hiện tại. Chỉ tạo/kiểm embedding, không huấn luyện router.

| Kết quả | Giá trị |
| --- | ---: |
| Văn bản duy nhất đã gửi | 38.018 |
| Dòng nguồn được ánh xạ đầy đủ | 38.055 |
| Request thành công | 437 / 437 |
| Token do OpenAI trả trong usage | 6.867.235 |
| Paid retry / fallback | 0 / 0 |
| Chi phí tính theo giá Standard | 0,1373447 USD, chưa gồm thuế |
| Gọi API đến response cuối | Khoảng10 phút24 giây |
| Gọi API và xuất JSONL | 10 phút48,6 giây |
| Từ bắt đầu chạy đến kiểm chứng mọi vector | **12 phút28,5 giây** |

Giờ Việt Nam/Thái Lan (UTC+7): bắt đầu23:49:46 ngày30/10; xuất xong00:00:35
ngày31/10; kiểm chứng xong00:02:15 ngày31/10/2026. Khóa và request ID không được
in ra báo cáo; key vẫn ở file local bị Git ignore.

Chi phí trên là phép tính `6.867.235 × 0,02 / 1.000.000`, dựa trên usage thật và
[giá OpenAI Standard](https://developers.openai.com/api/docs/models/text-embedding-3-small).
Chưa đọc hóa đơn/ledger billing của nhà cung cấp; không gọi con số này là số tiền
đã đối chiếu hóa đơn. Không dùng Batch, không tạo job khác hoặc gọi lại request.

## Cache và bằng chứng local

- Routing: `outputs/openai-standard-20261030/imported/caches/source-0` — **36.497 × 1.536 float32**.
- Preference: `outputs/openai-standard-20261030/imported/caches/source-1` — **1.558 × 1.536 float32**.
- Manifest request/input: `outputs/openai-standard-20261030/plan-16k`.
- Response gốc, request IDs, state: `outputs/openai-standard-20261030/run-001`.
- Receipt import: `outputs/openai-standard-20261030/imported/receipt.json`.
- Kiểm chứng độc lập: `outputs/openai-standard-20261030/verification.json` và `verify_final.py`.
- User approval một lần: `outputs/openai-standard-20261030/approval.json`.
- Build/code hashes: `outputs/openai-standard-20261030/execution-evidence.json`.

Tất cả cache/response/payload/approval thật ở trong `outputs/`, bị Git ignore.
Snapshot là tám model đã được chấp nhận của dự án; chuyển embedding không tự
giải quyết ba alias còn thiếu hoặc biến kết quả thành tái lập11 model của tác giả.

## Đối chiếu đã thực hiện

Importer nhận đủ437 custom ID, không thiếu/lỗi record, kiểm index, model,
dimension, finite/unit vectors và exact source hash. Cache được xuất với
`synthetic: false` và437 provider request ID riêng biệt.

Script độc lập đọc lại hai cache, kiểm query ID/thứ tự/prompt hash/source checksum,
provenance, dtype, dimension, finite values và norm. Sau đó đọc lại437 response
gốc và **so chính xác từng vector float32 cho toàn bộ38.055 dòng nguồn**, kể cả
các văn bản dùng lại. Tất cả khớp. Usage của response/state/receipt cùng6.867.235.

Routing fingerprint: `59f9090db4e92fcc0eaa545fe925d8ec43156073da2b399c779f8f301e3f46fc`.

Preference fingerprint: `474857af6dcf4b259ea1a8aeb0758bae67fc9e6023e1c43dadc2415bd4c4c713`.

## Bàn giao QA

Trạng thái: **XONG — embedding thật đã chạy và kiểm chứng đầy đủ**.

- Gate: focused suite cuối `tests/test_openai_embeddings_standard.py tests/test_openai_embedding_setup.py tests/test_openai_workflow.py tests/test_paper_preflight.py` → **91 passed**,9,67 giây; một pytest process, một BLAS thread. Whitespace diff sạch.
- E2E: key thật → SDK chính thức →437 Standard request → durable responses → output JSONL → offline import → hai DiskEmbeddingCache → so khớp mọi vector. Build kind `standard-openai-embeddings`; code hashes đã lưu.
- Probe:17 test của Standard runner đạt, gồm approval/cost scope, SDK streaming serialization, completed/orphan resume, ambiguity,429 không retry, immutable payload, rate pacing, deadline, Unicode JSONL và hai lần competing-handle lock; các test synthetic chặn socket thật. Luồng provider thật437/437 đạt.
- Đường hỏng: review tìm concurrent-resume replay, đã thêm OS lock và reviewer xác nhận. Khởi động đầu tiên gặp Unicode separators trong prompt; dừng trước khi tạo run/transport, sửa parser, test RED→GREEN và kiểm toàn bộ payload thật offline rồi mới gửi. Không dùng request trả phí để thử lỗi hoặc lấy lại kết quả.
- Người thứ hai: n/a — CLI một người; hai trường hợp lock cạnh tranh được kiểm, không chạy hai job thật.
- Lưu trữ: đọc lại toàn bộ manifest/hash/request ID/usage và từng vector từ đĩa. Cache có thể dùng lại cho mọi seed/experiment; không cần embedding lại corpus này.
- Chưa chạy: huấn luyện, tuning, thí nghiệm core và đọc billing invoice — ngoài phạm vi người dùng đã chốt “Chỉ tạo embedding OpenAI”.
- Bug đã biết: không còn bug material trong phạm vi này; quota/SLA tương lai và các giới hạn khoa học vẫn được công bố riêng.
- Quyết định: người dùng duyệt chi phí để đạt khoảng10 tiếng; chọn Standard đồng bộ, cùng model, trần dự toán local0,15 USD, corpus hiện tại một lần, không tăng worker hoặc model fallback.
- Tôi sẽ lo: không còn việc cần làm trong phạm vi embedding đã giao.
- Cần bạn: không cần thao tác thêm cho lần embedding này.

Không commit/push, không upload W&B, không sửa kết quả BGE hoặc slide/poster.
