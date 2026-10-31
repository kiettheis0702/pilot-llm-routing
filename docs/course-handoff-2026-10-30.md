# Bàn giao bộ bài nộp — 30/10/2026

> **TÀI LIỆU LỊCH SỬ — BGE / mốc 29–30/10/2026.** Các số liệu, trạng thái “hiện tại/current”, nhánh Git và việc còn chờ bên dưới chỉ đúng tại mốc ghi nhận. Không dùng làm kết quả OpenAI hiện hành. Đọc [trạng thái OpenAI](openai-followup-status.md), [bản giao việc hiện hành cho Claude](CLAUDE_UPDATE_OPENAI_2026-10-31.md) và [khôi phục từ checkout sạch](openai-reproduce.md). Giữ nội dung cũ để truy vết, không ghi đè bằng kết quả mới.

> **Đây là snapshot bàn giao local trước upload.** Trạng thái mới: W&B đã Public,
> 1.128 run và 13.314 metric đã kiểm không đăng nhập; GitHub `main` là bản đầy đủ.
> Xem [bàn giao hiện tại cho Claude](CLAUDE_HANDOFF.md) và
> [mapping public](../reports/publication-2026-10-30/README.md). Các mục “chưa có
> project/chưa online/chờ duyệt công khai” phía dưới đã được xử lý. Người dùng
> giao Claude hoàn thiện slide/poster; chưa nộp Moodle. Giữ phần dưới làm lịch sử QA.

**Phần chuẩn bị local đã có đủ để bạn xem và quyết định công khai. Bài chưa nộp
và chưa đủ điều kiện public W&B của môn học.**

## Đã làm

- Đối chiếu đề Moodle với paper cuối cùng trên ACL, chọn phạm vi core và ghi rõ
  các điểm khác paper. Không gọi bản 8 model là tái lập đúng 11 model.
- Giữ nguyên toàn bộ 1.098 điều kiện thí nghiệm cũ; kiểm lại 184 hash của gói.
- Chạy bổ sung chín cấu hình stress cosine và sáu mức cost cho hai population.
- Tạo công cụ tải/kiểm dữ liệu Hugging Face và dựng lại input; dữ liệu dựng lại
  khớp hash đã chốt, không cần chuẩn hóa thủ công.
- Chuẩn bị 1.128 record W&B, mapping experiment–script–ID, tag Lab 1/Lab 2;
  chạy SDK thật offline có plot. Không upload prompt hay gửi API trả phí.
- Dựng [slide 33 trang](../deliverables/midterm/8.pptx) và
  [poster một trang A2 dọc](../deliverables/poster/8.pptx), đều chỉnh sửa được.
  Ghi chú slide 60–89 từ, poster 129 từ; file khoảng 160 KiB và 29 KiB.
- So 128 điểm aggregate với CSV, kiểm bảng ablation và số liệu hai stress tests.
- Viết [phân tích khoa học](course-paper-analysis.md),
  [hướng dẫn chạy](course-reproduce.md) và [hướng dẫn bảo vệ](course-defense-guide-vi.md).

## Bạn cần quyết định / thực hiện

1. Đã xác nhận tài khoản W&B `kiettheis`, team
   `kiettheis-university-of-science-vnu-hcm` qua Chrome. Trang chủ còn 29 ngày
   Academic trial và yêu cầu hoàn tất hồ sơ Academic; chưa có project. Bước kế
   tiếp là xác thực SDK tại máy và duyệt upload đúng payload aggregate đã chuẩn
   bị, đặt project public; không gửi key vào chat, không bật tính năng tính phí.
2. Duyệt công khai repo `kiettheis0702/pilot-llm-routing` và chọn nhánh nộp có đầy
   đủ implementation. Bộ bài nộp nằm trên nhánh `codex/course-submission`.
   Repo public sẽ lộ cả lịch sử Git: cần rà trước khi đổi visibility.
3. Hai thành viên đọc và tập bảo vệ toàn bộ bài; kiểm lại PPTX bằng PowerPoint
   trên máy trình chiếu. Nộp đúng hai file 8.pptx ở hai mục khác nhau trên Moodle.

Khi hoàn tất xác thực SDK và có quyết định công khai, phần còn lại của trợ lý là upload theo
destination đã duyệt, kiểm URL khi không đăng nhập, thay placeholder link trong
mapping/slide/poster, kiểm lại và đẩy bản đã duyệt. Không cần mua thêm API LLM.

## QA và giới hạn

Trạng thái: XONG phần local đã mô tả; CHƯA XONG công khai và nộp môn học.

Gate: 472 tests đạt, 1 skip Windows; chạy tuần tự, RAM trước lượt cuối 14,05 GiB,
không có tác vụ nặng cạnh tranh. Baseline đã ghi nhận trước là 441 đạt, 1 skip.

E2E: tái dựng input thật, supplement CLI, SDK W&B offline có plot, toy pipeline,
RouterBench 64 dòng × 15 baseline methods, dựng và đọc lại hai PPTX. Smoke baseline
dùng context giả lập nên không phải kết quả tái lập paper.

Probe: toàn vẹn package, escape path, file không được liệt kê, input không hợp lệ,
payload bị sửa, online không được duyệt/thiếu entity, resume receipt đã hoàn tất,
cache hỏng bị từ chối trước download, numeric/OOXML readback và final render.

Đường hỏng: dữ liệu thiếu/sai hash và upload thiếu phê duyệt bị từ chối. Lỗi cấu
hình export đã sửa (workspace, font API, Excel precision, runtime module path).
Không sửa dữ liệu cũ để vượt gate.

Người thứ hai: n/a — công cụ local một người; review độc lập ghi riêng trong gói.

Lưu trữ: đã đọc lại notes, bảng, chart cache/workbook và hash từ PPTX cuối cùng;
ảnh render sau re-import trùng byte với bản đã kiểm bố cục.

Chưa chạy: W&B online/public, tải mới từ network ở máy sạch, toàn bộ GPU/grid trên
máy mới, mở bằng PowerPoint thật, Moodle submit. Hosted reconstruction đã dùng
cache có checksum; historical full-grid và replay vẫn là bằng chứng riêng.

Bug đã biết: một lượt regression gặp WinError 5 khi rename progress.tmp trong
test embedding cũ, trước khi tới lỗi disk-full được giả lập. Không có thay đổi
ở runtime/test đó. Test riêng chạy lại đạt, rồi toàn bộ suite ở thư mục mới đạt
472/1; chưa xác định tác nhân gây khóa/quyền file. Giữ log lỗi thay vì che bằng
retry trong code. Vẫn có giới hạn paper-equivalence và public-submission nêu trên.

Quyết định: giữ frozen runtime/protocol, không repin kết quả mới âm thầm; ghi rõ
phạm vi local và các thí nghiệm synthetic/counterfactual; không công khai khi chưa
có destination và quyết định của chủ repo.
