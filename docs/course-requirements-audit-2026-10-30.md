# Đối chiếu yêu cầu môn học — 30/10/2026

**Bản chụp lịch sử ngày 30/10:** các yêu cầu và liên kết bên dưới là nguồn đối
chiếu môn học; cột hiện trạng/việc cần làm mô tả thời điểm đọc, không phải trạng
thái hiện hành. Xem [kết quả OpenAI hiện hành](openai-followup-status.md),
[hướng dẫn chạy lại](followup-reproduce.md) và
[bàn giao slide/poster cho Claude](CLAUDE_UPDATE_OPENAI_2026-10-31.md).
Repository đã công khai và kết quả đã được tích hợp sau bản chụp này; việc đó
không xác nhận đã nộp Moodle hoặc đã kiểm tra khả năng bảo vệ của hai thành viên.

Đọc trực tiếp Moodle bằng phiên Chrome đã đăng nhập theo yêu cầu người dùng,
đề chi tiết gồm sáu trang, bảng đăng ký paper và bảng QA của lớp. Đây là bản
đối chiếu yêu cầu và kế hoạch đề xuất; chưa triển khai hay nộp các phần còn thiếu.
Kế hoạch thí nghiệm cục bộ 8 model đã hoàn tất không đồng nghĩa đã hoàn tất đồ án.

## Nguồn và đăng ký

- [Môn Nhập môn học máy – 24KHMT1](https://courses.ctda.hcmus.edu.vn/course/view.php?id=5001).
- [Tổng quan đề](https://hcmus-cs-ta.github.io/lab-book/intro-ml/project/statement/).
- [Đọc paper và slide](https://hcmus-cs-ta.github.io/lab-book/intro-ml/project/statement/reading).
- [Tái lập thực nghiệm](https://hcmus-cs-ta.github.io/lab-book/intro-ml/project/statement/reproduction).
- [Phản biện lý thuyết](https://hcmus-cs-ta.github.io/lab-book/intro-ml/project/statement/analysis).
- [Thực nghiệm phản biện và poster](https://hcmus-cs-ta.github.io/lab-book/intro-ml/project/statement/empirics).
- [Đăng ký paper](https://docs.google.com/spreadsheets/d/1pTiPmXL-D00KxnlFgVQYZfHK4jpiqX1tVnbwb8YOwms/edit):
  tab Registration, dòng 7 xác nhận nhóm 8, Adaptive LLM Routing under Budget
  Constraints, trạng thái Accepted. Không cần đăng ký lại paper.
- [Đăng ký nhóm](https://docs.google.com/spreadsheets/d/1c2yAwXf9xs4qUYs02H4Gv8nVnH2HZO3UH9ofenSvIEw/edit):
  đã tìm thấy hai thành viên nhóm 8. Không chép danh sách lớp hay thông tin cá nhân
  vào repository.
- [QA](https://docs.google.com/spreadsheets/d/1uv_sP1tX0MqBJBXXc05zt2zJCEUuD1wLO_p_I1msNxU/edit):
  Lab 1 dòng 2 hướng nhóm muốn giảm thực nghiệm về lựa chọn scope trong đề;
  không phải phê duyệt riêng cho việc giảm pool PILOT từ 11 xuống 8 model.

## Hạn và định dạng

Giờ dưới đây là giờ Moodle hiển thị tại thời điểm đọc. Cả ba mục trên tài khoản
đang mở đều ghi chưa có bài nộp; chưa xác minh việc một thành viên khác đã nộp.

| Mục | Hạn | Quy cách | Hiện trạng |
| --- | --- | --- | --- |
| [Link GitHub](https://courses.ctda.hcmus.edu.vn/mod/assign/view.php?id=117389) | 07/11/2026, 23:59 | Một link repository công khai; được tiếp tục commit đến hạn poster | Đã push nhánh codex/data-gate nhưng repository PRIVATE; main vẫn là nhánh mặc định cũ |
| [Midterm Slide](https://courses.ctda.hcmus.edu.vn/mod/assign/view.php?id=117390) | 21/11/2026, 23:59 | 8.pptx, 20–40 slide kể cả phụ lục, dưới 20 MB; mỗi slide có notes tối đa 500 từ | Chưa có deck PPTX trong dự án |
| [Final Poster](https://courses.ctda.hcmus.edu.vn/mod/assign/view.php?id=117391) | 05/12/2026, 23:59 | 8.pptx, đúng một slide A2 dọc | Có PDF A1 ngang cho báo cáo nghiên cứu cục bộ, chưa đúng quy cách bài nộp |

Trong điểm project: deck 45%, poster 50%, chất lượng repository/W&B/định dạng 5%.
Project và vấn đáp chiếm 40–50% điểm môn theo trang Moodle; không nhầm hai mẫu số.
Buổi cuối là workshop poster và vấn đáp ngay tại poster.

## Đối chiếu nội dung bắt buộc

| Yêu cầu | Bằng chứng hiện có | Phần cần làm |
| --- | --- | --- |
| Chọn full hoặc core scope và giải thích | Có 1.098 điều kiện cục bộ, budget/learning sweeps và 5 ablation | Đề xuất core: ít nhất hai thí nghiệm quan trọng, có headline comparison, cộng một phân tích do nhóm thiết kế. Gắn rõ từng thí nghiệm với claim/figure của paper |
| So kết quả nhóm với paper và phân tích độ lệch | Có số liệu cục bộ đầy đủ, đã khai báo thay đổi dữ liệu/model/embedding | Lập đối chiếu số paper–số nhóm, độ lệch tương đối khi thực sự so được; giải thích ảnh hưởng thay thế. Core không tự động làm 8 model tương đương 11 |
| Một project W&B công khai, mọi experiment có config/metrics/plots | Có manifests/traces và replay độc lập tại máy; chưa thấy tích hợp hay link W&B trong repository | Chuẩn bị logging, bảng script→run URL→kết luận, tag riêng lab tái lập và lab phản biện. Nếu nhập kết quả lịch sử phải ghi đúng nguồn/thời gian, không giả là lần chạy mới |
| Dataset tải bằng chương trình từ HF/Kaggle và tái tạo cấu trúc | Đã có scripts/fetch_data.py và revision/hash của ba nguồn HF | Nối và kiểm chứng luồng từ tải dữ liệu đến chuẩn bị/fit/evaluate thay cho phụ thuộc đường dẫn máy cũ; chạy tuần tự theo chính sách RAM |
| Tự triển khai đóng góp cốt lõi từ mô tả paper | Có triển khai và test số học cục bộ | Kiểm lại provenance/citation của phần tái sử dụng; không coi RouterBench benchmark code là implementation PILOT của tác giả |
| So với implementation tác giả nếu paper có cung cấp | Tình trạng nguồn PILOT chính thức vẫn là câu hỏi mở trong CONTEXT.md | Kiểm chứng lại nguồn; nếu có thì chạy và định vị khác biệt, nếu không có thì ghi bằng chứng và giới hạn |
| Deck giải thích đầy đủ paper | Có tài liệu thuật toán và báo cáo kết quả | Tạo 20–40 slide, hình tự vẽ, ít nhất hai hướng tiền nhiệm, novelty, notation, pipeline, thành phần, định lý/giới hạn và độ phức tạp; notes cho mọi slide |
| Citation trong deck | Có các nguồn trong README/docs | IEEE đánh số ở cuối slide có sử dụng nguồn; không có slide tài liệu tham khảo riêng; số thực nghiệm dẫn run W&B |
| Critique có ít nhất 3 điểm mạnh, 2 hạn chế lý thuyết, 2 hướng tương lai | Có ablations, prior holdout, so baseline và chi phí ước tính/thực | Viết lập luận gắn với giả định và dẫn chứng; mỗi hướng tương lai giải quyết hạn chế đã nêu |
| Thực nghiệm chứng minh hạn chế | Có dữ liệu ablation và 12/1.098 lượt vượt ngân sách theo chi phí lịch sử | Chọn giả thuyết có thể bác bỏ và thiết kế đối chứng đúng cho từng hạn chế; chỉ bổ sung run khi bằng chứng hiện tại chưa trả lời được. Không quy lỗi của adapter cục bộ cho paper |
| Poster nối hạn chế lý thuyết với thực nghiệm | PDF cũ chủ yếu tổng hợp nghiên cứu | Dựng một slide A2 dọc cho hai vấn đề nổi bật, có lý thuyết→thí nghiệm→kết luận và URL W&B; các vấn đề dư đưa README |
| Mọi thành viên hiểu và bảo vệ được toàn bộ bài | Chưa kiểm tra khả năng giải thích của nhóm | Nhóm đọc hiểu paper, code, kết quả và luyện vấn đáp. Đề cho phép hỗ trợ AI nhưng trách nhiệm giải thích vẫn thuộc mọi thành viên |

## Đề xuất thứ tự thực hiện

1. Chốt core scope và lập ma trận hai thí nghiệm chính + phân tích bổ sung. Giữ
   nguyên bộ evidence cục bộ hiện có, không đổi nhãn thành tái lập paper nguyên bản.
2. Chuẩn bị repository để công khai: kiểm tra nội dung/lịch sử, chọn nhánh hiển thị
   chính, làm README hướng vào người chấm. Việc đổi PRIVATE→PUBLIC cần quyết định
   rõ của chủ repository; lệnh push trước đây không phải phê duyệt công khai.
3. Chuẩn bị W&B và luồng tải/chuẩn bị dữ liệu có thể chạy lại. Chỉ xuất dữ liệu
   experiment cần thiết, không gửi raw prompt hay thông tin tài khoản. Chốt nơi
   nhận và nội dung trước khi publish.
4. Hoàn thiện so sánh với paper, phân tích độ lệch và deck giữa kỳ đúng định dạng.
5. Phát triển hai luận điểm phản biện, tận dụng run cũ khi phù hợp; bổ sung phép
   kiểm chứng có kiểm soát nếu thiếu. Sau đó dựng poster A2 dọc và câu hỏi vấn đáp.

Chưa có phê duyệt gọi API LLM trả phí, đổi visibility, publish lên W&B hay nộp
bài Moodle trong lượt đọc này. Không cần API trả phí để tiếp tục chuẩn bị các
phần trên. Báo cáo PDF 9 trang vẫn hữu ích làm tài liệu nội bộ, nhưng không thay
thế hai file PPTX mà đề yêu cầu.
