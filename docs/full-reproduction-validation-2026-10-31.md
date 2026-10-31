# Kiểm chứng chạy lại đầy đủ — 31/10/2026

**Trạng thái: regression đạt; các lượt chạy khoa học đầy đủ chưa được chứng
nhận hoàn thành tại mốc này.** Không dùng số test thay cho bằng chứng full grid.

## Phần đã hoàn thành

- [Regression sau review](https://github.com/kiettheis0702/pilot-llm-routing/actions/runs/41209176579):
  **729 passed, 37,06 giây**, trên standard Windows2025 runner mới, Python3.11.18,
  cài từ `uv.lock`, một worker/BLAS thread. Code:
  `a13a0d0a3ec7301ac13bd4e0a3b84d118a925c0b`.
- 15 probe mới: tám trường hợp kiểm full inventory và bảy trường hợp bảo toàn
  đầu vào/ép CPU. Đã quan sát RED trước bản sửa, rồi GREEN cùng toàn bộ suite.
  Probe CPU gọi thật một Python child qua đúng hàm PowerShell; kiểm cả child
  thành công và lỗi, cùng việc phục hồi lựa chọn CUDA của caller.
- Guard mới kiểm danh sách và checksum trước khi chuyển thư mục core. Kiểm
  đọc trên checkout ổ D chấp nhận117 file core đã restore; kiểm đọc trên
  workspace khoa học hiện có từ chối trước mọi thao tác ghi/di chuyển.
- Đọc lại đủ16 package manifest: báo cáo khoa học gốc còn nguyên. Không đổi
  producer đóng băng, ZIP139 file hoặc tag của hai lần bàn giao trước.
- Một lượt review độc lập đã phát hiện và xử lý ba điểm: chặn di chuyển
  kết quả cũ, ép MF CPU trước fit, thu hẹp câu kết luận embedder/seed trong
  tài liệu tiếng Việt. Không có lượt review lặp để thay thế các probe sửa lỗi.

## Phần đang chạy và phần chưa chạy

[Lượt full CPU](https://github.com/kiettheis0702/pilot-llm-routing/actions/runs/41206426268)
chạy code `dbd52c82c86b8016fa3e413d5ba38a00f49d42ed`. Gate của lượt này đạt
722 test; các guard sau review được kiểm riêng ở commit nêu trên. Native
scientific producers không đổi giữa hai mốc. Các scope chạy tuần tự trên
những runner riêng, không dùng RAM máy cá nhân.

| Phần | Yêu cầu để chứng nhận | Trạng thái tại mốc ghi này |
|---|---|---|
| Core |3 prior fit mới,2 tuning mới,858 điều kiện,78 reader, so bảng công bố |Đang chạy; chưa đọc summary hoàn thành |
| Followups |240 ablation,240 budget,12 noise,231 binary, đủ reader và so bảng |Chờ scope trước; chưa chứng nhận |
| MF CPU |Môi trường neural mới,3 fit+mutation, NumPy reader, ghi chênh lệch với CUDA |Chờ scope trước; không gọi là CUDA replay |
| GPU local |Môi trường CUDA mới, MF, Instructor encode/full grid, hai Hybrid fit, task shift và reader |Đầu vào/model/source đã chuẩn bị; còn chờ RAM>=8GiB trước cài/chạy |

Checkout GPU là clone riêng trên ổ D; ZIP được tải từ release công khai,
kiểm đúng SHA256 rồi restore139 file.16 file model và34 file source công khai
đã kiểm hash. Những việc này không chứng minh đã cài xong môi trường CUDA,
đã encode hay hoàn thành fit. Không dùng probe/bản ghi máy cũ để giả một probe mới.

Các lần kiểm bounded đã hoàn thành trước đó vẫn ở
[validation714 test](followup-reproduction-validation-2026-10-31.md); không
đổi nhãn chúng thành full-grid. Giá trị khoa học và số run W&B được giữ nguyên.
Nếu lượt chạy mới khác bảng công bố, phải giữ cả sai khác để điều tra.

## Tài liệu và bài trình bày

[Hướng dẫn đọc toàn dự án](project-reading-guide-vi.md) dùng OpenAI làm bản
hiện hành. Bản giải thích BGE cũ trên máy cá nhân đã có nhãn lịch sử và link
sang bản mới. README/hướng dẫn bảo vệ không còn gọi deck hiện hành là22 slide.

Deck24 slide, poster A2 và PDF preview được đưa vào vị trí `midterm/` và
`poster/` bằng bản sao nguyên byte từ lượt chỉnh của Claude; bộ33 slide ở
`deliverables/` giữ nguyên. [Nhật ký gốc](presentation-update-2026-10-31.md)
chứa hash và QA của lượt tạo file; những builder local mô tả trong nhật ký
không được thêm vào revision này.

Codex kiểm cấu trúc, notes và registry: deck notes275–497 từ, poster455 từ,
42+5 link run hợp lệ. Đã xem contact sheet toàn24 trang, poster và riêng các
slide15/18/20/21/22 ở cỡ đọc được. Ba câu ở slide22 cần Claude thu hẹp theo
[file bàn giao cuối](CLAUDE_FINAL_REVIEW_2026-10-31.md); Codex không sửa PPTX/PDF.
Việc mở trên máy khác, quét QR bằng điện thoại và khả năng giải thích của hai
thành viên chưa được xác nhận bởi các kiểm tra này.

Moodle không cần cho luồng chạy code. Phiên đăng nhập đã hết; sau khi người
dùng hỏi mục đích kiểm tra, khâu xác nhận bài nộp được để lại cho bước nộp
cuối. Không suy ra hiện trạng nộp bài từ phiên đã hết, không có bài nộp mới.
