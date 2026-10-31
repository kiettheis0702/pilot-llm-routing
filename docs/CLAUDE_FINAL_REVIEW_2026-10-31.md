# Claude — rà cuối bản OpenAI 24 slide, không áp lại map cũ

Đầu vào đã được Claude cập nhật; Codex chỉ kiểm đọc và ghi yêu cầu ở đây.
Giữ nguyên số liệu khoa học, biểu đồ có thể chỉnh sửa và thiết kế hiện hành.

- Deck `midterm/8.pptx`: SHA-256
  `5002638b52f6b18c8840fa971597a4e7417708d2269c6035cb2d37a4b68276ab`.
- Poster `poster/8.pptx`: SHA-256
  `051348291acac416d0e14a343726e2d4a7c5e3f1ba31201b19ff5cb14f01b8cf`.
- [Map gốc và tình trạng đã áp dụng](CLAUDE_UPDATE_OPENAI_2026-10-31.md).
- [Nhật ký của lượt sửa slide](presentation-update-2026-10-31.md).

## Chỉnh cách diễn đạt ở slide 22

Ở bảng “Why our numbers differ”, thu hẹp hai nhãn kết luận để đúng bằng chứng:

1. `embedder ruled out; rest untestable` → `gap persists with OpenAI; other
   causes unresolved`. Việc đổi sang embedder paper vẫn còn gap bác bỏ giả
   thuyết “chỉ do BGE” trong phép đối chiếu này, không loại trừ mọi tương tác
   embedding, tuning, split hoặc model pool.
2. `Seed variation — ruled out` → `gap persists across 3 seeds`. Khoảng min/max
   của ba seed không phải khoảng tin cậy và không loại trừ mọi seed tương lai.
3. Với câu dưới bảng về libraries/hardware, ghi rõ independent replay chứng
   minh trace của cấu hình đã chạy; không suy từ đó rằng mọi môi trường đều
   cho checkpoint/metric byte-identical. Chỉ bổ sung claim checkout mới theo
   bản xác minh đã hoàn thành, không theo tên job hoặc trạng thái đang chạy.

Đối chiếu notes của slide 22, thay câu tương ứng thay vì chỉ đổi nhãn. Notes
hiện có slide tới 497 từ; luôn giữ tối đa 500 từ. Không đổi số hay dấu của gap.

## Chỉ cập nhật sau khi có bằng chứng hoàn thành

Phần tái lập dùng đúng tài liệu validation mới nhất được dẫn trong README.
729 test phần mềm đã qua trong CI tại commit `6a58f962`; full core/followups/MF
là các stage riêng. Full core858 và followups723 điều kiện khớp bảng công bố.
MF CPU đạt338/338/340, còn CUDA tái chạy đạt338/340/343 đúng bản gốc; đây là
bằng chứng thực tế cần giữ giới hạn về libraries/hardware ở slide22.
Instructor mới đã chạy/kiểm đủ858 điều kiện:855 khớp,3 baseline MMLU lệch
nhẹ; toàn bộ PILOT khớp. Giữ nguyên bảng công bố, chỉ ghi đúng giới hạn của
lần dựng lại trong notes nếu cần. Không
gọi CPU MF là CUDA replay, không cộng replay vào
1.098 điều kiện OpenAI hoặc 3.626 run W&B, không ghi toàn bộ GPU đã chạy lại
khi còn chờ RAM/probe/fit. Kiểm trạng thái mới nhất trước khi sửa notes.

## Codex đã kiểm đọc bản đầu vào

Kiểm cấu trúc: deck 24 slide, 0,25 MB, notes 275–497 từ; poster một slide
420×594 mm, notes 455 từ. 42+5 link run nằm trong registry. Đã xem contact
sheet toàn bộ 24 trang và poster từ đúng hai PDF preview có hash trong nhật
ký; xem riêng slide 15, 18, 20, 21, 22 ở cỡ đọc được, chưa thấy tràn/đè khối
lớn. Đây không thay cho kiểm lại từng trang sau sửa.

Sau thay đổi, xuất lại PDF, xem slide bị sửa ở cỡ đọc thực tế, chạy kiểm cấu
trúc/notes/link, ghi hash mới vào nhật ký. Không sửa bộ `deliverables/` lịch
sử. Việc quét QR bằng điện thoại, mở trên máy khác và xác nhận Moodle vẫn
cần bằng chứng riêng; không đánh dấu hoàn thành từ kiểm tra XML.
