# Kiểm chứng chạy lại đầy đủ — 31/10/2026

**Trạng thái: regression, full CPU, MF CUDA và full Instructor đã kiểm xong;
Hybrid đang train, task shift còn chờ.** Không dùng số test thay cho bằng chứng
full grid; chênh lệch MF CPU và3 điều kiện Instructor được ghi riêng bên dưới.

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

## Kiểm chứng từng scope

[Lượt full CPU](https://github.com/kiettheis0702/pilot-llm-routing/actions/runs/41206426268)
chạy code `dbd52c82c86b8016fa3e413d5ba38a00f49d42ed`. Gate của lượt này đạt
722 test; các guard sau review được kiểm riêng ở commit nêu trên. Native
scientific producers không đổi giữa hai mốc. Các scope chạy tuần tự trên
những runner riêng, không dùng RAM máy cá nhân.

| Phần | Yêu cầu để chứng nhận | Trạng thái tại mốc ghi này |
|---|---|---|
| Core |3 prior fit mới,2 tuning mới,858 điều kiện,78 reader, so bảng công bố |Đạt; 1.740.921 learning round được replay; đủ858 điều kiện khớp bảng công bố |
| Followups |240 ablation,240 budget,12 noise,231 binary, đủ reader và so bảng |Đạt; đủ723 điều kiện khớp bảng;30+21 group reader và reader budget/noise |
| MF CPU |Môi trường neural mới,3 fit+mutation, NumPy reader, ghi chênh lệch với CUDA |Đạt về thực thi/readback;338/338/340 trên500, khác CUDA0/−2/−3 câu |
| Poster W1 |9 trường hợp synthetic,18 policy trace, so kết quả gốc |Đạt; toàn bộ stress/cost/comparison khớp, tổng reward/regret và số lần chọn được đọc lại |
| MF CUDA |Môi trường mới,3 fit+mutation, NumPy reader, so bảng gốc |Đạt;338/340/343 trên500; epoch84/88/100; mutation340→160 với trọng số không đổi |
| Instructor |Encode38.018 văn bản,858 điều kiện,78 reader |Đã chạy/kiểm đủ;855/858 khớp,3 điều kiện baseline MMLU khác; toàn bộ PILOT khớp |
| Hybrid và task shift |Hai fit5 epoch, readback neural;15.000 vòng task shift và reader |Binary đã qua probe/audit/declare, đang train; drift/task shift còn chờ |

[Job core hoàn thành](https://github.com/kiettheis0702/pilot-llm-routing/actions/runs/41206426268/job/124552610427)
lúc14:50 UTC ngày31/10, sau khoảng70 phút chạy native. Summary trong log ghi
`verified=true`, `matches_published=true`,78 reader và858 điều kiện. Đọc lại
độc lập từ summary so với CSV đóng băng xác nhận đúng tập ID và cả bốn trường:
served, reward, actual cost, estimated cost (rtol1e-10, atol1e-9).
Ba seed7/42/2026 và cả hai miền tuning được dựng mới;117 đầu vào core gốc
được giữ nguyên riêng.43/117 file dựng lại giống byte,74 file khác hash:
chỉ chứng nhận tương đương các chỉ số đã đối chiếu, không tuyên bố toàn bộ
prior/tuning giống byte giữa các máy. Cả16 package báo cáo vẫn nguyên vẹn.

[Job followups hoàn thành](https://github.com/kiettheis0702/pilot-llm-routing/actions/runs/41206426268/job/124552610557)
đã chạy240 ablation với30 reader/535.668 learning round,231 binary với21
reader/677.670 learning round,240 budget và12 noise. Binary có lượt tuning
mới18 candidate. Đọc lại riêng bằng CSV/JSON của thư viện chuẩn xác nhận
đúng toàn bộ ID và cả bốn trường nêu trên cho core và bốn nhóm followup;
sai khác lớn nhất của các giá trị đọc được bằng0. Có48 control một budget bin
trong bảng budget được tính ở ablation, không cộng trùng vào240 budget mới.

Checkout GPU là clone riêng trên ổ D; ZIP được tải từ release công khai,
kiểm đúng SHA256 rồi restore139 file.16 file model và34 file source công khai
đã kiểm hash. Môi trường mới đã thực sự nhận RTX5070, PyTorch2.8.0+cu128,
transformers4.57.1 và CUDA12.8. Wheel Windows CPython3.11 tải từ máy chủ PyTorch
chính thức có SHA256
`34c55443aafd31046a7963b63d30bc3b628ee4a704f826796c865fdfd05bb596`,
khớp index chính thức. Đường tải R2 ban đầu thất bại khi stream; cùng wheel
qua địa chỉ CloudFront tải xong và được kiểm hash trước khi cài.

Lượt cài thật phát hiện lệnh editable cũ cố đóng gói một repo flat layout.
Bản sửa `6a58f962e611a19b6e397f1637c0d3eba387e5ee` dùng `-r pyproject.toml --extra cost`
để cài dependency, giữ nguyên requirements neural. Đã quan sát lỗi thật rồi
cài thành công trong môi trường mới; regression local đạt728 test/1 skip
(quyền tạo symlink của Windows),28,91 giây. Lượt đầu gọi regression local
thiếu cache tokenizer và dùng đường dẫn tạm quá dài; chuẩn bị đúng cache đã
kiểm hash và đường dẫn ngắn giải quyết lỗi mà không sửa producer/test.
[Lượt MF CPU với bản sửa](https://github.com/kiettheis0702/pilot-llm-routing/actions/runs/41213653732)
đã hoàn tất: **729 passed,23,05 giây** ở gate, rồi cài môi trường neural mới,
chạy ba fit CPU100 epoch và một mutation thật, qua NumPy reader độc lập.
Lượt MF ở workflow cũ thất bại đúng lỗi editable install; core/followups của
workflow đó vẫn thành công. Không gọi toàn bộ workflow cũ là thành công.

MF CPU theo seed7/42/2026 đạt338/338/340 trên500, so với CUDA gốc338/340/343:
chênh0/−2/−3 câu (0/−0,4/−0,6 điểm phần trăm). Mutation test label của seed42
đổi338→162, trong khi trọng số giữ nguyên. Đây là kết quả CPU tái chạy, không
phải bằng chứng các thiết bị cho cùng điểm. Cả16 package báo cáo gốc nguyên vẹn;
các chỉ số khoa học công bố không bị thay bằng kết quả CPU.

MF CUDA có proof độc lập
`4b6a62ca64c03016cbc18aeeec21b77c0fb9542a94e422e9cd6993bee495f58c`;
đọc lại CSV xác nhận cả điểm, epoch và accuracy của ba seed khớp bảng công bố.
Poster W1 vẫn là synthetic supplement trên package BGE lịch sử, không phải
thêm điều kiện OpenAI.

Instructor đã encode đủ38.018 unique text, ánh xạ38.055 source row; reader
kiểm toàn bộ mapping/hash/token length và35 mẫu GPU theo source chính thức
(sai khác lớn nhất8,94e-8). Encoder ban đầu không ghi thêm khối sau37.632 vector;
hai lần đọc stack không dừng tiến trình thấy thao tác CUDA, nhưng chưa xác định
nguyên nhân driver. Người dùng duyệt dừng riêng encoder và tiếp tục cache.
Đã xác nhận khóa OS được giải phóng rồi chạy encoder gốc `--limit-chunks 1`
cho386 vector còn thiếu, giữ batch4/dynamic padding/FP32/cap512 và protocol.
Không dùng overlay batch1 của lần lịch sử.588 khối đầu được kiểm độc lập trước
và sau; SHA prefix giữ nguyên
`9315691d7389426f7f491a81861ba9ade2ce851eb8b48412e68939384b705fc0`.

Cache mới khác byte với cache khoa học gốc, không phải hỏng phần đã lưu trong
lượt mới. Đọc lại toàn bộ index, thứ tự ID, prompt hash và vector xác nhận
chỉ450 hàng thuộc đúng đoạn đuôi batch1/fixed padding của lượt gốc sai khác:
388 routing và62 preference; sai khác phần tử lớn nhất2,1234e-7. Các hàng còn
lại giống hệt. Lượt mới giữ batch4/dynamic padding, không áp overlay cũ.
Ba prior mới đạt193/189/190 trên313 cặp hiệu chỉnh theo seed7/42/2026, bằng
số đúng của bản gốc; cả hai tuning24 candidate đã xong. Full grid858 điều kiện
và78 reader đã hoàn tất, replay1.740.921 learning round. Đọc riêng inventory
xác nhận855 điều kiện khớp cả bốn chỉ số; ba điều kiện dưới khác tổng cộng
bảy giá trị. Cả ba vẫn phục vụ1275 câu; toàn bộ PILOT khớp bảng gốc.

| Điều kiện MMLU | Reward gốc → mới | Recorded cost gốc → mới | Estimated cost gốc → mới |
|---|---|---|---|
| seed42 / LinUCB / learn0,5 / budget0,5 |880 →880 |0,7957271 →0,7855726 |0,795627312312 →0,785446096231 |
| seed7 / EpochGreedy / learn1 / budget0,5 |864 →865 |0,8212302 →0,8164696 |0,821242306030 →0,816460966834 |
| seed7 / EpochGreedy / learn1 / budget1 |960 →960 |1,2642035 →1,2648615 |1,263999402261 →1,264663804271 |

Cost ở đây là đơn vị kế toán lịch sử, không phải phí API của lượt chạy lại.
Trace cho thấy6/8/1 quyết định triển khai thay đổi tương ứng, cùng thứ tự câu
hỏi. Hai luồng học liên quan đều giữ nguyên toàn bộ lựa chọn trên12.369 câu.
Tham số tuning được chọn và reference budget giống bản gốc; lower bound hiệu
chỉnh MMLU đổi từ0,1462959118294094 sang0,1462742919866876. Đây là bằng chứng
về chênh lệch của lần dựng lại, chưa phải phép can thiệp tách riêng nguyên
nhân. Giữ cả kết quả mới và báo cáo gốc; không nới tolerance để gọi858/858
khớp. Các số trong slide/poster vẫn lấy từ báo cáo khoa học gốc.

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
