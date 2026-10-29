# Bàn giao thí nghiệm cục bộ — 29/10/2026

Cập nhật đóng gói 30/10/2026: đã sửa nhãn liên kết `paired-deltas.csv` và bổ sung
giải thích hai loại đối chứng. Minor được ghi trong khối QA lịch sử bên dưới đã
được xử lý; bảng số liệu và kết quả tính toán giữ nguyên. Việc chưa commit/push
được nhắc bên dưới là trạng thái tại thời điểm kiểm chứng ngày 29/10.

Đã hoàn tất toàn bộ kế hoạch thí nghiệm cục bộ đã chốt: **8 model, 3 seed,
1.098 lượt đánh giá**, đủ baseline, ngân sách, các mốc dữ liệu học và 5 ablation.
**108/108 nhóm** được đối soát bằng tính toán độc lập. Không gọi API trả phí.

[Bộ kết quả và báo cáo](../reports/full-experiments-2026-10-29/README.md) chứa
6 biểu đồ, báo cáo 9 trang, poster A1 ngang và các bảng từng seed. Đã kiểm đủ
912 điểm trên biểu đồ với bảng dữ liệu và xem trực tiếp mọi trang PDF qua ảnh
render Poppler; không có chữ tràn trang, chồng lấn hay chú thích bị cắt.

## Kết quả cần trình bày đúng

Tại ngân sách bằng 0,5 lần mốc tham chiếu, PILOT đạt utility **0,6764** trên
RouterBench và **0,6675** trên MMLU, phục vụ đủ truy vấn. LinUCB đạt **0,7056**
trên RouterBench, ExploreOnly đạt **0,6899** trên MMLU. Vì vậy không kết luận
PILOT luôn tốt nhất. Toàn bộ ngân sách và kết quả ablation, kể cả bất lợi, đều
được giữ trong báo cáo.

Utility là tổng reward chia số truy vấn yêu cầu, tính 0 cho truy vấn không được
phục vụ. Có **12/1.098 lượt** vượt ngân sách theo chi phí lịch sử ghi sẵn, dù vẫn
đúng giới hạn chi phí ước tính; mức vượt lớn nhất 0,127955416 đơn vị benchmark.
Đây không phải khoản tiền phát sinh khi chạy dự án.

## QA và tái lập

```text
Trạng thái:     XONG — kế hoạch thí nghiệm cục bộ 8 model đã chạy và đối soát thật; mục tiêu tương đương paper 11 model còn mở.
Gate:           Môi trường mới từ uv.lock, pytest → 441 passed / 1 skipped; baseline HEAD → 310 passed. OpenSpec strict validation đạt; 14/14 đầu việc hoàn tất.
E2E:            858 lượt chính + 240 ablation; 108 nhóm replay độc lập; 2.142.672 vòng học và 1.828.233 truy vấn được phục vụ. Source SHA nằm trong manifest từng run.
Probe:          4/4 lỗi CLI thật; 15/15 test chống Python optimization; resume đúng identity và replay bằng công cụ đóng gói đạt; 912/912 điểm biểu đồ khớp bảng.
Đường hỏng:     Protocol đổi, trace hỏng, artifact dở dang và output trùng đều bị từ chối, không sửa evidence gốc. -O/PYTHONOPTIMIZE bị chặn trước I/O.
Người thứ hai:  n/a — CLI offline, một người/một worker; đã có một reviewer mã độc lập.
Lưu trữ:        Đọc lại manifest/trace/checkpoint, so SHA và tính lại quyết định/chi phí; 7 file dữ liệu và 47 pin protocol/prior giữ nguyên.
Chưa chạy:      Test symlink thật bị Windows từ chối quyền; không tái tạo GPU/cache và toàn bộ grid trên máy thứ hai. Full 11-model paper-equivalent chưa đủ bằng chứng đầu vào.
Bug đã biết:    Minor — nhãn link paired-deltas trong README gói kết quả nói mọi delta so với PILOT; các dòng learning-size thực tế so từng phương pháp với chính nó ở full learning. CSV và phép tính đúng.
Quyết định:     Giữ nguyên protocol/evidence lịch sử, source snapshot thay cho tuyên bố clone mới; tách kết quả cục bộ khỏi mục tiêu tương đương paper.
Tôi sẽ lo:      Không còn bước triển khai bắt buộc nào của kế hoạch cục bộ. Mọi nghiên cứu thay protocol cần một bộ run mới để giữ bằng chứng này.
Cần bạn:        Không cần thêm phê duyệt cho phần đã giao. Dùng kết quả với các giới hạn khoa học bên dưới.
```

Hai môi trường CPU mới đều cài offline từ lockfile, một worker với BLAS một
thread. Mọi bước nặng đều qua ngưỡng RAM 8 GiB; các preflight cuối ghi khoảng
11–12 GiB trống. Smoke toy thực hiện 50 vòng học/6 truy vấn deployment;
baseline smoke chạy 7 phương pháp × 100 dòng. Smoke này được gắn nhãn synthetic.

Sau review, công cụ được sửa để không thể bỏ qua `assert` khi bật tối ưu Python.
Đã tái hiện lỗi trước sửa bằng bộ evidence rỗng được báo `verified=true`, sau đó
15 test chuyển đỏ sang xanh; suite đầy đủ đạt 441/1. Replay dữ liệu thật bằng
công cụ đã sửa vẫn khớp 32.270 vòng học và 29.076 quyết định ở 11 cấu hình.
Không thay source số học trong `scripts/` hay kết quả thí nghiệm đã hoàn tất.

Bằng chứng: [reproduction sau review](../reports/full-experiments-2026-10-29/reproduction-post-review/report.json),
[kiểm tra dữ liệu và baseline](../reports/full-experiments-2026-10-29/reproduction/report.json),
[review và xử lý](../reports/full-experiments-2026-10-29/review/findings.md),
[kiểm tra PDF/hình](../reports/full-experiments-2026-10-29/deliverable-checks.json),
[lệnh tái lập](full-experiments.md).

## Giới hạn và công việc ngoài kế hoạch đã hoàn tất

- Claude-v2, Mistral-7B-chat và Mixtral-8x7B-chat vẫn bị loại vì chưa chứng minh
  được mapping checkpoint/provider cần thiết. Không tự gán tên gần giống.
- Chưa xác lập đúng release/shot của tác giả, mối quan hệ 86 nhãn task với 64 task,
  nguồn/preparation Arena và holdout 500 mẫu của paper. Local holdout có 313 cặp.
- BGE-M3, token cap 2.048, phép chiếu 64D/8D, quy ước chi phí và adapter baseline
  là các thay thế/giả định cục bộ đã khai báo trước khi đánh giá.
- Ba seed cho mô tả biến thiên; MMLU là tập con của RouterBench, không phải một
  kiểm chứng độc lập trên dataset mới hay bằng chứng về độ chắc chắn thống kê.
- Source snapshot kiểm tra byte và môi trường mới nhưng vẫn trên máy này. Chưa
  có commit/push mới. Raw data, vector và trace lớn bị Git bỏ qua; chuyển máy cần
  mang thêm các artifact đúng hash, không chỉ clone repository.

## Các quyết định của người triển khai

1. Tiếp tục workspace hiện có thay vì tạo workspace của skill: giữ liên kết các
   artifact và phần việc chưa commit; đổi cách ghi sổ, không đổi protocol. Rủi ro
   nếu sai là giảm mức cô lập; source hash và snapshot được giữ để đối chiếu.
2. Giữ mục tiêu tương đương paper riêng: tránh biến dữ liệu thiếu bằng chứng
   thành kết luận khoa học; chi phí là chưa thể tuyên bố tái lập đủ 11 model.
3. Không coi review đọc mã là lần chạy GPU/full-grid mới: dùng evidence đã có,
   replay độc lập và môi trường CPU sạch; chưa chứng minh lặp lại GPU trên máy khác.
4. Reviewer không xem PDF trực tiếp; primary agent đã xem đủ mọi trang và đối
   chiếu số liệu. Phần thị giác chỉ có một lượt người đánh giá, được công khai.
5. Reviewer không điều tra lại nguồn upstream: dựa trên khảo sát nguồn gốc trước
   đó và checksum, giữ các câu hỏi tương đương nguồn chưa giải quyết.

Minor được hoãn duy nhất là nhãn liên kết CSV đã nêu; không có lỗi Critical hoặc
Important còn mở trong phạm vi review này.
