# Hướng dẫn bảo vệ — Team 8

Hai thành viên đều cần hiểu toàn bộ bài. Chia người trình bày không thay thế việc
đọc và chạy thử. File giữa kỳ có 33 slide, ghi chú tiếng Anh trên từng trang;
poster gồm đúng hai điểm phản biện có thí nghiệm.

## Mạch trình bày đề xuất

1. Bài toán: mỗi câu hỏi chỉ chọn một model, cần cân bằng chất lượng và ngân sách.
2. Ý tưởng: học prior từ preference, cập nhật theo reward của model đã chọn,
   sau đó đóng băng router và áp chính sách ngân sách khi deployment.
3. Lý thuyết: prior chỉ cải thiện bound khi gần tham số đúng hơn điểm khởi tạo
   zero; phân tích PI-OFUL không tự động chứng minh mọi chi tiết cosine PILOT.
4. Thực nghiệm: hai nhóm core là budget–quality và learning-size/regret. Có
   1.098 điều kiện, không phải 1.098 lần gọi API. Báo cả kết quả bất lợi.
5. Phản biện: cosine mất độ lớn; ước lượng chi phí không bảo đảm chi phí thực.
6. Kết: chỉ khẳng định trong cấu hình local 8 model, embedding BGE-M3, ba seed.

## Những câu rất dễ được hỏi

**Model trong RouterBench là gì?** Là các model đã tạo câu trả lời trong bộ dữ
liệu lịch sử. Mỗi model là một arm để router chọn. Ta dùng reward và cost được
ghi sẵn, không chạy lại GPT/Claude hiện tại cho mỗi câu.

**Vì sao không dùng Claude Max/GPT Pro?** Giai đoạn này không cần sinh câu trả
lời mới: dữ liệu đã có. Subscription hỗ trợ đọc/viết/phân tích, nhưng thay model
lịch sử bằng model hiện tại sẽ đổi benchmark. Embedding đã chạy miễn phí local.

**Tại sao 8 model trong khi paper 11?** Ba bridge danh tính/phiên bản còn thiếu
bằng chứng đủ chắc. Không tự ghép tên gần giống. Đây là biến thể local có giới
hạn, không phải tái lập chính xác. Course cho core scope không đồng nghĩa tự
động duyệt thay đổi pool là tương đương.

**Thí nghiệm có leakage không?** Tuning riêng; chọn hyperparameter và ước lượng
cost trên tuning. Learner chỉ nhận reward của arm đã chọn. Deployment không
cập nhật model. Full reward chỉ dùng ở evaluator để tính regret.

**Tại sao PILOT có regret tốt mà utility deployment thấp hơn?** Regret đo
chất lượng trong learning, còn utility đo trên tập deployment với controller
ngân sách. Score, calibration, phân phối câu hỏi và budget ảnh hưởng khác nhau.
Chúng không phải cùng một mục tiêu đo. Chưa chứng minh nguyên nhân duy nhất.

**93% ở paper có giống 0,6764 của mình không?** Không. 93% là tỷ lệ so với GPT-4;
0,6764 là utility tuyệt đối ở nửa ngân sách tham chiếu local. Khác mẫu số và cấu
hình nên không lấy hai số này trừ/chia rồi gọi là lỗi tái lập.

**Stress cosine chứng minh paper sai?** Không. Chọn context (1,0), reward 0,2 và
0,8, prior cùng hướng. Cosine của cả hai bằng nhau nên mất phân biệt độ lớn.
Đây là test cố ý vi phạm giả định direction-only. Nó chỉ ra giới hạn khi mô hình
reward bị sai, không bác bỏ theorem có điều kiện.

**Control có phải LinUCB baseline không?** Là linear-mean control với cùng oracle
prior và A,b như cosine. Baseline LinUCB trong kết quả chính khởi tạo khác. Không
trộn hai phép so sánh.

**25/48 nghĩa là 52% request bị lỗi?** Không. Là 25 điều kiện thí nghiệm trên 48
điều kiện main PILOT full-learning vượt budget trong phép nhân cost 1,10, giữ
nguyên lựa chọn. Không phải xác suất ngoài đời hay số request; không reroute.

**Có vượt ngân sách gốc không?** Main PILOT full-learning: 0/48. Toàn bộ phương
pháp/ablations: 12/1.098. Đây là cost lịch sử mô phỏng, không có khoản API bị trừ.

**Ba seed đã đủ kết luận thống kê?** Chỉ báo mô tả mean/min/max, không gọi là
confidence interval, không bịa p-value. MMLU nằm trong RouterBench, không độc lập.

**Lỗi công thức appendix có làm theorem hỏng?** Chỉ ra đúng dòng và identity
ridge đã sửa; phân biệt lỗi in thiếu reward/inverse với chứng minh sai. Không
phóng đại kết luận.

**W&B nằm đâu?** [Project công khai](https://forge.coreweave.com/wandb/kiettheis-university-of-science-vnu-hcm/pilot-llm-routing-team8)
có 1.128 run lịch sử, đã đọc lại không đăng nhập đủ 13.314 chỉ số. Dùng
[mapping hiện tại](../reports/publication-2026-10-30/experiment-map.csv) để đi từ
thí nghiệm sang script và run. Đây là historical aggregates, không phải các
lượt gọi LLM mới. GitHub `main` là bản đầy đủ; chưa nộp Moodle.

## Tập thao tác trước buổi bảo vệ

- Mỗi người giải thích một round chọn arm/cập nhật A,b bằng tay.
- Tìm một số trên slide trong CSV và chỉ ra ba seed tạo ra mean đó.
- Chạy toy smoke, giải thích vì sao synthetic không phải kết quả paper.
- Tự giải thích cả hai poster tests và một cách kiểm chứng cải tiến tương ứng.
- Mở cả hai PPTX bằng PowerPoint trên máy trình chiếu, kiểm font/chart/notes;
  bản hiện tại đã kiểm cấu trúc và render, chưa xác nhận bằng ứng dụng PowerPoint.

Mốc trên Moodle khi kiểm 30/10: GitHub công khai trước 23:59 07/11/2026;
slide giữa kỳ 21/11; poster cuối kỳ 05/12. Kiểm lại Moodle trước khi nộp.
