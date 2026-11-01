# Hướng dẫn bảo vệ — Team 8

Hai thành viên đều cần hiểu toàn bộ bài. Chia người trình bày không thay thế việc
đọc và chạy thử. Bắt đầu bằng [hướng dẫn đọc toàn dự án](project-reading-guide-vi.md).
File hiện tại `midterm/8.pptx` có24 slide; bộ đóng gói lịch sử có33 slide.
Poster hiện tại có1 slide A2. Bản OpenAI đã được cập nhật;
[yêu cầu rà cuối cho Claude](CLAUDE_FINAL_REVIEW_2026-10-31.md) thu hẹp cách
diễn đạt ở slide22. Giữ các số liệu công khai và cả kết quả bất lợi.

## Mạch trình bày đề xuất

1. Bài toán: mỗi câu hỏi chỉ chọn một model, cần cân bằng chất lượng và ngân sách.
2. Ý tưởng: học prior từ preference, cập nhật theo reward của model đã chọn,
   sau đó đóng băng router và áp chính sách ngân sách khi deployment.
3. Lý thuyết: prior chỉ cải thiện bound khi gần tham số đúng hơn điểm khởi tạo
   zero; phân tích PI-OFUL không tự động chứng minh mọi chi tiết cosine PILOT.
4. Thực nghiệm: hai nhóm core là budget–quality và learning-size/regret. Có
   1.098 điều kiện OpenAI=858 chính+240 ablation, riêng với1.098 điều kiện BGE
   lịch sử; đây không phải số lần gọi API. Báo cả kết quả bất lợi.
5. Phản biện: cosine mất độ lớn; ước lượng chi phí không bảo đảm chi phí thực.
6. Kết: chỉ khẳng định trong cấu hình local8 model, embedding OpenAI, ba seed;
   BGE là mốc so sánh lịch sử. Ba model còn thiếu chưa được xác minh.

## Kết quả bổ sung đã kiểm chứng ngày31/10

Trên cùng500 cặp test local, RouteLLM-MF đạt trung bình68,07%, PILOT63,53%.
Giữ kết quả này dù thứ hạng ngược paper. Hai mô hình dùng cùng split/cache,
nhưng khác kiến trúc, chiều và số epoch; từng protocol và cách chọn checkpoint
được công bố riêng.500 cặp được giữ khỏi những lượt fit mới, nhưng từng xuất
hiện trong lịch sử dự án và chưa xác nhận trùng500 cặp tác giả. Không gọi ba
seed là1500 quan sát test độc lập. [Bằng chứng](../reports/openai-routellm-2026-10-31/README.md).

Với phản hồi nhiễu5% trên MMLU, trung bình true reward/request giảm0,3137điểm
phần trăm so với0%noise; coverage100%. Đây là thay thế phản hồi đã chọn bằng
bit ngẫu nhiên, không phải cưỡng ép đảo5%nhãn, và không làm nhiễu nhãn chấm cuối.
Đây là biến thể local, không phải xác nhận đúng số của Table5.
[Bằng chứng](../reports/openai-noise-2026-10-31/README.md).

Ở20%ngân sách ước lượng MMLU, quy tắc dừng main chỉ phục vụ2,27%query.
Biến thể bỏ qua query chưa trả được rồi tiếp tục theo thời gian đạt97,31%
coverage, với cùng scores/cost/order. Đây là phân tích chính sách local; giữ
kết quả main và không nói mọi mức ngân sách đều cải thiện.
[Bằng chứng](../reports/openai-budget-2026-10-31/README.md).

Instructor-XL đã chạy858 cấu hình và được kiểm chứng độc lập đủ78 nhóm.
Ở50%allocation, PILOT đạt89,1763%quality/GPT4 trênRouterBench và84,0791%
trênMMLU, coverage100%; so vớiOpenAI tăng2,5810 và0,5512điểm phần trăm.
Ở100%allocation, chất lượng lại giảm nhẹ0,5978 và0,2270điểm. Không nói
Instructor luôn tốt hơn. Đây là so sánh cả công thức embedding và tuning lại:
khác tokenizer/cap512 so vớiOpenAI full text, instruction local chưa xác nhận
tác giả. [Bằng chứng](../reports/instructor-core-2026-10-31/README.md).

Đổi miền MMLU sangGSM8K đã chạy và kiểm chứng đủ15000 round, một seed42.
Sau chuyển miền, PILOT đạt65,605%mean true reward, Hybrid tĩnh65,810%; cả hai
coverage100%. Không kết luận PILOT tốt hơn sau chuyển miền. Hybrid chọnGPT4
cho toàn bộGSM8K; PILOT đạt99,6885%quality/GPT4 với98,7623%cost/GPT4.
Khác thông tin huấn luyện và kiểu đầu vào; mức giảm raw reward còn phản ánh
độ khó tác vụ. [Bằng chứng](../reports/openai-task-shift-2026-10-31/README.md).

HybridLLM hai model đã huấn luyện5 epoch, chọn epoch4 chỉ bằng validation,
đọc lại đủ5x1000 validation và3227 test logits với sai khác0. Ở native threshold
0,5 đạt81,6664%quality/GPT4,23,1174%cost/GPT4, coverage100%. PILOT seed42 tại
allocation0,5 đạt72,4896%quality và46,1573%cost. Hai số0,5 là hai nút điều khiển
khác nhau; không nói cùng budget hoặc một phương pháp thắng ở mọi điểm.
Giữ đủ đường cong và khác biệt thông tin huấn luyện/đầu vào, một seed.
[Bằng chứng](../reports/openai-hybrid-2026-10-31/README.md).

## Những câu rất dễ được hỏi

**Model trong RouterBench là gì?** Là các model đã tạo câu trả lời trong bộ dữ
liệu lịch sử. Mỗi model là một arm để router chọn. Ta dùng reward và cost được
ghi sẵn, không chạy lại GPT/Claude hiện tại cho mỗi câu.

**Vì sao không dùng Claude Max/GPT Pro?** Giai đoạn này không cần sinh câu trả
lời mới: dữ liệu đã có. Subscription hỗ trợ đọc/viết/phân tích, nhưng thay model
lịch sử bằng model hiện tại sẽ đổi benchmark. Embedding OpenAI đã hoàn tất từ
lượt trả phí được duyệt trước đó; các lượt fit/đánh giá bổ sung dùng cache,
không gọi thêm API có phí.

**Tại sao 8 model trong khi paper 11?** Ba bridge danh tính/phiên bản còn thiếu
bằng chứng đủ chắc. Không tự ghép tên gần giống. Đây là biến thể local có giới
hạn, không phải tái lập chính xác. Course cho core scope không đồng nghĩa tự
động duyệt thay đổi pool là tương đương.

**Thí nghiệm có leakage không?** Với phần bandit, tuning riêng; chọn hyperparameter và ước lượng
cost trên tuning. Learner chỉ nhận reward của arm đã chọn. Deployment không
cập nhật model. Full reward chỉ dùng ở evaluator để tính regret.

**Tại sao PILOT có regret tốt mà utility deployment thấp hơn?** Regret đo
chất lượng trong learning, còn utility đo trên tập deployment với controller
ngân sách. Score, calibration, phân phối câu hỏi và budget ảnh hưởng khác nhau.
Chúng không phải cùng một mục tiêu đo. Chưa chứng minh nguyên nhân duy nhất.

**93% ở paper có giống utility local không?** Không. Hiện đã chuẩn hóa đúng mẫu
số GPT-4 trên cùng toàn bộ query được yêu cầu. OpenAI PILOT ở50%allocation
RouterBench đạt86,60%quality/GPT4 với46,23%recorded-cost/GPT4. Đó không phải
điểm25%cost như paper; pool/split/release vẫn khác hoặc chưa xác minh. Không
nội suy một điểm rồi gọi là phép đo hay tính lỗi tái lập với93%/25%.

**Stress cosine chứng minh paper sai?** Không. Chọn context (1,0), reward 0,2 và
0,8, prior cùng hướng. Cosine của cả hai bằng nhau nên mất phân biệt độ lớn.
Đây là test cố ý vi phạm giả định direction-only. Nó chỉ ra giới hạn khi mô hình
reward bị sai, không bác bỏ theorem có điều kiện.

**Control có phải LinUCB baseline không?** Là linear-mean control với cùng oracle
prior và A,b như cosine. Baseline LinUCB trong kết quả chính khởi tạo khác. Không
trộn hai phép so sánh.

**23/48 nghĩa là gần nửa request bị lỗi?** Không. Là23 điều kiện OpenAI trên48
điều kiện main PILOT full-learning vượt budget khi nhân cost1,10, giữ nguyên
lựa chọn. BGE lịch sử là25/48. Đây không phải xác suất ngoài đời hay số request;
không reroute trong phép thử này.

**Có vượt ngân sách gốc không?** OpenAI main PILOT full-learning:0/48. Số
12/1098 thuộc bộ BGE lịch sử; không gán nó cho bộ OpenAI. Đọc đúng population
trong CSV sensitivity trước khi nói về toàn bộ phương pháp/ablations. Đây là
cost lịch sử mô phỏng, không phải khoản API đang bị trừ.

**Ba seed đã đủ kết luận thống kê?** Chỉ báo mô tả mean/min/max, không gọi là
confidence interval, không bịa p-value. MMLU nằm trong RouterBench, không độc lập.

**Lỗi công thức appendix có làm theorem hỏng?** Chỉ ra đúng dòng và identity
ridge đã sửa; phân biệt lỗi in thiếu reward/inverse với chứng minh sai. Không
phóng đại kết luận.

**W&B nằm đâu?** [Project công khai](https://forge.coreweave.com/wandb/kiettheis-university-of-science-vnu-hcm/pilot-llm-routing-team8)
có3626 bản ghi:1128 lịch sử giữ nguyên,2498 summary views mới. Đã đọc lại không
đăng nhập đủ35402 chỉ số mới cùng config và xác minh lại toàn bộ dữ liệu cũ.
Dùng [registry mới](../reports/openai-publication-2026-10-31/experiment-map.csv)
cho OpenAI/Instructor/Hybrid; [registry cũ](../reports/publication-2026-10-30/experiment-map.csv)
chỉ minh chứng số BGE/synthetic/cost cũ. Đây là các điều kiện/tổng hợp có cửa sổ
chồng lấn, không phải2498 lần gọi LLM hoặc dataset độc lập. Artifact mới70 file
bảng/biểu đồ; artifact cũ19 file giữ nguyên. Trạng thái Moodle cần xác nhận
trên tài khoản đang đăng nhập; không suy ra từ trạng thái lần kiểm30/10.

## Tập thao tác trước buổi bảo vệ

- Mỗi người giải thích một round chọn arm/cập nhật A,b bằng tay.
- Tìm một số trên slide trong CSV và chỉ ra ba seed tạo ra mean đó.
- Chạy toy smoke, giải thích vì sao synthetic không phải kết quả paper.
- Tự giải thích cả hai poster tests và một cách kiểm chứng cải tiến tương ứng.
- Mở cả hai PPTX bằng PowerPoint trên máy trình chiếu, kiểm font/chart/notes;
  bản hiện tại đã xuất bằng PowerPoint trên máy soạn theo nhật ký cập nhật,
  chưa xác nhận trên máy trình chiếu/in khác.

Mốc trên Moodle khi kiểm 30/10: GitHub công khai trước 23:59 07/11/2026;
slide giữa kỳ 21/11; poster cuối kỳ 05/12. Kiểm lại Moodle trước khi nộp.
