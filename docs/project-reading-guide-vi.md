# Đọc toàn bộ dự án PILOT — bản OpenAI hiện hành

Tài liệu này là điểm bắt đầu cho cả hai thành viên nhóm 8. Kết quả khoa học
được chốt ngày 31/10/2026; những lần chạy lại để kiểm tra khả năng tái lập không
được cộng thêm vào số điều kiện nghiên cứu hay số run W&B đã công bố.

## Đọc theo thứ tự nào?

1. Đọc tài liệu này để nối bài toán, thuật toán, thí nghiệm và kết luận.
2. Đọc [giải thích từng slide và poster](huong-dan-slide-poster-vi.md), đồng thời
   mở [deck 24 slide](../midterm/8.pptx) và [poster A2](../poster/8.pptx).
3. Đọc [kết quả OpenAI chính](../reports/openai-core-2026-10-31/KET_QUA_VI.md)
   và [tổng hợp các nghiên cứu phụ](openai-followup-status.md). Tìm lại ít nhất
   một con số trong CSV, qua registry, đến đúng run W&B.
4. Tự trả lời [câu hỏi bảo vệ](course-defense-guide-vi.md), không nhìn notes.
5. Đọc [khôi phục checkout sạch](openai-reproduce.md) và
   [thứ tự chạy các nghiên cứu phụ](followup-reproduce.md) để biết người chấm
   có thể dựng lại gì, bằng đầu vào nào và với giới hạn nào.

`CLAUDE_HANDOFF.md`, bộ `deliverables/` 33 slide và các tài liệu BGE ngày
30/10 là lịch sử. Không dùng chúng làm nguồn số liệu hiện hành. Bản giải thích
cũ `GIAI_THICH_TOAN_BO_DU_AN_VI.md` ở máy cá nhân cũng thuộc mốc BGE đó.

## 1. Đồ án làm gì?

Nhóm tái lập và phản biện paper *Adaptive LLM Routing under Budget Constraints*.
Mỗi câu hỏi được giao cho **một** LLM trong pool. Router phải cân bằng chất
lượng với ngân sách, chỉ nhận phản hồi của model đã chọn khi học online.
Nhóm huấn luyện **router**, không huấn luyện lại GPT-4 hay tải tám LLM về máy.

Nguồn yêu cầu môn học, đăng ký đã được chấp nhận và rubric nằm trong
[bản đối chiếu yêu cầu](course-requirements-audit-2026-10-30.md). Phần trạng thái
nộp bài ở tài liệu đó chỉ phản ánh ngày kiểm tra 30/10, không phải trạng thái
Moodle hiện tại. Chọn core scope nghĩa là tập trung các thí nghiệm trọng tâm;
không có nghĩa mọi thay đổi so với paper được xem là tương đương.

Ba loại model cần phân biệt:

| Loại | Trong bản hiện hành | Vai trò |
|---|---|---|
| Model trả lời | Tám model được xác minh trong dữ liệu | Các arm router được phép chọn |
| Model embedding | `text-embedding-3-small`, 1.536 chiều | Chuyển nội dung câu hỏi thành vector |
| Router | PILOT, LinUCB và các baseline | Chọn arm từ vector và trạng thái đã học |

BGE-M3 là embedder của đợt lịch sử; Instructor-XL là nghiên cứu độ nhạy riêng.
Embedding không phải một arm cạnh tranh với GPT-4.

## 2. Dữ liệu và chi phí có thật không?

RouterBench cung cấp câu hỏi cùng kết quả và chi phí đã ghi sẵn của các model.
Khi router chọn arm, chương trình đọc đúng outcome của arm đó. Đây là thí
nghiệm thuật toán thực chạy trên môi trường offline, không phải gọi lại mọi
LLM cho mỗi câu hỏi. Chi phí trong biểu đồ là chi phí benchmark được hạch toán.

Vector OpenAI là vector thật đã thu được trong lượt embedding được duyệt trước
đó, gồm 38.018 văn bản duy nhất. Những lần fit và replay hiện tại dùng cache.
Không suy ra số request API từ số điều kiện, số seed hoặc số bản ghi W&B.

Pool hiện tại có tám arm; paper dùng 11. Ba ánh xạ phiên bản còn thiếu chưa có
bằng chứng đủ chắc. Ngoài ra còn khác hoặc chưa xác minh được release, 86 nhãn
task raw so với 64 task paper, split/order và một số thiết lập tác giả.
[Bằng chứng và giới hạn](openai-followup-source-audit.md).

## 3. Thuật toán chạy qua ba giai đoạn

**Học prior từ preference.** Vector 1.536 chiều được chiếu xuống 64 chiều.
Từ các cặp bình chọn của con người, nhóm học không gian chung cho câu hỏi và
vector đại diện từng arm. Vector này chỉ là tham số router, không phải trọng
số của LLM. Prior tạo điểm xuất phát cho bandit.

Trong core, 1.245 cặp train và 313 cặp calibration được giữ theo giao thức
đã chốt. Accuracy calibration 60,70% không phải accuracy trên test500.
Nghiên cứu preference500 dùng split riêng 847/211/500; không trộn hai mẫu số.

**Học bandit.** Với mỗi câu hỏi, router tính score gồm dự đoán cosine và bonus
khám phá. Nó chọn một arm, quan sát reward của arm đó rồi cập nhật A,b của
chính arm ấy. Không lấy reward của arm không chọn để học. Các tham số α,
cửa sổ và ước lượng cost được chọn trên tuning; không chọn bằng deployment.

**Deployment dưới ngân sách.** Router được đóng băng. Bộ điều khiển dùng cost
ước lượng, các bin và tiền dư chuyển tiếp để chọn arm khả thi. Quy tắc fallback
và dừng là một phần của phương pháp đã khai báo. Không có arm khả thi có thể
làm luồng dừng dù tổng ngân sách vẫn còn; đây là kết quả phải báo, không xóa.

Full reward của các arm chỉ xuất hiện trong evaluator khi tính regret hay đối
chiếu kết quả. Việc evaluator biết bảng benchmark không cho phép learner dùng
toàn bộ bảng đó để cập nhật.

## 4. Đọc số liệu cho đúng

| Chỉ số | Ý nghĩa | Không được nhầm với |
|---|---|---|
| Quality/GPT-4 | Tổng reward được phục vụ / reward GPT-4 trên cùng toàn bộ query yêu cầu | Accuracy chỉ trên query đã phục vụ |
| Recorded cost/GPT-4 | Chi phí outcome được chọn / chi phí GPT-4 trên cùng query | Ngân sách được cấp hoặc hóa đơn API mới |
| Allocation | Tỷ lệ ngân sách theo cost GPT-4 **ước lượng từ tuning** | Tỷ lệ chi phí thực đã tiêu |
| Coverage | Số query được phục vụ / số query yêu cầu | Chất lượng câu trả lời |
| Cumulative oracle regret | Tổng reward tốt nhất trừ reward arm đã chọn trong learning | Utility deployment dưới budget |
| Mean/min/max ba seed | Thống kê mô tả ba lần chạy | Khoảng tin cậy hoặc ba bộ test độc lập |

Ví dụ ở RouterBench, 50% allocation của PILOT cho 86,5953% quality/GPT-4,
46,2348% recorded cost/GPT-4 và 100% coverage. Điểm này không phải điểm
93% quality tại 25% cost trong paper. Không nội suy rồi gọi là kết quả đo khớp.

## 5. Kết quả nào cần nhớ và giải thích được?

Core có 858 điều kiện; ablation có 240. Tổng **1.098 điều kiện OpenAI** tách
khỏi 1.098 BGE lịch sử. Hai thí nghiệm chính là chất lượng theo ngân sách và
ảnh hưởng learning-size/regret. Các nghiên cứu phụ trả lời câu hỏi bổ sung.

- Ở nửa allocation, PILOT thấp hơn LinUCB trong kết quả local. PILOT có regret
  thấp nhất **trong các learner thích nghi**; always-GPT-4 có regret thấp hơn.
  Regret tốt không tự động cho deployment tốt dưới bộ điều khiển ngân sách.
- Trên MMLU ở 20% allocation, main chỉ phục vụ 2,27% query; biến thể skip riêng
  phục vụ 97,31%. Kết quả skip không thay thế hay sửa ngược kết quả main.
- Prior và các ablation cho hiệu ứng lẫn lộn. Bỏ cập nhật online làm chất
  lượng giảm mạnh, nhưng không được kết luận mọi thành phần luôn có lợi.
- Trên 500 cặp local, RouteLLM-MF đạt trung bình 68,0667%, PILOT 63,5333%.
  Giữ thứ hạng ngược paper. Khác kiến trúc, chiều, epoch và chọn checkpoint;
  500 cặp từng xuất hiện trong lịch sử dự án và chưa xác nhận là 500 cặp tác giả.
- Instructor cải thiện quality ở một số allocation và giảm ở điểm khác.
  Đó là so sánh cả công thức embedding và tuning lại, không cô lập tên model.
- Sau task shift, PILOT 65,605% true reward/query, Hybrid 65,810%, coverage
  đều 100%. Không kết luận PILOT thắng hoặc một seed đã chứng minh thích nghi.
- Hybrid binary tại threshold 0,5 đạt 81,6664% quality/GPT-4 và 23,1174% cost.
  PILOT seed42 tại allocation 0,5 đạt 72,4896% và 46,1573%. Hai số 0,5 điều
  khiển hai thứ khác nhau; không gọi đây là so sánh cùng budget.

Mỗi số trên có bảng nguồn trong [trang kết quả hiện hành](openai-followup-status.md).
Các lần chạy lại trên checkout mới kiểm tra khả năng tái lập; nếu có sai khác,
phải lưu và giải thích riêng, không sửa bảng khoa học gốc để làm kiểm tra xanh.

## 6. Hai phản biện trên poster

**W1 — cosine bỏ mất độ lớn.** Hai vector cùng hướng nhưng có độ lớn khác
nhau cho cùng cosine. Case synthetic cố ý đặt reward 0,2/0,8 và prior cùng
hướng. Ở 2.000 round, α=1, cosine regret 600, linear-mean control 1,2.
Control dùng cùng prior và cập nhật; không phải LinUCB zero-prior trong core.
Đây là case vi phạm giả định direction-only, không bác bỏ theorem có điều kiện.

**W2 — budget ước lượng khác bill thực.** Giữ nguyên lựa chọn và reward rồi
nhân recorded cost lên 1,10. Có 23/48 điều kiện main PILOT full-learning vượt
budget, so với 0/48 ở 1,00. Đây là số điều kiện, không phải tỷ lệ request lỗi,
không reroute, không phát sinh khoản thanh toán. BGE lịch sử là 25/48.

Hướng cải tiến tương ứng: thử score giữ độ lớn và dự phòng ngân sách theo
phân phối cost. Chúng là đề xuất, chưa được chứng minh hiệu quả bằng kết quả
poster. Không gọi đề xuất là tính năng đã triển khai.

## 7. Bằng chứng được nối ra sao?

CSV/JSON và package manifest trong `reports/` là bằng chứng khoa học được
đóng băng. Registry nối điều kiện → script → run W&B. Project có 3.626 bản
ghi: 1.128 lịch sử và 2.498 view/tổng hợp mới. Một view không đồng nghĩa một
dataset hay một lần gọi LLM độc lập.

Checkout Git chứa code/báo cáo; archive riêng chứa 139 đầu vào, gồm cache
embedding, có checksum. Hướng dẫn restore, fit, replay độc lập và các biên
kiểm tra hoàn thành nằm trong [reproduction guide](followup-reproduce.md).
Test phần mềm, replay trace, refit và neural inference là các mức bằng chứng
khác nhau; test xanh không tự chứng minh đã chạy lại toàn bộ grid GPU.

## 8. Việc hai thành viên phải tự làm

Mỗi người tự giải thích một round cập nhật A,b; phân biệt prior/calibration/test;
đọc đúng mẫu số của một biểu đồ; tìm một số về đúng CSV và W&B; giải thích
hai phản biện cùng giới hạn; trả lời vì sao đây là bản local tám arm chứ chưa
phải tái lập chính xác paper. Chỉ đọc thuộc kết luận là chưa đủ.

Mở PPTX trên máy dùng để trình chiếu/in để kiểm font, chart, notes và QR.
Xác nhận file/link đã nộp trên đúng tài khoản Moodle. Việc AI chuẩn bị tài liệu
không chứng minh hai thành viên đã hiểu, cũng không thay cho thao tác nộp bài.
