# Kết quả OpenAI core — 31/10/2026

Đã hoàn tất huấn luyện và đánh giá bằng cache OpenAI thật: **858 điều kiện, 78 lượt học, 312 snapshot**, ba seed 7/42/2026, trên RouterBench và tập con MMLU. Cả 78 nhóm đã được tính lại độc lập. Lượt công việc này không gọi thêm API trả phí.

Đầu vào là `text-embedding-3-small`, 1.536 chiều; học projection 64 chiều trong 30 epoch và prior trong 40 epoch. Giữ nguyên 8 model, split, thứ tự câu hỏi và 1.245 cặp học / 313 cặp hiệu chỉnh. Tuning mới chỉ dùng 1.000 câu RouterBench hoặc 398 câu MMLU.

## Các số liệu nên dùng trong đồ án

Trung bình ba seed, dùng toàn bộ câu hỏi deployment làm mẫu số. Câu không được phục vụ đóng góp reward bằng 0. Chất lượng và chi phí thực tế đều so với GPT-4 trên đúng cùng tập câu hỏi.

| Tập dữ liệu | Ngân sách cấp theo ước lượng | Chất lượng PILOT / GPT-4 | Chi phí ghi nhận / GPT-4 | Câu được phục vụ |
|---|---:|---:|---:|---:|
| RouterBench | 20% | 77,81% | 19,23% | 100% |
| RouterBench | 50% | 86,60% | 46,23% | 100% |
| RouterBench | 100% | 97,94% | 91,57% | 100% |
| MMLU | 20% | 1,72% | 1,10% | 2,27% |
| MMLU | 50% | 83,53% | 47,53% | 100% |
| MMLU | 100% | 98,44% | 91,81% | 100% |

Ở mức cấp 50%, LinUCB có chất lượng cao nhất trong 13 phương pháp: 91,28% trên RouterBench và 86,96% trên MMLU; PILOT đứng thứ 4. Ở mức cấp 100%, PILOT đứng đầu theo tổng reward, đồng thời dùng chi phí cao hơn các router thích chọn model rẻ. Cần đọc cả chất lượng, chi phí và coverage; không kết luận một phương pháp luôn tốt nhất.

Không có điều kiện nào trong 858 điều kiện mới vượt ngân sách theo chi phí ghi nhận. Đây là kết quả quan sát, không phải bảo đảm cho dữ liệu khác.

## OpenAI so với BGE

Đã ghép đúng 858 điều kiện core cũ và mới theo ID, dataset, seed, phương pháp, lượng dữ liệu học và mức cấp ngân sách. Riêng PILOT học đầy đủ, trung bình đều trên 8 mức ngân sách và 3 seed, thay đổi chất lượng là **−0,334 điểm phần trăm trên RouterBench** và **−2,760 điểm phần trăm trên MMLU**.

OpenAI không làm kết quả tốt hơn đồng đều. Đây là so sánh cả cách biểu diễn và tuning mới: BGE có giới hạn 2.048 token làm cắt 13 đầu vào preference; OpenAI dùng toàn văn. Không quy toàn bộ chênh lệch cho riêng encoder. MMLU là tập con RouterBench, không phải bằng chứng độc lập thứ hai.

## Vì sao MMLU ngân sách thấp bị sụt mạnh

Đã kiểm tra độc lập 6 trường hợp: ba seed tại mức cấp 20% và 30%. Ở 20%, cả ba seed dừng sau **29/1.275 câu**; ở 30%, dừng sau **350/1.275 câu**.

Tại câu dừng, ngưỡng chọn không nhận model nào; phần dự phòng chia ngân sách còn lại cho từng câu trong bin thấp hơn cả giá ước lượng của model rẻ nhất. Quy tắc hiện tại vì thế kết thúc stream, dù tổng ngân sách chưa tiêu hết và vẫn có model mua được nếu chỉ xét phần tiền hiện có. Đây là hành vi của protocol đã đóng băng, không phải sai số phép cộng hoặc hết toàn bộ tiền. Bằng chứng từng trường hợp nằm trong `low-budget-diagnostic.json`.

Giữ nguyên các kết quả này. Nếu thử cơ chế tiếp tục/đổi bin thì phải ghi thành biến thể mới, dùng cùng checkpoint và mẫu số; không thay ngầm kết quả main sau khi đã xem outcome. Đây là hướng kiểm tra miễn phí đáng ưu tiên tiếp theo.

## Giới hạn tái lập paper

- Vẫn là 8 model; ba alias/version chưa đủ bằng chứng để nâng thành 11 model.
- 86 nhãn tác vụ thô so với 64 trong paper vẫn chưa giải thích được; không bỏ nhãn cho giống số.
- 313 cặp dùng hiệu chỉnh hệ số khởi tạo, không phải tập kiểm tra preference độc lập.
- 64 chiều và 30/40 epoch là cấu hình local công bố rõ, chưa xác minh trùng tác giả.
- Giữ cơ chế ngân sách guarded đã khai báo. Không báo đã đạt lại “93% chất lượng / 25% chi phí” của paper.
- Không có thí nghiệm OpenAI mới cho ablation, Instructor-XL, HybridLLM hoặc RouteLLM trong gói core này.
- Một số dòng `research_blockers` trong fit manifest kế thừa nguyên văn manifest chuẩn bị dữ liệu BGE cũ. Danh tính embedding thực tế phải đọc từ `prior.embedding.recipe`, cache fingerprint và protocol OpenAI đã kiểm tra, không suy từ các ghi chú lịch sử đó.

## File dùng cho báo cáo và tái kiểm tra

- `quality-cost.svg/png`: chất lượng–chi phí của 5 router; bảng CSV có đủ 13 phương pháp.
- `pilot-bge-openai.svg/png`: PILOT cũ và mới; thanh dọc là min–max của ba seed.
- `learning-size.svg/png`, `learning-regret.svg/png`: kết quả theo bốn prefix học lồng nhau.
- `per-seed.csv`, `mean-min-max.csv`, `paired-bge.csv`, `learning-prefixes.csv`: số liệu để vẽ/chọn bảng; `model-catalog.csv` giải nghĩa các baseline fixed.
- `tracking-run-configs.jsonl`: 858 cấu hình, chuẩn bị offline; chưa upload các run mới.
- `evidence/`: protocol, tuning, xác minh prior/cache, log test và kiểm tra cuối.
- `package-files.json`: hash toàn bộ gói kết quả. Raw trace/checkpoint nằm ở `outputs/openai-core-20261031/`.

## Bàn giao tự QA

Trạng thái: XONG — đã chạy thật phạm vi OpenAI core.

Gate: 596 passed, 1 skipped; lần cuối đạt trong 23,90 giây. Có một lần gặp WinError 5 khi đổi tên file tạm trong test cache BGE không sửa đổi. Test đó và cả suite đạt khi chạy lại trên thư mục mới; nguyên nhân lỗi truy cập chưa xác định. Giữ cả log lỗi và log chạy lại.

E2E: 858 điều kiện thật trên cache API đã hoàn tất, 1.740.921 lượt học, 1.450.207 lượt deployment được phục vụ; 78/78 nhóm tính lại độc lập. Protocol SHA-256: `5cee68cc19a1f440e01684497475f50de83f9212e6bb04cf9726828cd7bc6859`.

Probe: 4/4 kiểm tra lỗi thực tế đạt — pin bị đổi, artifact dở dang, inventory thiếu và bản sao artifact hoàn tất bị sửa byte. Actual-cache resume đạt; 12 test mới về inventory, prior, lineage, normalization và ghép điều kiện đều đạt. Chẩn đoán thêm 6/6 trường hợp dừng sớm khớp quy tắc.

Đường hỏng: từ chối dữ liệu/hash sai trước khi ghi kết quả mới; không ghi báo cáo hoàn tất khi thiếu điều kiện. Một lần gọi helper readback bằng `exec` thiếu `__file__` đã được sửa cách gọi bằng `runpy`; các phép kiểm số liệu trước đó đã đạt, raw output không đổi.

Người thứ hai: n/a — pipeline offline một người.

Lưu trữ: đã đọc lại manifest/file hash, CSV, checkpoint, trace; ba gói lịch sử 184/13/8 file vẫn qua kiểm chứng. Đã xem cả bốn PNG, trục/legend/nội dung hiển thị đủ, giữ SVG chỉnh sửa được.

Chưa chạy: các thí nghiệm mở rộng ngoài core và tái lập 11 model đã nêu; không upload, không sửa slide/poster.

Bug đã biết: lỗi truy cập file tạm Windows gián đoạn chưa xác định nguyên nhân. Dừng sớm ở ngân sách thấp là hành vi của protocol hiện tại, không coi đó là tái lập paper chính xác.

Quyết định: giữ protocol main đã chốt và toàn bộ điểm bất lợi; mọi thay đổi cơ chế ngân sách sau đây thuộc biến thể mới.

Tôi sẽ lo: phạm vi core đã hoàn tất, bằng chứng và cấu hình lưu local.

Cần bạn: không có thông tin hay phê duyệt nào còn thiếu để hoàn thành phạm vi này.
