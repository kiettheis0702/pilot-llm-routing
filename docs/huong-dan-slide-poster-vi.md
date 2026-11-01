# Hướng dẫn đọc slide và poster — Team 8 (bản tiếng Việt)

Cập nhật 31/10/2026 theo [`CLAUDE_UPDATE_OPENAI_2026-10-31.md`](CLAUDE_UPDATE_OPENAI_2026-10-31.md). File này giải thích **bằng tiếng Việt** toàn bộ nội dung của hai file nộp (slide viết bằng tiếng Anh):

| File | Nội dung | Ghi chú |
| --- | --- | --- |
| `midterm/8.pptx` | Deck giữa kỳ, **24 slide**, notes tiếng Anh ở mọi slide (275–497 từ/slide) | Số liệu chính nay lấy từ nghiên cứu **OpenAI text-embedding-3-small**; BGE chỉ còn ở chỗ có nhãn "historical BGE" |
| `poster/8.pptx` | Poster **1 slide A2 dọc 420 × 594 mm** | W1 vẫn là thí nghiệm tổng hợp (synthetic); W2 nay dùng số OpenAI |
| `midterm/8-preview.pdf`, `poster/8-preview.pdf` | Bản PDF xuất bằng PowerPoint để xem nhanh | Không nộp; chỉ để duyệt |
| `midterm/references.txt` | Danh sách 13 tài liệu IEEE của deck | Đối chiếu số [n] ở chân slide |
| `docs/presentation-update-2026-10-31.md` | Nhật ký thay đổi + QA của lượt cập nhật này | Nguồn từng con số, hash file, việc chưa kiểm |

Bản trước khi cập nhật (22 slide, số BGE) được giữ nguyên byte trong `outputs/presentation-backup-20261031-bge/` (thư mục local, không lên Git).

Đọc theo thứ tự: mục 0 (thay đổi lần này) → mục 1 (vì sao làm như vậy) → mục 3 (từng slide) → mục 4 (poster) → mục 5 (thuật ngữ) khi gặp từ lạ.

---

## 0. Lần cập nhật 31/10 thay đổi gì (đọc trước)

1. **Embedder**: trước đây nhóm dùng BGE-M3 chạy local thay cho embedder của paper. Nay đã có vector **thật** của chính embedder paper dùng (`text-embedding-3-small`, 1.536 chiều, toàn văn) từ một lượt gọi API đã được duyệt trước đó (437 request; giá tính theo bảng giá khoảng 0,137 USD trước thuế, không phải hóa đơn). Mọi thí nghiệm mới **dùng lại cache**, không gọi thêm API trả phí. Câu cũ "0 paid API calls" đã bị bỏ.
2. **Thước đo**: thay "service utility" tuyệt đối bằng **chất lượng / GPT-4** và **chi phí ghi nhận / GPT-4**, cùng mẫu số là toàn bộ câu được yêu cầu (câu không được phục vụ tính 0). Trục ngân sách gọi là **allocation** = tỉ lệ so với chi phí GPT-4 *ước lượng trên tuning*. Allocation 50% ≈ 49% chi phí GPT-4 *ghi nhận thật* trên RouterBench.
3. **Kết luận đổi**:
   - Câu "PILOT được lợi nhiều nhất khi học thêm" **không còn đúng** với số OpenAI và đã bị bỏ.
   - "Regret thấp nhất" chỉ còn đúng **trong nhóm learner thích nghi** (luôn gọi GPT-4 có regret thấp hơn).
   - Ablation có kết quả lẫn lộn, không nói "prior luôn giúp".
4. **Thêm 2 slide**:
   - slide 19: chẩn đoán bộ điều khiển ngân sách, giải thích vì sao ở allocation thấp luồng bị dừng;
   - slide 20: bản cục bộ của 6 nghiên cứu phụ trong paper.

   Deck từ 22 lên 24 slide, vẫn trong giới hạn 20–40.
5. **Link**: số OpenAI dẫn tới registry W&B mới (2.498 run, `reports/openai-publication-2026-10-31/experiment-map.csv`). Link cũ (1.128 run) chỉ dùng cho số BGE lịch sử và thí nghiệm W1 trên poster. QR ghi project có **3.626 run**.

---

## 1. Đề bài yêu cầu gì và đã đáp ứng thế nào

Nguồn: trang đề môn học [statement](https://hcmus-cs-ta.github.io/lab-book/intro-ml/project/statement/) và các trang con [reading](https://hcmus-cs-ta.github.io/lab-book/intro-ml/project/statement/reading), [reproduction](https://hcmus-cs-ta.github.io/lab-book/intro-ml/project/statement/reproduction), [analysis](https://hcmus-cs-ta.github.io/lab-book/intro-ml/project/statement/analysis), [empirics](https://hcmus-cs-ta.github.io/lab-book/intro-ml/project/statement/empirics) (đọc ngày 30/10/2026).

Điều quan trọng nhất của đề: **deck không có thuyết trình miệng**, giám khảo đọc slide cùng notes. Vì vậy notes mỗi slide được viết đầy đủ (lập luận, giả thiết, giới hạn), còn slide giữ hình, bảng, biểu đồ.

### Deck (45% điểm project)

| Yêu cầu của đề | Cách đáp ứng | Slide |
| --- | --- | --- |
| 20–40 slide, < 20 MB, notes ≤ 500 từ ở mọi slide | 24 slide, 0,25 MB, notes 275–497 từ | tất cả |
| Trích dẫn IEEE đánh số ở chân slide, không có slide References | Chân slide có [n]; số đánh theo thứ tự xuất hiện, script tự kiểm | tất cả |
| Hình do nhóm tự vẽ, không dán hình paper | Sơ đồ vẽ bằng shape; biểu đồ là chart gốc PowerPoint, sửa được | 2, 6, 8, 9, 10, 15–20 |
| Bối cảnh: bài toán + **ví dụ thực tế của nhóm** + ≥ 2 công trình trước | Chatbot hỗ trợ sinh viên HCMUS; LinUCB và Hybrid LLM | 2, 3 |
| Đóng góp theo lời paper, rồi nhóm lại lý thuyết / kỹ thuật / thực nghiệm | Dải "In the paper's terms" + 3 thẻ | 4 |
| Method 5 lớp | Mỗi lớp có slide riêng, kicker ghi "layer x of 5" | 5–12 |
| Kết quả đặt cạnh số của paper, **mỗi số dẫn tới run W&B** | Link s7 · s42 · s2026 dưới số chính; phụ lục bảng run | 15–21, 24 |
| Nêu phạm vi (full/core) và lý do | Core scope + phần phụ có nhãn "local adaptation" | 13 |
| So sánh paper vs nhóm, có độ lệch khi số so được | Bảng 7 dòng; ghi "not computed" khi không có điểm khớp | 21 |
| Giải thích mọi khoảng lệch | Bảng nguyên nhân + bằng chứng + trạng thái | 22 |

### Poster (50% điểm project)

| Yêu cầu của đề | Cách đáp ứng |
| --- | --- |
| 1 file pptx, đúng 1 slide, A2 dọc | 420 × 594 mm, kiểm bằng script và PowerPoint |
| Mỗi điểm yếu: lý thuyết đặt **cạnh** thí nghiệm, được "nối" với nhau | Mỗi khối W1/W2: Theory (trái) → mũi tên "predicts" → Experiment (phải) → Verdict |
| Chọn 2 điểm yếu nổi bật nhất | W1 cosine mất độ lớn; W2 ngân sách chỉ đúng với chi phí ước lượng |
| ≥ 3 điểm mạnh có bằng chứng, ≥ 2 giới hạn lý thuyết, ≥ 2 hướng phát triển gắn với giới hạn | Hàng cuối: S1–S3, Theory audit, F1→W1, F2→W2 |
| Số thực nghiệm dẫn tới run W&B | Link run dưới mỗi verdict + QR tới project |

### Hạn nộp (theo Moodle đọc 30/10, cần kiểm lại)

- **07/11/2026**: link GitHub public.
- **21/11/2026**: deck `midterm/8.pptx`.
- **05/12/2026**: poster `poster/8.pptx`.
- W&B Academic trial hết **29/11/2026**, *trước* hạn poster. Nếu link W&B ngừng hoạt động, QR trên poster sẽ chết. Cần giải quyết tài khoản trước 05/12.

---

## 2. Chuẩn poster đã tham khảo và áp dụng

Đề chỉ quy định A2 dọc, 1 slide, theory cạnh experiment. Phần thiết kế dựa trên các hướng dẫn phổ biến:

- **Lưới 2–3 cột** cho poster học thuật cỡ A2 trở xuống ([tổng hợp hướng dẫn](https://conceptviz.app/blog/academic-poster-design-complete-guide)). Poster dùng 2 cột cho hai khối điểm yếu và 3 cột ở hàng cuối.
- **Cỡ chữ**: hướng dẫn cho A0 thường là tiêu đề ≥ 72 pt, đầu mục ≥ 48 pt, thân ≥ 24 pt. A2 có cạnh bằng một nửa A0 và được đọc ở khoảng 1 m, nên poster dùng tiêu đề 54 pt, đầu khối 30 pt, thân 16 pt, chữ biểu đồ 13 pt, tài liệu 11 pt.
- **"Better Poster" của Mike Morrison**: đặt *phát hiện chính* thành một câu lớn ([giới thiệu](https://datadreaming.org/posts/2020-02-27-better-research-poster/2020-02-27-better-research-poster.html), [phê bình](https://betterposters.beehiiv.com/p/critique-the-morrison-billboard-poster-19-04-11)). Poster dùng bản thu gọn: dải **MAIN FINDING** một câu ngay dưới tiêu đề.
- **Font sans-serif cho thân, tối đa 2–3 họ font**: Calibri (thân), Cambria (tiêu đề), Cambria Math (công thức).
- **Màu có nghĩa, nhất quán ở cả deck và poster**: teal = PILOT/chất lượng; cam (amber) = chi phí/ngân sách và W2; đỏ san hô (coral) = rủi ro/giới hạn và W1; xám = baseline.

---

## 3. Giải thích từng slide của deck

Mỗi mục gồm **ý chính**, **giải thích**, và **câu dễ bị hỏi** (nếu có). Số liệu đều do script đọc trực tiếp từ file kết quả trong repo, không gõ tay. "pp" là **điểm phần trăm** (chênh lệch giữa hai tỉ lệ %), khác với "%" tương đối.

### Slide 1 — Trang bìa
Tên paper, nhóm, lớp. Hình bên phải: truy vấn `q` vào router `R`, router chọn 1 trong 8 mô hình (vòng tròn to ≈ đắt). Notes nêu phạm vi: **bản tái lập của sinh viên trên biến thể cục bộ đã khai báo**, gồm 8 mô hình, embedder của paper (vector từ cache), kết quả LLM phát lại từ RouterBench. Đây không phải code của tác giả.

### Slide 2 — Bài toán và ví dụ của nhóm
- **Ý chính**: mỗi truy vấn chỉ chọn 1 LLM, tổng chi phí cả tháng ≤ B, và chỉ biết chất lượng câu trả lời của mô hình đã chọn.
- **Ví dụ của nhóm**: chatbot hỗ trợ sinh viên HCMUS.
  - "Khi nào hết hạn đăng ký học phần?" thì mô hình rẻ là đủ.
  - "Vì sao gradient descent phân kỳ khi bước quá lớn?" thì cần mô hình mạnh.
- **Hỏi**: *Sao không hỏi cả 8 mô hình rồi chấm?* Vì tốn gấp 8 lần và cần người chấm, đúng thứ hệ thống muốn tránh.

### Slide 3 — Hai công trình trước
- **LinUCB**: mỗi arm có một mô hình tuyến tính (ridge) cộng bonus thăm dò. Hạn chế: bắt đầu từ 0 (*cold start*), không có ngân sách.
- **Hybrid LLM**: bộ phân loại có giám sát quyết định có dùng mô hình nhỏ không. Hạn chế: cần câu trả lời và điểm của **cả hai** mô hình cho mọi câu huấn luyện, chỉ 2 mô hình, không học tiếp.
- **PILOT**: prior từ dữ liệu sở thích, cập nhật kiểu LinUCB, chính sách ngân sách. Đổi lại có **giả thiết mới**, và poster kiểm chính các giả thiết đó.

### Slide 4 — Đóng góp của paper
Ba đóng góp theo lời paper, rồi nhóm lại:
- **Lý thuyết**: PI-OFUL.
- **Kỹ thuật**: pretraining 2 pha → LinUCB khởi tạo bằng prior → knapsack trực tuyến ZCL có chia bin.
- **Thực nghiệm**: 93% chất lượng GPT-4 với 25% chi phí.

### Slide 5 — Ký hiệu (lớp 1/5)
- **D** là chiều embedding (`text-embedding-3-small`: 1.536); **d** là chiều không gian routing (64).
- Thành công được đo bằng regret khi học và bằng phần thưởng trên mọi câu được yêu cầu **so với GPT-4** khi triển khai.
- Lý thuyết dùng phần thưởng **tuyến tính**, còn thuật toán dùng **cosine**. Đây là điểm yếu W1 của poster.

### Slide 6 — Pipeline (lớp 2/5)
- Ba làn: offline một lần (Arena → Pha 1 → Pha 2), học trực tuyến (không ngân sách), triển khai (router đóng băng + chính sách chi phí).
- Dải "Our run": `text-embedding-3-small` [5], vector thật 1.536 chiều của toàn văn → chiếu xuống 64 chiều, 8/11 mô hình, kết quả phát lại.
- **Hỏi**: *Vì sao tách học và triển khai?* Để ổn định và đổi ngân sách không phải học lại. Hệ quả: regret đo ở pha học, chất lượng đo ở pha triển khai.

### Slide 7 — Pretraining từ sở thích người (lớp 3/5)
- Pha 1 dùng triplet loss với khoảng cách cosine; Pha 2 dùng softmax 2 chiều + BCE.
- **Cấu hình**: 1.245 cặp huấn luyện (1.193 triplet), 30 + 40 epoch, lr 0,03, margin 0,2.
- **Độ chính xác hiệu chỉnh (calibration) 186 / 190 / 194 trên 313** (trung bình 60,70%).
  - Lưu ý: 313 cặp này dùng để đặt λ_a, nên đây là tập **hiệu chỉnh**, **không phải** tập kiểm tra độc lập.
  - BGE lịch sử ra 189/188/193, trùng trung bình.
- Giao thức phụ 847/211/500 (cho kết quả 63,53%) được ghi riêng ở slide 20; không trộn với 60,70%.

### Slide 8 — Học trực tuyến (lớp 3/5)
- Bốn bước: cosine; khởi tạo bằng prior; chọn arm theo cận trên; chỉ cập nhật arm đã chọn.
- Prior và λ_a lấy từ lượt fit OpenAI. λ_a = 1/(độ chính xác **calibration**) là heuristic.
- **Không có ngân sách khi học.** Biểu đồ cột chỉ là minh họa, không phải dữ liệu.

### Slide 9 — Tiêu ngân sách theo bin (lớp 3/5)
- Bước 3 nay ghi rõ phần nhóm cài thêm:
  1. **Guard**: arm đủ điều kiện còn phải có chi phí ước lượng ≤ tiền còn trong bin.
  2. Nếu không có arm nào, **fallback** sang arm có ĉ ≤ B_còn lại / Q_còn lại.
  3. Nếu vẫn không có, **dừng cả luồng** (quy tắc "Insufficient budget" của Algorithm 3).
- Ở MMLU allocation 20%, cả 3 seed dừng sau **29/1.275** câu dù còn ≥ 94% ngân sách ước lượng chưa tiêu.
- Bỏ guard (bản "paper-literal") cho kết quả **giống hệt** trên toàn lưới.

### Slide 10–11 — Lý thuyết và phác thảo chứng minh
Không đổi:
- prior gần θ* hơn gốc 0 thì cận nhỏ hơn;
- 4 giới hạn;
- 2 lỗi in (trang 23945, 23946), lỗi trình bày chứ không phải định lý sai.

**Không** được nói cosine trong thực tế thừa hưởng định lý tuyến tính.

### Slide 12 — Độ phức tạp (lớp 5/5)
- Bản của nhóm giải hệ dày: dự đoán O(K d³), trạng thái O(K d²) ≈ 260 KiB.
- Cập nhật hạng 1 (đưa về O(K d²)) **chưa cài**, nên không khẳng định.

### Slide 13 — Phạm vi
- **Core scope**: E1 + E2 + phân tích tự thiết kế (5 ablation, chẩn đoán bộ điều khiển, 2 thí nghiệm poster).
- Thêm **phần phụ có nhãn**: bản cục bộ của 6 nghiên cứu phụ.
- Vì sao không full: 3/11 mô hình không ghép được với dữ liệu RouterBench; không có code chính thức. Lý do "embedder trả phí" đã bỏ vì nay dùng đúng embedder.
- Ba con số ở góc:
  - **1.098 điều kiện OpenAI** (858 + 240), khác với 1.098 của BGE lịch sử;
  - 3 seed;
  - **0 request trả phí mới**.
- Notes ghi chi tiết lượt embedding cũ: 38.018 văn bản, 437 request, 6.867.235 token, khoảng 0,137 USD theo bảng giá, không phải hóa đơn.

### Slide 14 — Giao thức và thước đo
- Split không đổi. Tuning lại trên vector OpenAI vẫn chọn PILOT α = 1, LinUCB α = 5, EpochGreedy cửa sổ 10.
- Replay độc lập khớp toàn bộ **108 nhóm học** (78 main + 30 ablation).
- **Quality / GPT-4** = tổng phần thưởng ÷ tổng phần thưởng của GPT-4 trên **cùng các câu được yêu cầu**; câu không phục vụ tính 0.
- **Recorded cost / GPT-4** = chi phí ghi nhận của câu đã chọn ÷ chi phí GPT-4 trên cùng các câu.
- **Allocation** = ngân sách cấp, tính theo chi phí GPT-4 *ước lượng* trên tuning. Không nhầm với chi phí đã tiêu thật.

### Slide 15 — E1: PILOT thua ở 50%, chỉ dẫn đầu khi tiền dư

| Điểm đo | PILOT | Baseline so sánh |
| --- | --- | --- |
| RouterBench, 50% | **86,60%** chất lượng, chi phí 46,23% | LinUCB **91,28%**, chi phí chỉ 29,62% |
| MMLU, 50% | 83,53%, chi phí 47,53% | LinUCB 86,96%, chi phí 45,07% |
| RouterBench, 100% | 97,94%, chi phí 91,57% | LinUCB 96,59% |
| MMLU, 100% | 98,44%, chi phí 91,81% | EpochGreedy 95,43% |

- Ở RouterBench 50%, LinUCB vừa tốt hơn vừa rẻ hơn. Khoảng 3 seed không chồng nhau (PILOT 84,98–88,61%, LinUCB 90,22–91,93%).
- PILOT chỉ dẫn đầu ở 100% và 125% (RouterBench), và từ 75% trở lên (MMLU).
- Vùng trái: ở ≤ 10% các learner chỉ phục vụ ≤ 1,1% câu RouterBench. Ở MMLU 20–30%, PILOT chỉ phục vụ 2,27% / 27,45%. Random phục vụ hết nên cao hơn ở đó. Deck giữ nguyên, không cắt trục.
- **Hỏi**: *Dùng đúng embedder của paper rồi sao vẫn thua?* Xem slide 22: khoảng lệch không do embedder.

### Slide 16 — E2: regret

| Phương pháp | RouterBench | MMLU |
| --- | --- | --- |
| PILOT | **4.275** | **1.758** |
| EpochGreedy | 4.747 | 2.274 |
| LinUCB | 6.826 | 2.749 |
| ExploreOnly | 11.765 | 5.508 |
| Random | 11.760 | 5.526 |
| Luôn gọi GPT-4 (đường tím) | **3.675** | **1.416** |

- **Luôn gọi GPT-4** có regret *thấp hơn* PILOT, nhưng nó không học và bỏ qua chi phí. Vì vậy tiêu đề chỉ nói "**trong nhóm learner thích nghi**".
- Ở 10% đầu luồng MMLU, LinUCB thấp hơn PILOT (213 so với 264).
- Bốn điểm là **snapshot lồng nhau của một luồng**, không phải 4 lần chạy lại.

### Slide 17 — E2: học thêm gần như không làm PILOT tốt hơn (ở 50%)
- Từ 10% lên 100% dữ liệu học:
  - PILOT chỉ đổi **+0,49 pp** (RouterBench) / **+1,52 pp** (MMLU);
  - EpochGreedy tăng **+9,65 / +12,00 pp**.
- LinUCB cao nhất ở **mọi** mức dữ liệu.
- Câu cũ "PILOT được lợi nhiều nhất" bị bỏ.
- **Câu chắc chắn gặp**: *Regret thấp mà chất lượng không cao?* Regret đo khi học (không ngân sách, có thăm dò); chất lượng đo khi triển khai (điểm đóng băng + chính sách chi phí). Điểm cosine bị nén nên PILOT hay chọn mô hình đắt (ở 50% tiêu 46,2% chi phí GPT-4, LinUCB chỉ 29,6%).

### Slide 18 — Ablation (thay đổi chất lượng, điểm %, ở 50%)

| Biến thể | RouterBench | MMLU | Cách đọc |
| --- | ---: | ---: | --- |
| Prior only (không cập nhật trực tuyến) | −6,70 | −13,88 | Giảm mạnh nhất: phản hồi trực tuyến quan trọng |
| Zero prior directions | +2,14 | −0,45 | Trái dấu. Vẫn dùng phép chiếu và λ học từ sở thích, nên **không** phải bỏ hết thông tin sở thích |
| λ = 1 cho mọi arm | +1,55 | +0,32 | Heuristic 1/accuracy không được ủng hộ |
| Một bin | −0,35 | −0,62 | Nhỏ; **không** phải cách sửa lỗi dừng sớm |
| Chiếu 8 chiều | −0,90 | +1,65 | Lẫn lộn |

Mọi biến thể đều phục vụ 100% câu ở mức này. Ba seed chỉ mang tính mô tả, không kiểm định thống kê.

### Slide 19 — (mới) Bộ điều khiển ngân sách: luồng dừng khi vẫn còn tiền
- **Chẩn đoán** (MMLU 20%, cả 3 seed):
  1. bin gần tiêu hết nên ngưỡng không nhận arm nào;
  2. phần fallback (tiền bin ÷ số câu còn lại trong bin) nhỏ hơn giá ước lượng của mô hình rẻ nhất;
  3. Algorithm 3 dừng: phục vụ 29/1.275 câu, còn ≥ 94% ngân sách ước lượng.
- **Biến thể (phần mở rộng cục bộ, cùng điểm số PILOT đóng băng)**:

  | Allocation | Chính sách | Chất lượng / GPT-4 | Coverage |
  | --- | --- | ---: | ---: |
  | 20% | Guarded stop (main) | 1,72% | 2,27% |
  | 20% | Guarded skip (bỏ qua câu đó, đi tiếp) | 68,90% | 97,31% |
  | 20% | Fixed per query (mỗi câu ≤ B/Q) | 68,94% | 99,92% |
  | 30% | Guarded stop (main) | 20,17% | 27,45% |
  | 30% | Guarded skip | 75,29% | 99,87% |

- Fixed per query dùng kém ngân sách lớn: 91,08% ở 100%, so với 98,44% của main.
- Một bin không chữa được: 15,18% ở 20%.
- Bộ so sánh "retrospective P − λC" nhìn được điểm dự đoán và chi phí ước lượng **tương lai** (không phải phần thưởng thật). Nó có nhiều thông tin hơn, và không được gọi là tối ưu.
- **Không thay** kết quả main. Đây là phần mở rộng có nhãn.

### Slide 20 — (mới) Bản cục bộ của các nghiên cứu phụ trong paper
- **Biểu đồ: Hybrid LLM gốc và PILOT**, cặp GPT-4/Llama-2 đã xác minh, seed 42.
  - Hybrid ở ngưỡng 0,5: **81,67%** chất lượng với **23,12%** chi phí.
  - PILOT ở allocation 50%: **72,49%** với **46,16%**.
  - Hai số "0,5" là **hai thứ khác nhau** (ngưỡng xác suất và tỉ lệ ngân sách), không phải cùng ngân sách.
  - Hybrid được học từ phần thưởng của **cả hai** mô hình.
  - Giữ cả đường cong, không nội suy điểm giả.
- **Binary bandit** (App. D.4, 3 seed, 50%):
  - chất lượng: PILOT 72,45%, LinUCB 72,09%, EpochGreedy 70,81%;
  - regret: PILOT **879**, LinUCB 729, EpochGreedy 731, luôn GPT-4 719. Regret này so với oracle 2 arm, không so với số 8 arm.
  - Kết luận: **không có chuyện PILOT luôn thắng**.
  - Luôn GPT-4 chỉ phục vụ 0,96% câu ở 50% (dừng sớm), nên chất lượng triển khai thấp **không phải** độ chính xác của GPT-4.
- **Task shift** MMLU → GSM8K (seed 42, không ngân sách):
  - trước khi đổi: PILOT 80,52% so với Hybrid tĩnh 79,58%;
  - sau khi đổi: 65,61% so với 65,81%.
  - Không nói PILOT thắng sau khi đổi, cũng không nói định tuyến tĩnh thất bại.
- **Instructor-XL** (Sec. 5.4), so với OpenAI:
  - +2,58 / +0,55 pp ở 50%;
  - −0,60 / −0,23 pp ở 100%.
  - Đây là so sánh "công thức" (đổi tokenizer, giới hạn 512 token, tuning lại), không phải bằng chứng nhân quả của riêng encoder.
- **Dự đoán sở thích** (App. D.3, split cục bộ 847/211/500): RouteLLM-MF **68,07%** so với PILOT **63,53%**, tức **ngược** thứ hạng trong paper. Ba seed dùng chung 500 cặp, không phải 1.500 mẫu độc lập.
- **Nhiễu phản hồi** (App. D.5, MMLU): thay 0/1/3/5% phản hồi học bằng giá trị ngẫu nhiên cho 68,86 / 68,76 / 68,47 / 68,55% phần thưởng (−0,31 pp ở 5%). Đây là xác suất *thay*, không phải xác suất *lật nhãn*.

### Slide 21 — Paper và bản tái lập
| Mục | Kết luận |
| --- | --- |
| Headline 93%@25% (RouterBench) | Điểm đo gần nhất: 77,81% @ 19,23% chi phí (alloc. 20%) và 80,68% @ 28,57% (alloc. 30%). **Không nội suy** điểm 25%, **không tính** sai số tái lập (8 vs 11 mô hình) |
| Headline 86%@27% (MMLU) | Không có điểm phủ đủ câu gần 27% (dừng sớm ở 20–30%); 83,53% @ 47,53% ở 50% |
| PILOT vs baseline bandit tốt nhất | **Một phần**: −4,68 / −3,44 pp ở 50%; +1,35 / +3,02 pp ở 100% |
| Regret | **Tái lập trong nhóm learner**; luôn-GPT-4 thấp hơn |
| Dự đoán sở thích (App. D.3) | **Ngược thứ hạng** cục bộ: MF 68,07% > PILOT 63,53% |
| Embedder (Sec. 5.4) | Nhất quán: Instructor-XL đổi +2,58 / +0,55 pp ở 50% |
| Dữ liệu | Cùng 36.497 câu; 86 nhãn task (paper 64); 8 mô hình |

### Slide 22 — Vì sao số khác
- **PILOT thua ở 20–75%**:
  - Dùng embedder của paper mà **khoảng lệch vẫn tồn tại**. Điều này không ủng hộ giả thuyết “chỉ do BGE” trong phép đối chiếu đã chạy; chưa loại trừ tương tác embedding, tuning, split và pool.
  - OpenAI cũng **không tốt hơn đều** BGE: trung bình −0,33 pp (RouterBench), −2,76 pp (MMLU). Phép so này còn đổi cả toàn văn/cắt 13 input và tuning lại, nên không tách riêng encoder.
  - Nguyên nhân còn lại (8 mô hình, 3 phiên bản chưa xác minh, 86 vs 64 nhãn, split/cấu hình tác giả) **chưa được phân định** vì thiếu đầu vào/thiết lập tác giả tương ứng.
- **60,70% vs 65,0%**: **không so được** (313 cặp hiệu chỉnh vs 500 cặp kiểm tra).
- **Heuristic λ**: đã kiểm, không được ủng hộ.
- **Sụp ở allocation thấp**: đã **giải thích** bằng quy tắc dừng (slide 19).
- **Thiết lập tự chọn**: kiểm một phần.
- **Seed**: khoảng lệch vẫn tồn tại trên **ba seed đã thử** (min/max không chồng). Đây không phải khoảng tin cậy và không loại trừ mọi seed khác.
- **Code gốc**: không kiểm được.

Ghi chú rà cuối: slide 22 trong PPTX đầu vào còn dùng các nhãn khẳng định mạnh
hơn ở trên. [Yêu cầu cho Claude](CLAUDE_FINAL_REVIEW_2026-10-31.md) đã chỉ rõ
câu cần thay. Independent replay chứng minh trace của cấu hình đã chạy;
không tự chứng minh mọi phần cứng/thư viện đều cho checkpoint giống hệt.

### Slide 23 — Những gì được và không được khẳng định
- **Tái lập, có giới hạn**: regret thấp nhất trong nhóm learner; dẫn đầu chất lượng khi tiền dư (97,94% / 98,44% ở 100%) nhưng tiêu ~92% chi phí GPT-4.
- **Không tái lập**: thua LinUCB ở 50% và tiêu nhiều hơn; không có điểm khớp 93%@25%; embedder của paper không xóa khoảng lệch; RouteLLM-MF dự đoán sở thích tốt hơn.
- **Kiểm tiếp ở poster**: cosine, chi phí.
- QR: GitHub và W&B (3.626 run).

### Slide 24 — Phụ lục: run W&B của từng số chính
Bảng 8 số × 3 seed, mỗi ô là link run trong **registry mới**. Cách tự kiểm: mở 3 link dòng đầu (RouterBench PILOT 50%) và so với giá trị từng seed trong `reports/openai-core-2026-10-31/per-seed.csv` (84,98%, 86,20%, 88,61%); trung bình là 86,60%. Regret lấy từ `learning-prefixes.csv`.

---

## 4. Giải thích poster

**Tiêu đề**: "Where PILOT's guarantees stop", tức các bảo đảm của PILOT dừng ở đâu.

**MAIN FINDING**: cosine xóa chênh lệch phần thưởng gấp 4 lần (regret 600 so với 1,2), và sai số chi phí 10% đẩy **23/48** lần chạy vượt ngân sách (số OpenAI; BGE cũ là 25/48).

**Ô "Our reproduction, in short"**: regret thấp nhất trong nhóm learner (4.275 so với 4.747); ở allocation 50% PILOT 86,6% so với LinUCB 91,3% chất lượng GPT-4; 8/11 mô hình, `text-embedding-3-small`, 3 seed.

### W1 — Cosine bỏ mất độ lớn của phần thưởng (không đổi)
- Lý thuyết, thí nghiệm và kết quả (600 vs 1,2; 9/9 cấu hình) giữ nguyên.
- Đây là thí nghiệm **tổng hợp (synthetic)**, không phụ thuộc embedder. **Không** được gọi nó là thí nghiệm OpenAI.
- Link run trỏ tới bản ghi lịch sử nhóm `lab2-critique`.

### W2 — Ngân sách chỉ khả thi với chi phí ước lượng (số OpenAI)
- **Lý thuyết** không đổi: ZCL cần biết trọng số thật; PILOT chỉ có ĉ; tổng chi = Σĉ + Σĉ·e.
- **Thí nghiệm**: 48 điều kiện PILOT chính học đủ của nghiên cứu OpenAI. Nhân chi phí ghi nhận với f ∈ {0,8; 1; 1,1; 1,25; 1,5; 2}, **giữ nguyên lựa chọn**, không định tuyến lại.
- **Kết quả**:
  - f = 1: **0/48**;
  - f = 1,1: **23/48** (48%), so với 10% của toàn bộ 858 điều kiện core.
  - Lý do PILOT dễ vượt hơn: nó tiêu sát ngân sách (trung vị **86%** allocation, so với 16% chung).
  - BGE lịch sử: 0 → 25/48, cùng xu hướng, có nhãn riêng.
- **Giới hạn phải nói**:
  - đây là số **điều kiện thí nghiệm**, không phải số truy vấn hay xác suất;
  - không có nhà cung cấp nào bị tính tiền;
  - ở f = 1, không điều kiện core nào vượt, nhưng 7/240 điều kiện ablation có vượt.

### Hàng cuối
- **Điểm mạnh**:
  - S1: regret thấp nhất trong nhóm learner;
  - S2: prior từ dữ liệu miễn phí, **60,7% độ chính xác calibration** so với 50% đoán mò;
  - S3: 0/48 lần vượt.
- **Theory audit**:
  - λ = 1 tốt hơn (+1,55 / +0,32 pp);
  - O(K d²) **có thể đạt** với cập nhật hạng 1, còn bản của nhóm là O(K d³) giải hệ dày.
- **Hướng phát triển**: F1 → W1, F2 → W2, vẫn chỉ là đề xuất. Biến thể "guarded skip" của slide 19 sửa chuyện dừng sớm, không phải sai số chi phí, nên không đưa vào W2.

---

## 5. Thuật ngữ Anh – Việt

| Thuật ngữ | Nghĩa |
| --- | --- |
| Routing / router | Chọn mô hình nào trả lời một truy vấn / bộ chọn |
| Arm | Một lựa chọn trong bài toán bandit; ở đây là một LLM |
| Contextual bandit | Bandit có ngữ cảnh: mỗi vòng thấy vector của câu hỏi rồi mới chọn arm |
| Bandit feedback | Chỉ quan sát phần thưởng của arm đã chọn |
| Regret (cumulative, oracle) | Tổng phần thưởng mất so với việc luôn chọn arm tốt nhất cho từng câu |
| UCB, optimism | Chọn theo cận trên: ước lượng + bonus thăm dò |
| Prior, preference prior | Vector mô hình học từ dữ liệu so sánh của người dùng |
| Calibration pairs | Cặp dùng để hiệu chỉnh (đặt λ_a); không phải tập kiểm tra độc lập |
| Quality / GPT-4 | Tổng phần thưởng ÷ tổng phần thưởng của GPT-4 trên cùng các câu được yêu cầu |
| Recorded cost / GPT-4 | Chi phí ghi nhận ÷ chi phí GPT-4 trên cùng các câu |
| Allocation | Ngân sách cấp, tính theo % chi phí GPT-4 *ước lượng* trên tuning |
| Coverage | Tỉ lệ truy vấn được phục vụ |
| pp (percentage points) | Điểm phần trăm: hiệu giữa hai tỉ lệ % |
| Guarded stop / skip | Bộ điều khiển chính (dừng luồng) / biến thể bỏ qua câu rồi đi tiếp |
| Knapsack (online) | Bài toán cái túi với món đến lần lượt |
| Bin, rollover | Chia luồng thành nhóm; ngân sách thừa chuyển sang nhóm sau |
| Ablation | Bỏ hoặc thay một thành phần để xem ảnh hưởng |
| Counterfactual | Phản thực tế: tính lại kết quả khi đổi một yếu tố, giữ phần còn lại |
| Confounded | Hai nguyên nhân đổi cùng lúc nên không tách được |
| Local adaptation | Bản làm lại cục bộ theo ý tưởng của paper, không phải bản của tác giả |
| Seed | 3 seed cho khoảng min–max, **không phải** khoảng tin cậy |

---

## 6. Lịch sử thay đổi

- **30/10**: 33 → 22 slide; thêm ví dụ thực tế, bảng tiền nhiệm, đóng góp 3 loại, độ phức tạp, phân tích sai lệch, link W&B từng số; chuyển phê bình sang poster.
- **31/10** (bản này): đổi toàn bộ số kết quả sang nghiên cứu OpenAI; thêm slide 19–20; sửa các khẳng định cũ theo mục 0; cập nhật poster W2, các ô tóm tắt, QR và tài liệu tham khảo; thêm 3 tài liệu IEEE ([5] OpenAI embedding, [12] RouteLLM, [13] Instructor). Chi tiết và hash file: `docs/presentation-update-2026-10-31.md`.

---

## 7. Cách dựng lại và tự kiểm

```bash
cd tools/course/pptx
npm ci                      # cài pptxgenjs, qrcode, jszip đúng phiên bản (lần đầu)
npm run build               # tạo midterm/8.pptx, poster/8.pptx, midterm/references.txt
cd ../../..
python tools/course/pptx/check_pptx.py   # số slide, kích thước, notes ≤ 500 từ, link W&B nằm trong 2 registry, không còn câu đã lỗi thời
```

Xuất PDF xem trước bằng PowerPoint thật (Windows):

```powershell
pwsh -File tools/course/pptx/render.ps1 -Pptx midterm/8.pptx -OutDir outputs/preview/midterm -Pdf midterm/8-preview.pdf
pwsh -File tools/course/pptx/render.ps1 -Pptx poster/8.pptx -OutDir outputs/preview/poster -Pdf poster/8-preview.pdf -Width 3300
```

Muốn sửa chữ thì sửa trong `build_deck.mjs` / `build_poster.mjs` rồi dựng lại. Script tự dừng nếu số liệu nguồn thay đổi làm một câu khẳng định không còn đúng (ví dụ LinUCB không còn là baseline tốt nhất ở 50%). Sửa tay trong PowerPoint cũng được, nhưng lần dựng lại sau sẽ ghi đè.

## 8. Việc nhóm còn phải làm

1. Mở hai file bằng PowerPoint **trên máy sẽ dùng khi bảo vệ**, kiểm font (Calibri/Cambria/Cambria Math), chart, notes và link. Bản này đã render và xuất PDF bằng PowerPoint trên máy làm việc; chưa kiểm trên máy khác.
2. Đọc notes từng slide, nhất là slide 15–22. Mỗi người phải giải thích được vì sao "regret thấp nhất" chỉ đúng trong nhóm learner, và vì sao 60,70% không so trực tiếp với 65,0%.
3. Nộp link GitHub trước **07/11**; deck trước **21/11**; poster trước **05/12**. Kiểm lại hạn trên Moodle. Không coi việc code đã vào `main` là đã nộp.
4. Giải quyết tài khoản W&B trước khi trial hết hạn **29/11**, rồi quét lại QR trên poster.
5. In thử poster A2 (hoặc xem PDF ở 100%) để kiểm độ dễ đọc ở khoảng cách 1 m.
