# Bàn giao cho Claude: slide và poster PILOT — Team 8

> **TÀI LIỆU LỊCH SỬ — BGE / mốc 29–30/10/2026.** Các số liệu, trạng thái “hiện tại/current”, nhánh Git và việc còn chờ bên dưới chỉ đúng tại mốc ghi nhận. Không dùng làm kết quả OpenAI hiện hành. Đọc [trạng thái OpenAI](openai-followup-status.md), [bản giao việc hiện hành cho Claude](CLAUDE_UPDATE_OPENAI_2026-10-31.md) và [khôi phục từ checkout sạch](openai-reproduce.md). Giữ nội dung cũ để truy vết, không ghi đè bằng kết quả mới.

**Cập nhật ngày 30/10/2026.** Đây là bản bàn giao lịch sử BGE ngày 30/10, từng được ưu tiên hơn các ghi chú “pending/chưa có W&B” trong tài liệu lịch sử. Người dùng giao Claude làm slide/poster; không cần chạy lại thí nghiệm, mua API hoặc sửa số liệu để làm đẹp kết quả.

## 1. Việc Claude cần làm

Đọc tài liệu này và các nguồn bên dưới, rồi thiết kế hai file PowerPoint có thể chỉnh sửa:

1. **Midterm:** `midterm/8.pptx`, 20–40 slide tính cả appendix, dưới 20 MB, speaker notes ở **mọi slide**, mỗi slide không quá 500 từ notes.
2. **Poster:** `poster/8.pptx`, đúng **một slide A2 dọc, 420 × 594 mm**.

Hai file trùng tên theo yêu cầu môn nên để trong hai thư mục khác nhau. Dùng biểu đồ/sơ đồ tự dựng, chữ và biểu đồ chỉnh sửa được; trích dẫn IEEE đánh số ở chân slide liên quan, **không làm slide tài liệu tham khảo riêng**. Thuyết trình bằng tiếng Việt, giữ thuật ngữ khoa học cần thiết bằng tiếng Anh. Có thể đổi phong cách thiết kế; không đổi định nghĩa metric, dữ liệu, phạm vi hay kết luận.

Đã có PPTX tham khảo trong repo: `deliverables/midterm/8.pptx` (33 slide) và `deliverables/poster/8.pptx` (1 slide A2). Chúng đã qua kiểm cấu trúc, số liệu và render; chưa mở kiểm bằng Microsoft PowerPoint thật, còn dòng trạng thái W&B cũ. Không coi đây là bản mới đã được người dùng duyệt. Không dùng `research-poster.pdf` làm mẫu kích thước nộp: PDF đó là **A1 ngang**.

## 2. Link chính thức và nguồn cần đọc

| Mục | Link |
| --- | --- |
| GitHub, nhánh nộp `main` | https://github.com/kiettheis0702/pilot-llm-routing |
| W&B công khai, 1.128 run | https://forge.coreweave.com/wandb/kiettheis-university-of-science-vnu-hcm/pilot-llm-routing-team8 |
| Artifact kết quả, 19 file | https://forge.coreweave.com/wandb/kiettheis-university-of-science-vnu-hcm/pilot-llm-routing-team8/artifacts/evaluation-results/pilot-team8-verified-results/v0 |
| Mapping experiment → script → run URL | https://github.com/kiettheis0702/pilot-llm-routing/blob/main/reports/publication-2026-10-30/experiment-map.csv |
| Bằng chứng public, đọc không cần key | https://github.com/kiettheis0702/pilot-llm-routing/blob/main/reports/publication-2026-10-30/public-verification.json |
| Paper cuối cùng và metadata | https://aclanthology.org/2025.findings-emnlp.1301/ |
| PDF paper cuối cùng | https://aclanthology.org/2025.findings-emnlp.1301.pdf |
| Website tác giả | https://ast-fri.github.io/contextual-bandit-routing-of-llms/ |
| Đề môn học | https://hcmus-cs-ta.github.io/lab-book/intro-ml/project/statement/ |
| Moodle lớp, cần tài khoản sinh viên | https://courses.ctda.hcmus.edu.vn/course/view.php?id=5001 |

Để lấy nội dung file trên GitHub, thay `https://github.com/kiettheis0702/pilot-llm-routing/blob/main/` bằng `https://raw.githubusercontent.com/kiettheis0702/pilot-llm-routing/main/`.

Đọc các file dưới đây theo thứ tự; đường dẫn tính từ gốc repo:

1. `docs/course-paper-analysis.md`: phân tích paper, định lý, giả thiết và hai phê bình chính.
2. `docs/course-defense-guide-vi.md`: câu hỏi bảo vệ và câu trả lời.
3. `reports/full-experiments-2026-10-29/README.md`: thí nghiệm đầy đủ, giới hạn và bảng kết quả.
4. `reports/course-submission-2026-10-30/slide-content.json`: số liệu đầy đủ cho 8 nhóm biểu đồ, 128 điểm aggregate; có source IDs. Trạng thái publication trong file frozen là lịch sử, dùng link hiện tại ở trên.
5. `reports/course-submission-2026-10-30/analysis/analysis.json`: stress cosine, cost sensitivity và so sánh với paper.
6. `docs/course-reproduce.md`: lệnh và giới hạn tái lập; `docs/wandb-upload-2026-10-30.md`: kiểm chứng upload.

Không lấy trạng thái ban đầu trong `docs/course-requirements-audit-2026-10-30.md`, ghi chú 24/10 trong `CONTEXT.md`, hoặc chữ “upload pending” trong gói frozen làm trạng thái hiện tại.

## 3. Nhóm, paper và mục tiêu

- Môn Introduction to Machine Learning, lớp 24KHMT1, HCMUS — University of Science, VNU-HCM.
- **Team 8: Thái Kiệt, Trần Lê Hải.** Không tự bịa MSSV, tên giảng viên hoặc phân công cá nhân.
- Paper đã được đăng ký/chấp nhận: **Adaptive LLM Routing under Budget Constraints**.
- Pranoy Panda, Raghav Magazine, Chaitanya Devaguptapu, Sho Takemori, Vishal Sharma; Findings of EMNLP 2025, pp. 23934–23949.
- Dùng bản ACL cuối cùng cho số trang, phương trình và trích dẫn; bản arXiv cũ có thể khác. BibTeX trên website tác giả từng có tiêu đề không khớp, lấy metadata từ ACL.
- Đây là **student reproduction / declared local variant**, không phải code chính thức của tác giả. Chưa tìm được link implementation chính thức trong nguồn đã kiểm ngày 30/10; không khẳng định code tác giả không tồn tại.

Bài toán: mỗi truy vấn chọn một LLM có chất lượng tốt với ngân sách giới hạn; chỉ phản hồi của model được chọn được dùng để học. Dùng preference prior để giảm cold start, contextual bandit để cập nhật, rồi budget controller cho giai đoạn triển khai.

Luồng cần tự vẽ:

`query → BGE-M3 embedding → learned projection → preference-informed prior → selected-feedback contextual bandit → freeze scores → budget bins + rollover → quality / coverage / recorded cost`

Hai giai đoạn phải phân biệt: online learning cập nhật từ selected-arm feedback; deployment đóng băng router và dùng cơ chế ngân sách. Bộ evaluator có outcome của tất cả arm để tính empirical oracle regret; learner không được đọc chúng để chọn arm.

## 4. Phạm vi và dữ liệu thực tế

| Hạng mục | Bản local đã làm | Giới hạn so với paper |
| --- | --- | --- |
| Model routing | **8 model được chấp nhận** | Paper dùng 11; còn 3 cầu nối alias Claude-v2/Mistral/Mixtral chưa đủ bằng chứng |
| Embedding | **BGE-M3 local**, 1.024 chiều → projection 64D | Paper dùng text-embedding-3-small; 8D chỉ là ablation |
| RouterBench 0-shot | 36.497 query; 401.467 outcome từ 11 model raw | 86 nhãn task raw so với 64 trong paper; bằng số query không chứng minh cùng population |
| Routing split | tuning 1.000 / learning 32.270 / deployment 3.227 | Tuning-only cho hyperparameter và cost estimate |
| Preferences | 1.558 cặp Arena đủ điều kiện, train 1.245 / holdout 313 | Không trùng tập 500 cặp paper |
| Seeds | 7, 42, 2026 | Báo mean/min/max, không gọi là CI hay significance |
| Prior production | projection 30 epochs, prior 40 epochs, 64D | Không lấy lượt development 2/5 epochs, 8D làm thí nghiệm chính |
| Token limit | 2.048; routing không truncate; preferences 8 train + 5 holdout truncate | Ghi giới hạn, không nói tất cả input nguyên vẹn |
| MMLU | Subset nằm trong RouterBench | **Không phải tập tái lập độc lập** |

Tất cả chất lượng/cost của LLM là dữ liệu benchmark có sẵn; không gọi 8/11 LLM thật, không sử dụng Claude Max/ChatGPT subscription để sinh các outcome này. Embedding chạy local, không mua API. “Cost” trong kết quả là đơn vị benchmark lịch sử, không phải hóa đơn thật.

Tám arm đúng theo `configs/frozen_experiment_protocol.json`: `WizardLM/WizardLM-13B-V1.2`, `claude-instant-v1`, `claude-v1`, `gpt-3.5-turbo-1106`, `gpt-4-1106-preview`, `meta/code-llama-instruct-34b-chat`, `meta/llama-2-70b-chat`, `zero-one-ai/Yi-34B-Chat`. Đây là model lịch sử trong benchmark, không đổi nhãn thành model Claude/GPT hiện hành. MMLU giữ nguyên split lồng: tuning 398 / learning 12.369 / deployment 1.275.

## 5. Những gì đã hoàn thành

- Pipeline offline, Random, 8 fixed-arm baselines, LinUCB, ExploreOnly, EpochGreedy và PILOT; pretraining preference, checkpoint/readback, cost estimate tuning-only, budget deployment.
- Cache BGE-M3 cho toàn bộ routing và preference input; sáu prior 64D/8D × ba seed được đọc lại kiểm độc lập.
- **1.098 deployment conditions = 858 chính + 240 ablation**; 108/108 policy/variant groups đã replay độc lập.
- **1.128 W&B runs = 1.098 + 18 synthetic policy records (9 cấu hình × 2) + 12 cost-accounting records (6 factors × 2 populations)**.
- Toàn bộ 1.128 run đã finished; 13.314 metric được đối chiếu local và đọc lại không đăng nhập. Anonymous chỉ có quyền đọc.
- Artifact W&B v0: 19 file / 7.167.970 byte, gồm tám hình, báo cáo nghiên cứu chín trang, research poster và dữ liệu aggregate; tên/kích thước/digest đã kiểm sau upload.
- Gói research frozen 184 file; các gói frozen giữ nguyên bytes/checksum. Bản mapping public mới nằm riêng ở `reports/publication-2026-10-30/`.
- Bộ test trước phát hành: 490 đạt, một Windows symlink test skip. Toy pipeline và RouterBench 64 dòng đã smoke tuần tự ở lượt trước; smoke không phải số liệu paper.
- GitHub `main` là nhánh đầy đủ dùng để bàn giao. **Chưa nộp Moodle.**

W&B ghi **historical aggregate imports**, không phải 1.128 lượt gọi model mới. Mỗi run có một dòng history aggregate; không bịa chuỗi loss/metric theo epoch từ những dòng này. Biểu đồ chính dùng CSV frozen hoặc `slide-content.json`.

## 6. Thí nghiệm và kết quả phải trình bày trung thực

### Thí nghiệm A — ngân sách và chất lượng

Ngân sách tương đối `.05, .1, .2, .3, .5, .75, 1, 1.25` của **local reference budget**. So sánh cùng split, seed, feedback và evaluator. Service utility = tổng reward / tổng số yêu cầu, yêu cầu không phục vụ tính 0. Không nhầm với accuracy chỉ trên phần đã phục vụ.

Local reference = mean estimated tuning-query cost của `gpt-4-1106-preview` × số query deployment. Nó không phải chi phí GPT-4 thật đã chi trên deployment; không đổi tên trục thành “% actual GPT-4 cost”.

Ở full learning và **0,5 local reference budget**, mean ba seed:

| Population | PILOT | Đối chiếu tốt hơn |
| --- | ---: | ---: |
| RouterBench | **0,6764** | LinUCB **0,7056** |
| MMLU subset | **0,6675** | ExploreOnly **0,6899** |

PILOT không tốt nhất trong hai điểm headline local này. Không giấu kết quả bất lợi. Các điều kiện PILOT headline trên phục vụ đủ yêu cầu; vẫn báo coverage và cost riêng trên toàn grid.

Paper headline: RouterBench đạt **93% hiệu năng GPT-4 ở 25% chi phí GPT-4**; MMLU **86% ở 27%**. Đây là tỷ lệ so với GPT-4, **khác mẫu số với local service utility**. Không trừ 0,6764 khỏi 0,93, không dựng phần trăm sai số tái lập từ hai đại lượng này.

### Thí nghiệm B — quy mô learning và regret

Prefix `.1, .25, .5, 1` từ một stream liên tục, snapshot bất biến; tính cumulative empirical oracle regret và frozen deployment tại nửa ngân sách. Không gọi đây là bốn lần độc lập restart hoặc regret trên deployment.

### Ablation

Năm biến thể: `zero-prior-direction`, `unit-initialization-strength`, `prior-only`, `single-budget-bin`, `projection-dimension-8`. Lấy giá trị từ `slide-content.json` / `paired-deltas.csv`; không suy ra số từ hình raster.

### Preference holdout

Ba seed: **189/313, 188/313, 193/313**, mean **60,7029%**. Paper báo PILOT 65%, RouteLLM 63,6% trên 500 cặp khác. Chênh mô tả với 65% là −4,2971 điểm phần trăm, **không phải matched reproduction error**. Không coi preference accuracy là chứng minh prior parameter gần optimum.

## 7. Hai phê bình chính cho poster

### A. Cosine làm mất thông tin độ lớn

Lý thuyết PI-OFUL có mô hình reward tuyến tính. Practical PILOT dùng cosine score nên không tự động được hưởng toàn bộ bảo đảm ấy.

Stress có `x=[1,0]`, reward hai arm `0,2 / 0,8`, oracle priors `(0,2,0) / (0,8,0)`, λ=1. Cosine mean của cả hai bằng 1; linear mean giữ độ lớn. Hai policy bắt đầu cùng A,b và chỉ cập nhật arm được chọn.

Tại **T=2.000, α=1**: cosine-PILOT regret **600**, linear-prior-control **1,2**. Có đủ chín cấu hình T∈{100,500,2000}, α∈{0,5;1;2}; không chọn duy nhất một điểm rồi giấu những điểm khác.

Kết luận đúng: minh họa rủi ro mất magnitude dưới stress này và khoảng cách giữa practical scoring với giả thiết tuyến tính. **Không phải phản ví dụ bác bỏ định lý OFUL có điều kiện.** Control có cùng prior, không gọi nhầm là LinUCB zero-prior thông thường.

### B. Ràng buộc ngân sách nhạy với sai số cost

Counterfactual nhân recorded costs theo `{0,8;1;1,1;1,25;1,5;2}`, **giữ nguyên lựa chọn model và reward**; không train/reroute lại.

Với main PILOT/full learning: factor 1 có **0/48** điều kiện vượt budget; factor 1,1 có **25/48**. Đây là số **điều kiện thí nghiệm**, không phải 52% query và không ước lượng xác suất vượt ngân sách ngoài thực tế.

Trên toàn grid gốc, mọi phương pháp: **12/1.098** điều kiện vượt recorded budget, mức vượt tối đa **0,127955416** đơn vị benchmark. Không nói bộ điều khiển bảo đảm hard budget cho mọi run.

Poster nên đặt hai khối sát nhau theo cấu trúc **lập luận → thiết kế kiểm chứng → hình → kết luận và giới hạn**, có link/QR thật đến W&B và GitHub.

## 8. Lý thuyết, điểm mạnh và hướng phát triển

Proposition 2.1 / Appendix C.1: prior-informed OFUL thay đại lượng liên quan `||theta*||` bằng `||theta_pref − theta*||`; có lợi khi prior gần hơn và giả thiết tuyến tính/noise thỏa. Không phải bảo đảm vô điều kiện cho practical cosine-PILOT.

Ý tưởng ridge:

`A = XᵀX + λI`, `b = Xᵀy + λ theta_pref`,

`theta_hat − theta* = A⁻¹Xᵀ epsilon − λA⁻¹(theta* − theta_pref)`.

Nếu bàn typo thiếu reward/inverse ở PDF cuối trang 12/13, phải đọc lại công thức trong final PDF và gọi là vấn đề trình bày đại số; không tự suy thành định lý sai. Accuracy preference không chứng minh khoảng cách tham số, heuristic λ từ accuracy không phải transfer certificate đã chứng minh.

Ít nhất ba điểm mạnh: tận dụng preference cho cold start; học chỉ với selected feedback; tách learning/deployment và đưa ngân sách vào quyết định. Ít nhất hai giới hạn lý thuyết: khoảng cách cosine/linear assumptions và cost uncertainty (có thể bổ sung điều kiện prior quality).

Độ phức tạp implementation dense hiện tại: prediction `O(Kd³)`, state `O(Kd²)`, update `O(d³)` do giải hệ trực tiếp. Rank-one inverse để prediction `O(Kd²)` là hướng tối ưu tương lai, chưa triển khai.

Hướng phát triển: magnitude calibration head; cost reserve theo quantile/hard bounds; kiểm matched 11-arm/embedding protocol khi mapping đủ bằng chứng. Phải ghi **đề xuất**, chưa có thí nghiệm chứng minh các cải tiến này hiệu quả.

Hai công trình tiền nhiệm cần so sánh động cơ/phương pháp, không bịa baseline local mới:

- Li et al., WWW 2010, contextual-bandit/LinUCB: https://arxiv.org/abs/1003.0146
- Ding et al., ICLR 2024, HybridLLM: https://openreview.net/pdf?id=02f3mUtqnM

## 9. Hình, bảng và đường dẫn source

Gốc `reports/full-experiments-2026-10-29/`:

- `aggregate/per-seed.csv`: đầy đủ từng điều kiện và seed.
- `aggregate/mean-min-max.csv`, `aggregate/paired-deltas.csv`, `aggregate/learning-prefixes.csv`: tổng hợp, paired delta và learning prefixes.
- `figures/figure-data.json`: dữ liệu các hình nghiên cứu.
- `figures/service-utility`, `learning-regret`, `learning-size`, `recorded-cost`, `coverage`, `ablations`: có PNG/PDF/SVG.
- `deliverables/research-report.pdf`: báo cáo chín trang, nguồn để đọc thêm.

Gốc `reports/course-submission-2026-10-30/`:

- `slide-content.json`: nguồn tiện nhất cho biểu đồ PowerPoint editable.
- `analysis/figures/magnitude-stress.png`, `analysis/figures/cost-sensitivity.png`.
- `analysis/analysis.json`: số liệu chính xác hai phê bình.
- `tracking-payloads.json`: config/metric/ID từng run. Ghép ID với **mapping public mới**, không dùng URL rỗng ở mapping frozen cũ.

Dùng các figure tự sinh trong repo hoặc tự vẽ lại từ CSV; không chụp figure paper rồi gọi là hình tự dựng. Mọi chart có tên trục, đơn vị, population, split/seed aggregation và mẫu số rõ.

## 10. Gợi ý bố cục 33 slide

1–4: cover, bài toán, notation, vị trí so với LinUCB/HybridLLM.
5–10: pipeline, preference training, prior, selected-feedback update, learning/deployment, budget bins/rollover.
11–15: novelty, giả thiết PI-OFUL, statement và proof intuition, practical cosine gap, complexity.
16–20: scope local vs paper, dữ liệu/split, chống leakage, protocol/baselines/seeds, tiêu chí chất lượng-cost-coverage.
21–25: budget-quality RouterBench và MMLU, learning-size/regret, ablations, so sánh paper có giới hạn.
26–29: cosine critique + stress; cost critique + counterfactual; strengths/limitations; future directions.
30–33: reproducibility/public links, takeaways trung thực, appendix công thức/chi tiết holdout và phụ lục câu hỏi bảo vệ.

Đây là gợi ý, có thể chỉnh trong giới hạn 20–40 slide. Đừng nén toàn bộ thí nghiệm/thảo luận thành vài dòng; notes dùng để giải thích giả thiết và giới hạn. Không đặt reference-only slide.

## 11. Mốc nộp và việc còn lại

Theo Moodle đã đọc ngày 30/10/2026, cần kiểm lại trước nộp:

| Mục | Hạn 23:59, giờ Việt Nam | Trang nộp |
| --- | --- | --- |
| Link GitHub public | 07/11/2026 | https://courses.ctda.hcmus.edu.vn/mod/assign/view.php?id=117389 |
| Midterm | 21/11/2026 | https://courses.ctda.hcmus.edu.vn/mod/assign/view.php?id=117390 |
| Poster | 05/12/2026 | https://courses.ctda.hcmus.edu.vn/mod/assign/view.php?id=117391 |

Thang điểm trong project: slide 45%, poster 50%, reproducibility/W&B/format 5%; đừng nhầm với trọng số project+oral trong toàn môn.

Còn làm: Claude hoàn thiện hai PPTX theo yêu cầu; nhóm kiểm Microsoft PowerPoint thật, số liệu/citation/link, luyện bảo vệ và tự nộp Moodle. Mỗi thành viên cần hiểu toàn bộ dự án. Chưa có kiểm chứng chạy lại toàn bộ GPU/grid trên máy sạch; clone không chứa dữ liệu thô/cache/model weights lớn, dùng hướng dẫn tái dựng có pin/checksum.

W&B hiện hiển thị Academic trial đến 29/11/2026. Đợt này không bật tính năng trả phí, không thêm phương thức thanh toán. Chưa xác nhận quyền lưu trữ/truy cập sau trial, nên nhóm cần giải quyết tài khoản miễn phí/Academic trước hạn poster 05/12 và kiểm lại link; bản kết quả public trên GitHub là bản độc lập. Claude không tự đăng ký gói, nạp tiền hoặc gọi API tính phí.

## 12. Checklist trước bàn giao PPTX

- Đúng tên/định dạng/số slide/kích thước A2 dọc; slide dưới 20 MB; có notes mọi slide.
- Đọc được ở cỡ trình chiếu/in thật, không chữ tràn, chart nhỏ hoặc legend không rõ.
- Số liệu khớp source; phân biệt mean/min/max với CI; so sánh cùng mẫu số.
- Ghi rõ 8-arm BGE-M3 local variant, MMLU nested, historical simulation; không hứa tái lập khớp paper.
- Không bỏ kết quả PILOT thua baseline; không gọi stress là bác bỏ theorem; không gọi cost sensitivity là test provider thật.
- Trích dẫn ở chân slide, liên kết công khai thật; không dựng ảnh/logo/quote như bằng chứng nghiên cứu.
- Giữ file chỉnh sửa được; xuất PDF preview để nhóm xem, sau đó kiểm PPTX bằng PowerPoint.

**Prompt ngắn để bắt đầu:** “Hãy dùng tài liệu này làm brief, đọc nguồn GitHub public và final ACL paper, thiết kế midterm/8.pptx (20–40 slides, notes mọi slide, <20 MB) và poster/8.pptx (1 slide A2 portrait). Dùng dữ liệu verified, kết luận trung thực, IEEE footnotes, biểu đồ chỉnh sửa được. Không chạy lại thí nghiệm hoặc gọi API trả phí. Trước xuất file, tự kiểm nội dung, citation, số liệu, layout và link theo checklist.”
