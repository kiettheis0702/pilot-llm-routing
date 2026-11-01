# Nhật ký cập nhật slide và poster — 31/10/2026

Thực hiện theo [`CLAUDE_UPDATE_OPENAI_2026-10-31.md`](CLAUDE_UPDATE_OPENAI_2026-10-31.md). Phạm vi: chỉ sửa file trình bày và công cụ dựng chúng. Không chạy lại thí nghiệm, không gọi API, không đổi số hay kết luận của các gói kết quả, không commit, không nộp Moodle.

## 1. File và hash

Hash đầu vào khớp đúng baseline trong file hướng dẫn (deck `38176f62…f1e3e1f5d5`, poster `9c2806ab…bd0a0c737`). Bản cũ được giữ nguyên byte trong `outputs/presentation-backup-20261031-bge/` (thư mục local, không lên Git).

| File | SHA-256 (sau cập nhật) | Kích thước |
| --- | --- | ---: |
| `midterm/8.pptx` (24 slide) | `5002638b52f6b18c8840fa971597a4e7417708d2269c6035cb2d37a4b68276ab` | 247.784 B |
| `poster/8.pptx` (1 slide, 420 × 594 mm) | `051348291acac416d0e14a343726e2d4a7c5e3f1ba31201b19ff5cb14f01b8cf` | 42.323 B |
| `midterm/8-preview.pdf` | `bd664a9417222ecd81a43290de5297edcb86c67df25219c5c8b4954b0baab72a` | 1.256.104 B |
| `poster/8-preview.pdf` | `272b26e5199a1a50284f42a9702ebf8f328f6c92a811cd7975ededf484dce5e5` | 472.659 B |
| `midterm/references.txt` (13 mục IEEE) | `2859bd0774f6d34686bf78e50f5974623f6e64ed66d853c6382de46173381340` | 2.023 B |

Lưu ý: pptxgenjs ghi thời điểm tạo vào `docProps`, nên mỗi lần dựng lại hash sẽ khác dù nội dung giữ nguyên.

## 2. Mã nguồn đã sửa

- `tools/course/pptx/lib.mjs`
  - Thêm bộ đọc cho các gói OpenAI: core, ablation, budget, cost, binary, Hybrid, task-shift, RouteLLM, noise, Instructor, prior calibration.
  - Thêm tra cứu run trong registry mới `reports/openai-publication-2026-10-31/experiment-map.csv`.
  - Thêm 3 tài liệu IEEE: OpenAI embedding, RouteLLM, Instructor. Cả ba đã kiểm trên trang gốc (ACL Anthology, arXiv, kết quả tìm kiếm trang OpenAI).
  - Thêm `dropBlankScatterPoints()` để scatter có X riêng cho từng series không chứa `<c:v>` rỗng.
- `tools/course/pptx/build_deck.mjs`
  - Viết lại slide 13–24.
  - Sửa slide 1, 5–9, 12 theo bảng ánh xạ của hướng dẫn.
  - Có kiểm tra ngầm: build dừng nếu dữ liệu nguồn làm một câu khẳng định sai. Ví dụ: LinUCB không còn tốt nhất ở 50%, PILOT không còn thấp nhất trong 5 router, khoảng seed bắt đầu chồng nhau, guarded-stop khác main, hoặc tập "PILOT dẫn đầu" không còn là đoạn cuối của lưới.
- `tools/course/pptx/build_poster.mjs`: W2 và các ô tóm tắt dùng số OpenAI; W1 giữ nguyên (thí nghiệm tổng hợp).
- `tools/course/pptx/check_pptx.py`
  - Nhận link run từ cả hai registry.
  - Thêm quét câu đã lỗi thời trong chữ slide và notes: `no paid API`, `1,128 runs`, `held-out accuracy`, `helps PILOT most`…
  - Sửa regex `pending` thành `\bpending\b`, vì từ "spending" từng bị bắt nhầm.
- `docs/huong-dan-slide-poster-vi.md`: cập nhật toàn bộ phần giải thích tiếng Việt.

## 3. Thay đổi theo slide và nguồn số liệu

Mọi số do script đọc trực tiếp từ CSV/JSON với độ chính xác đầy đủ; biểu đồ dùng giá trị chưa làm tròn.

| Slide | Thay đổi | Nguồn |
| ---: | --- | --- |
| 1 | Notes: embedder của paper từ cache (437 request đã duyệt trước), không request trả phí mới; BGE chỉ còn là lịch sử | hướng dẫn mục "Embedding payment wording" |
| 5 | φ = text-embedding-3-small, D = 1.536 | `tracking-run-configs.jsonl` (`embedding_dimension`) |
| 6 | Dải "Our run" và notes: vector thật 1.536 chiều → 64 chiều | như trên |
| 7 | 313 cặp gọi là **calibration**; 186/190/194 (60,70%); BGE 189/188/193 chỉ trong notes; giao thức 847/211/500 ghi riêng | `reports/openai-core-2026-10-31/evidence/priors/seed-*/verification.json`; `reports/openai-routellm-2026-10-31/paired-preference500.csv` |
| 8 | λ_a theo calibration; không có ngân sách khi học | — |
| 9 | Guard ước lượng + quy tắc dừng tường minh; MMLU 20%: dừng sau 29/1.275, ≥ 94% ngân sách chưa tiêu | `reports/openai-core-2026-10-31/low-budget-diagnostic.json` |
| 12 | Bản của nhóm: dự đoán O(K d³), trạng thái O(K d²); chưa cài cập nhật hạng 1 | — |
| 13 | 1.098 = 858 + 240 (OpenAI, tách khỏi BGE lịch sử); bỏ "0 paid API calls"; chi tiết lượt embedding (38.018 văn bản, 437 request, 6.867.235 token, ~0,137 USD theo bảng giá, không phải hóa đơn) | `docs/openai-followup-status.md` |
| 14 | Định nghĩa quality/GPT-4, recorded cost/GPT-4, allocation (50% ≈ 48,98% / 47,79% chi phí GPT-4 ghi nhận); α 1 / 5 / cửa sổ 10; 108 nhóm replay | `mean-min-max.csv` (`allocated_budget_fraction_gpt4_mean`) |
| 15 | Vẽ lại E1 (5 router × 8 allocation × 2 tập). Ví dụ: RB 50% PILOT 86,5953% / 46,2348% so với LinUCB 91,2752% / 29,6213% | `reports/openai-core-2026-10-31/mean-min-max.csv` (main, learning 1.0) |
| 16 | Regret từ prefix lồng nhau, thêm đường "Always GPT-4" (3.675 / 1.416, thấp hơn PILOT); tiêu đề giới hạn "among adaptive learners" | `learning-prefixes.csv` (trung bình 3 seed) |
| 17 | Tiêu đề mới: học thêm gần như không đổi PILOT ở 50% (+0,49 / +1,52 pp); EpochGreedy +9,65 / +12,00 pp | `mean-min-max.csv` (budget 0.5, learning 0.1→1.0) |
| 18 | Ablation tính bằng pp: −6,70/−13,88; −0,90/+1,65; −0,35/−0,62; +1,55/+0,32; +2,14/−0,45; coverage 100% | `reports/openai-ablations-2026-10-31/paired-main.csv` (trung bình 3 cặp, budget 0.5) |
| 19 (mới) | Chẩn đoán bộ điều khiển và 4 chính sách MMLU; bảng 20%/30% đúng như hướng dẫn | `reports/openai-budget-2026-10-31/means.csv`; `low-budget-diagnostic.json` |
| 20 (mới) | Hybrid gốc vs PILOT (scatter, seed 42); binary 3 seed; task shift; Instructor; RouteLLM; noise | `reports/openai-hybrid-2026-10-31/*.csv`; `openai-binary-2026-10-31/means.csv`, `learning-prefixes.csv`; `openai-task-shift-2026-10-31/window-metrics.csv`; `instructor-core-2026-10-31/mean-min-max.csv`; `openai-routellm-2026-10-31/paired-preference500.csv`; `openai-noise-2026-10-31/means.csv` |
| 21 | Bảng 7 dòng; không nội suy điểm 25%; không tính sai số tái lập với 93%/25%; dòng preference ghi "ranking reversed" | như trên |
| 22 | Embedder bị loại trừ (OpenAI − BGE −0,33 / −2,76 pp, kèm điều kiện); thêm dòng "collapse at low allocation" | `reports/openai-core-2026-10-31/paired-bge.csv` |
| 23 | Kết luận có điều kiện; QR "3,626 public runs" | `reports/openai-publication-2026-10-31/public-verification.json` (`server_runs`) |
| 24 | Bảng run lấy từ registry mới; link riêng tới map BGE cũ và `learning-prefixes.csv` | registry mới |

Poster:
- MAIN FINDING và W2 đổi sang 0 → **23/48** (OpenAI, `cost-sensitivity.csv`, population `pilot-full`).
  - Đối chứng là 858 điều kiện core (`all-core`: 10% ở 1,1×).
  - Trung vị chi tiêu / allocation: 86% so với 16%, tính lại từ `openai-core per-seed.csv`.
  - BGE 0 → 25/48 ghi rõ là lịch sử.
  - Ở 1,0×: 0/858 core, 7/240 ablation (`all-openai` − `all-core`).
- W1 giữ nguyên, ghi rõ "synthetic", độc lập với embedder.
- Ô tóm tắt, S1–S3, theory audit (λ = 1: +1,55 / +0,32 pp; O(K d²) "có thể đạt", bản của nhóm là O(K d³)), QR "3,626 runs".

## 4. Đã kiểm

| Kiểm tra | Kết quả |
| --- | --- |
| `npm run build` (deck + poster) | Đạt; 24 slide, 13 tài liệu, `cite.check()` đúng thứ tự IEEE |
| `python tools/course/pptx/check_pptx.py` | **OK**: 24 slide, 0,25 MB, notes 275–497 từ; poster 420 × 594 mm, notes 455 từ; 42 + 5 link run W&B đều nằm trong registry đã xác minh; không còn câu lỗi thời |
| Validator OOXML của skill pptx | **All validations PASSED** cho cả hai file |
| Đối chiếu số | Script tạm (trong scratchpad, không lưu vào repo) tìm 71 con số chính của hướng dẫn trong chữ slide/notes của deck và 14 con số của poster: không thiếu số nào |
| Render bằng PowerPoint (COM) | Xuất PNG từng slide và PDF; đã **xem từng slide trong 24 slide và toàn poster**. Đã sửa: dải "Our run" ở slide 6–7 tràn dòng, ô task-shift ở slide 20 xuống 3 dòng, ô "Three strengths" trên poster tràn khung |
| Font trong PDF | Deck: Calibri, Cambria, Cambria Math, Consolas (ID run); poster: Calibri, Cambria, Cambria Math. Nhãn dữ liệu biểu đồ trước đó rơi về Arial, đã đổi sang font chữ thân của theme |
| Link GitHub | Cả 5 đích đều là file/thư mục đã track tại commit `8ec760c`, trùng `origin/main` |

## 5. Chưa kiểm / còn mở

- **Chưa mở file trên máy khác** hoặc máy sẽ dùng khi bảo vệ. PowerPoint trên máy này đủ font nên không thấy thay font; máy khác cần kiểm lại.
- **Chưa mở từng URL W&B trên trình duyệt** và chưa quét QR bằng điện thoại. Mới kiểm rằng mọi link run nằm trong registry đã được đọc lại ẩn danh (bằng chứng API của Codex, không phải kiểm giao diện).
- **Chưa chạy bộ pytest** (681 test của phần nghiên cứu). Lượt này không đổi mã nghiên cứu, và `check_pptx.py` không có test nào tham chiếu. RAM trống chỉ khoảng 5,4–6,3 GB, dưới ngưỡng 8 GB cho job nặng. Các lần render PowerPoint (một tiến trình, không phải build/test nhiều worker) vẫn chạy ở mức RAM này.
- **Chưa in thử A2.** Mới xem PNG 3.300 px và PDF.
- Deck tăng từ 22 lên **24 slide** (thêm slide 19–20 theo phần "supplemental slide" của hướng dẫn), vẫn trong giới hạn 20–40.
- Hướng dẫn ghi "Retain Vietnamese delivery". Slide vẫn bằng tiếng Anh theo yêu cầu trước đó của người dùng; phần giải thích tiếng Việt nằm trong `docs/huong-dan-slide-poster-vi.md`. Nếu nhóm cần slide tiếng Việt thì phải quyết định riêng.
- Các điều kiện tiên quyết chưa giải được vẫn giữ nguyên trên slide:
  - 3 ánh xạ phiên bản mô hình;
  - 86 nhãn raw so với 64 task;
  - split, thứ tự và cấu hình của tác giả;
  - 500 cặp của tác giả;
  - cặp Mistral/Mixtral của mục D.1.
- `deliverables/midterm/8.pptx` (33 slide) là bản lịch sử, không bị sửa.
- Không commit, không push, không nộp Moodle.
