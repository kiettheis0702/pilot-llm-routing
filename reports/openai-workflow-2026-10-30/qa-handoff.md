# Bàn giao QA luồng offline OpenAI

Trạng thái: **XONG phần import/cache/training/đánh giá offline**; chưa có thực nghiệm OpenAI thật.

Gate: toàn bộ pytest chạy tuần tự, một BLAS thread →564 passed,1 skipped trong23,36s;
trước lượt cuối9,44GiB RAM vật lý trống, không job nặng cạnh tranh. Baseline
preflight517; có23 case mới ở phạm vi này và24 test setupAPI đồng thời.

E2E: CLI fixture1536D xuất plan, import thiếu3/8, resume đủ8, đọc cache, fit prior,
học selected-feedback8 vòng và deployment8 query riêng; chạy hai lượt folder mới,
child process bị chặn mạng. Một fit BGE thật trên847/211/500 cũng được thực hiện.

Probe: sai thứ tự request/index; duplicate text; interrupted/truncated output;
error/missing/foreign request; conflict request ID; sai model/dimension/norm,
Boolean/NaN; source/chunk hỏng; overwrite; final-test mutation; synthetic source
mix và workload discovery Windows. Các kết quả/đếm trong verification.json.

Đường hỏng: importer giữ progress/chunk đã nhận, báo incomplete hoặc lỗi rõ;
không publish complete cache khi thiếu và không tự gửi generation. Source text
thay whitespace bị từ chối. Synthetic cost-size bị chặn mặc định và ghi nhãn
đúng khi cho phép fixture. Không dọn hoặc kill tiến trình người dùng.

Người thứ hai: n/a — CLI local; một reviewer độc lập đã hoàn tất read-only.

Lưu trữ: load lại cache/selected checkpoint bằng reader hiện có; kiểm lại hashes,
recovery không đếm usage hai lần, lambda checkpoint vẫn từ calibration.

Chưa chạy: actual Batch submit/status/download/import từ nhà cung cấp,
huấn luyện hoặc main core bằng vector OpenAI thật,11-model reproduction, W&B upload.
Không dùng request thật để smoke test vì chưa có phê duyệt chi phí trong lượt này.

Bug đã biết trong phần bàn giao: không còn material finding chưa xử lý được xác định.

Quyết định: giữ main split1245/313, tách study500; không suy diễn alias hoặc công bố
65,2% là tái lập paper; cache OpenAI vẫn cần response được cung cấp và input khớp.

Tôi sẽ lo: phạm vi offline trên đã hoàn tất; không có job chạy tiếp sau bàn giao.

Cần bạn: chưa cần key hoặc quyết định mở rộng scope; chỉ gửi embedding có phí sau
phê duyệt riêng đúng snapshot/job/ngân sách. Không commit/push trong lượt này.
