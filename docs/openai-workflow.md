# Việc cần làm cho đồ án và luồng OpenAI đã chuẩn bị

Cập nhật31/10/2026: embedding OpenAI thật đã được người dùng duyệt và hoàn tất
bằng Standard API; cả hai cache đã được so khớp mọi vector. Dùng các đường dẫn
trong [báo cáo hoàn tất](../reports/openai-embedding-complete-2026-10-31.md), thay
cho các đường dẫn placeholder ở hướng dẫn chuẩn bị bên dưới. Đây là ghi chú
lịch sử ở mốc chỉ làm embedding. Sau đó người dùng mở rộng phạm vi: OpenAI
core858+ablation240 và các phần bổ sung đã chạy; xem [trạng thái hiện tại](../README.md)
và [bản cập nhật Claude](CLAUDE_UPDATE_OPENAI_2026-10-31.md). Không đọc các
trạng thái chuẩn bị bên dưới như trạng thái nghiên cứu mới nhất.

Ngày 30/10/2026. Tiếp tục theo yêu cầu chọn việc hữu ích và thực hiện; chưa có
phê duyệt chi phí. Không gửi API, không upload và không sửa slide/poster.

Công cụ gửi/theo dõi Batch được chuẩn bị riêng trong
[openai-embedding-setup.md](openai-embedding-setup.md). Phần này nối file output
tải về từ công cụ đó vào cache và training; không đọc key hoặc gọi submit/status.

## Phần nào thực sự cần

Với core scope đã chọn, ưu tiên hai thực nghiệm **budget–quality** và
**learning-size/regret**, baseline dùng cùng protocol, metric cùng mẫu số GPT-4,
phân tích sai khác và hai lập luận phản biện có bằng chứng. Nghiên cứu BGE tám
arm đã có nền đó; bước nâng chất lượng trực tiếp là chạy lại với embedding paper.

Đủ 11 arm giúp gần paper hơn nhưng chưa có bằng chứng phiên bản cho ba alias.
Không chờ vô hạn mới làm tiếp phần độc lập, và không gọi kết quả tám arm là khớp
paper. Core scope giảm phạm vi thực nghiệm, không tự chứng minh thay pool tương đương.
Quan hệ 86/64 nhãn và hyperparameter tác giả chưa công bố tiếp tục là giới hạn.

**500 cặp preference không phải điều kiện bắt buộc của hai thí nghiệm core.**
Không cắt train chỉ để làm số đếm giống paper. Đã kiểm riêng partition500 nhưng
giữ nó là thí nghiệm bổ sung. Protocol chính mới
[openai_core_protocol.json](../configs/openai_core_protocol.json) giữ train1.245 /
calibration313, seed/stream và cấu hình30/40–64D của study cũ để giảm thay đổi
đồng thời. Accuracy trên calibration không được gọi là final-test accuracy.
Payload OpenAI giữ toàn văn còn BGE cũ truncate13 input, nên so sánh trực tiếp
là so recipe biểu diễn; chưa cô lập hoàn toàn tác động của riêng encoder.

## Những phần đã làm tiếp

1. **Nhập response offline:** `tools.paper_preflight.import_embeddings` đọc file
   Batch đã tải, nối `custom_id` và index, kiểm hash/model1536D/vector/coverage,
   lưu request ID và chunk sau từng request thành công. File thiếu/lỗi không
   tạo complete cache. Resume đọc lại chunk đã lưu và không gửi request mới.
   Định dạng đối chiếu [Batch API](https://developers.openai.com/api/docs/guides/batch)
   và [response embedding](https://developers.openai.com/api/reference/resources/embeddings/methods/create).
2. **Nối pipeline hiện có:** `_aligned_cache` trước đây chỉ chấp nhận recipe BGE.
   Đã thêm profile OpenAI có hash chính xác của input, source, plan và request IDs.
   Thay whitespace giữ cùng normalized hash vẫn bị từ chối vì encoder nhận text khác.
   Kiểm tra BGE cũ được giữ nguyên; không tắt validation để nhận model mới.
3. **Huấn luyện ba phần:** `tools.paper_preflight.train` fit train, chọn cấu hình
   và tính lambda ở calibration, rồi mới chấm final test. Checkpoint giữ lambda
   calibration; final-test metrics lưu riêng. Test đổi nhãn/vector của final test
   xác nhận candidate, W,b,theta và calibration statistics không đổi.
4. **Chạy thật các CLI:** fixture1536D đi qua xuất request → import thiếu → resume
   → cache → prior → eight-round selected-feedback learning → tám query deployment
   riêng; child process bị chặn mạng, frozen router không đổi. Số liệu này synthetic,
   không phải vector hay hiệu năng OpenAI.
5. **Một fit BGE thật có giới hạn:** dùng cache1.024D sẵn, tám arm, 64D,30/40 epoch,
   seed42 trên847/211/500. Calibration141/211, test326/500=65,2%; mọi arm có lambda
   calibration hợp lệ. Không sweep, không chọn cấu hình theo500, không thay kết quả cũ.

65,2% không chứng minh tái lập con số65% của tác giả: pool, embedding, corpus và
split khác;500 cặp đã từng thuộc dữ liệu lịch sử của nhóm. Cũng không suy ra
cải thiện so với accuracy313 cặp cũ, vì hai phép đo có population khác nhau.

## Khi có output OpenAI đã tải về

Chạy tuần tự, output folder mới, một BLAS thread; CLI tự kiểm tài nguyên.
Các lệnh dưới đây **không gửi API** và yêu cầu response file đã tồn tại.
File tải về từ workflow setup là `output.jsonl`; có thể đưa thêm `errors.jsonl`
vào `--outputs` để receipt ghi rõ request thiếu/lỗi, không tự gửi lại.

```powershell
.venv/Scripts/python.exe -X utf8 -m tools.paper_preflight.import_embeddings --plan-dir outputs/paper-preflight-20261030/embedding-plan --outputs PATH_TO_DOWNLOADED_OUTPUT.jsonl --inputs outputs/course-20261030/rebuilt-inputs/inputs/routing-inputs.json outputs/course-20261030/rebuilt-inputs/inputs/preference-inputs.json --output-dir outputs/openai-import
```

Nếu output thiếu hoặc lần đọc bị gián đoạn, giữ thư mục cũ và thêm `--resume`
với đúng plan/input để nhập tiếp file output đã phục hồi. Không gửi lại generation.
`receipt.json` báo request nào đã nhận, request nào thiếu, usage từ response và
đường dẫn cache. Token usage là số do file response khai báo; không phải hóa đơn
đã được đối chiếu nhà cung cấp. Authenticity của response là caller-declared,
không được chứng thực độc lập chỉ bằng file JSON.

Khi cả298 request hoàn tất, cache routing là `caches/source-0`, preference là
`caches/source-1` theo đúng thứ tự hai file input phía trên. Luôn đọc receipt để
kiểm định danh, không dựa riêng vào tên folder.

Để fit **main core**, dùng prepared/split cũ đã ghim:

```powershell
.venv/Scripts/python.exe -X utf8 -m scripts.research_pipeline fit --prepared-dir outputs/local-pipeline-verification-20261029/prepared-identity --cache-dir outputs/openai-import/caches/source-1 --cost-sizes outputs/research-cost-inputs-20261029/cost-size-identity_match.json --output-dir outputs/openai-core-prior-s42 --dimension 64 --projection-epochs 30 --prior-epochs 40 --seed 42
```

Sau đó fit các seed đã chốt, tune alpha/window và bounds từ tuning, tạo index/
protocol OpenAI mới với hash thật rồi chạy core. Runner frozen BGE cũ tiếp tục
phục vụ tái dựng study cũ, không tự đổi đường dẫn hoặc repin thành OpenAI.

Để chạy **supplementary500**, dùng CLI riêng:

```powershell
.venv/Scripts/python.exe -X utf8 -m tools.paper_preflight.train --cache-dir outputs/openai-import/caches/source-1 --comparisons outputs/paper-preflight-20261030/input-audit/preference-three-way.json --cost-sizes outputs/research-cost-inputs-20261029/cost-size-identity_match.json --output-dir outputs/openai-preference-supplement
```

Các candidate short/current/long là lựa chọn local đã công bố, không phải setting
được khôi phục từ mã tác giả. Chạy supplemental không phải điều kiện mới để trả
phí hoặc để hoàn thành main core.

## Còn lại

- Cần duyệt rõ đúng lượt embedding trước khi gửi bất kỳ request có phí nào.
- Chưa có vector/kết quả OpenAI thật; import chỉ được kiểm bằng schema fixture.
- Chưa có matched eleven-arm result hoặc proof về alias/release tác giả.
- Binary, HybridLLM/RouteLLM đúng setting, drift/noise và Instructor-XL để sau core.

Gói kiểm chứng và code hashes:
[reports/openai-workflow-2026-10-30](../reports/openai-workflow-2026-10-30/README.md).
