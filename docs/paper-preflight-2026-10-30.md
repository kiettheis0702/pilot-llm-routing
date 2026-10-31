# Chuẩn bị miễn phí trước embedding OpenAI — 30/10/2026

Người dùng đã giao thực hiện mọi phần hữu ích trước khi tốn phí. Nhánh làm việc:
`codex/paper-preflight`. Không gửi API có phí, không upload W&B, không sửa slide,
không thay dữ liệu hay gói kết quả cũ. Protocol mới ở
[configs/paper_preflight_protocol.json](../configs/paper_preflight_protocol.json).

## 1. Danh tính model và nguồn tác giả

Đã tải và lưu hash của 17 nguồn/file metadata chính thức trong
`outputs/paper-preflight-20261030/sources/`; manifest công bố chỉ gồm URL/hash
ở gói báo cáo. Không tải trọng số LLM hay gọi provider.

| Model | Bằng chứng mới | Quyết định |
| --- | --- | --- |
| Mistral-7B | RouterBench `routers/common.py` gán cùng bảng giá cho `mistralai/mistral-7b-chat` và `mistralai/Mistral-7B-Instruct-v0.1`; bảng Arena ghim revision liên kết ID `mistral-7b-instruct` với v0.1 | Có dấu vết v0.1 nhưng bảng giá chưa chứng minh checkpoint đã sinh outcome. Chưa promote alias. |
| Mixtral-8x7B | Cùng phép gán bảng giá giữa alias chat và `Mixtral-8x7B-Instruct-v0.1`; Arena ghi rõ v0.1 | Cùng giới hạn; chưa promote alias. |
| Claude-v2 | Arena xác định Claude-2.0; RouterBench chỉ ghi Claude-v2 | Thiếu cầu nối phiên bản phía RouterBench; chưa promote alias. |

Nguồn trực tiếp:

- [RouterBench common.py, revision 30e0925](https://github.com/withmartian/routerbench/blob/30e0925ac07a45108b9ee82086040a5664ea83e6/routers/common.py).
- [Bảng model Arena, revision 6f2eeeb](https://huggingface.co/spaces/lmarena-ai/arena-leaderboard/blob/6f2eeeba09bf42144dbc118fe1a8e12e4c533f9d/leaderboard_table_20240426.csv).
- [README RouterBench](https://github.com/withmartian/routerbench/blob/30e0925ac07a45108b9ee82086040a5664ea83e6/README.md)
  xác nhận Martian là gateway. Đã xem cây mã của `martian-evals` ở main và hai
  nhánh API/dataset; adapter truyền tên model qua gateway, chưa tìm được bảng
  ánh xạ phiên bản provider tạo ra các outcome.
- [Website tác giả](https://ast-fri.github.io/contextual-bandit-routing-of-llms/)
  và [ACL](https://aclanthology.org/2025.findings-emnlp.1301/) chưa cung cấp link
  implementation PILOT trong các trang đã kiểm. Link source ở footer website
  nói về mã website. Đây không phải kết luận rằng mã tác giả không tồn tại.

**Bộ 11 model vẫn chưa được xác minh.** Không sửa `configs/model_mapping.json`
để biến dấu vết giá hoặc suy luận tên thành bằng chứng phiên bản.

## 2. Dữ liệu, split và thứ tự

Đã kiểm lại 36.497 query, 86 nhãn task thô, trong đó 57 nhãn MMLU. Có 16 dòng
prompt trùng vượt số nhóm duy nhất, nhưng **không có nhóm prompt chuẩn hóa nào
đi qua hai split routing**. Giữ nguyên tuning 1.000 / learning 32.270 /
deployment 3.227 và hash thứ tự riêng từng stream. MMLU vẫn là subset lồng,
không tự chia lại và không gọi là một dataset tái lập độc lập.

Quan hệ 86 nhãn với 64 task trong paper vẫn chưa có nguồn giải thích. Bộ 0-shot
được ghim revision/hash; bằng số query chưa chứng minh đúng release tác giả.
Không bỏ nhãn hoặc query để đạt một con số mong muốn.

Tập preference cũ có 1.245 train và 313 holdout. Holdout này dùng để tính hệ số
khởi tạo và đã được báo accuracy; cần phân biệt với một final test độc lập.

Đã tạo **đề xuất chia lại**, giữ nguyên toàn bộ nhóm prompt:

| Vai trò | Số cặp | Cách dùng dự kiến |
| --- | ---: | --- |
| Train | 847 | Học projection và prior |
| Calibration | 211 | Chọn cấu hình và tính lambda theo arm |
| Test | 500 | Chỉ đánh giá sau khi đóng băng mọi lựa chọn |

Cả tám arm có hỗ trợ trong từng phần. File `fit-comparisons.json` chỉ đưa train
và calibration vào định dạng fit cũ; `test-comparisons.json` tách riêng. Các file
cũ không bị thay thế và chưa fit model mới theo split này.

**Giới hạn quan trọng:** 500 cặp mới gồm 400 cặp từng thuộc train cũ và 100 cặp
thuộc holdout cũ. Đây là chia lại corpus để huấn luyện mới, không phải tập chưa
từng được xem trong toàn bộ lịch sử dự án, và không phải đúng 500 cặp của tác
giả. Khi đủ 11 model phải dựng lại protocol/split và đếm token phần bổ sung.

## 3. Thí nghiệm ngân sách đã chạy miễn phí

Thêm `affordability_mode='paper_literal'` vào `BinnedCostPolicy`. Chế độ mặc
định `guarded` giữ hành vi và báo cáo cũ. Chế độ mới chỉ bỏ điều kiện
`cost <= available` khỏi nhánh threshold; giữ nguyên fallback, rollover, cách
tính utilization và dừng stream. Đây là một cách đọc mã giả, không phải chứng
nhận implementation tác giả giống hoàn toàn.

Đã chạy **96 điều kiện = 48 cặp**, gồm RouterBench/MMLU × ba seed × tám mức
ngân sách × hai cơ chế, từ cùng frozen PILOT, BGE-M3, 8 model và cùng thứ tự
deployment. Không học lại router và không mua câu trả lời LLM.

Kết quả: **0/48 cặp thay đổi lựa chọn trên phần cùng phục vụ**, không đổi số
query phục vụ hoặc reward/cost; không có vượt ngân sách trong 96 điều kiện.
Cả 48 chế độ guarded khớp total reward, coverage và recorded cost của run cũ.
Do đó chưa có bằng chứng guard gây chênh lệch kết quả trong grid này.

Negative control tính tay chứng minh guard có thể tác động: B=1, hai bin,
cost arm tốt=.8 trong khi bin chỉ có .5. Literal vẫn chọn arm đó, cho số dư
−.3 và có thể vượt B ở lượt tiếp; guarded loại nó. Đây là fixture synthetic,
không phải kết quả RouterBench. Kết quả không tác động trên grid thật không
phủ định sự khác nhau của hai thuật toán ở đầu vào khác.

## 4. Cùng mẫu số GPT-4

Đã tính lại cả 1.098 điều kiện cũ:

`quality_fraction = total observed reward / total GPT-4 reward on all requested queries`

`cost_fraction = total recorded selected cost / total GPT-4 recorded cost on all requested queries`

Query chưa phục vụ vẫn nằm trong mẫu số. Ngân sách được cấp và chi phí đã ghi
nhận là hai cột khác nhau. Không dùng metric mới để chọn lại hyperparameter.

Tại 0,5 local reference budget, trung bình ba seed:

| Population | Chất lượng/GPT-4 | Recorded cost/GPT-4 | Coverage |
| --- | ---: | ---: | ---: |
| RouterBench | 87,18% | 46,29% | 100% |
| MMLU subset | 82,78% | 47,48% | 100% |

Đây vẫn là kết quả **BGE-M3, 8 arm, split local**. Chuẩn hóa mẫu số giúp đọc
được trục so sánh; chưa làm chúng trở thành phép tái lập khớp với headline paper.

## 5. Embedding đã chuẩn bị, chưa gửi

Model paper: `text-embedding-3-small` ([paper §3.2](https://aclanthology.org/2025.findings-emnlp.1301.pdf)).
Dùng mặc định OpenAI 1.536 chiều; chiều này là lựa chọn mặc định được công bố
của API, chưa được xác minh riêng như một hyperparameter của tác giả. Projection
64D là tầng khác và vẫn là giả định local.

Cache BGE cũ cắt 13 input preference ở giới hạn 2.048 token của tokenizer BGE.
Payload OpenAI mới giữ toàn bộ văn bản. Vì vậy so trực tiếp với kết quả BGE cũ
là so hai recipe biểu diễn; muốn quy riêng tác động cho model embedding phải
khớp văn bản thực sự đi vào hai encoder và học lại các tầng sau.

Đã xuất request JSONL chỉ ở máy:

- 38.055 dòng nguồn → **38.018 văn bản khác nhau chính xác theo UTF-8**.
- **6.867.235 token**, 298 HTTP embedding request dự kiến.
- Tối đa 128 văn bản / 50.000 token mỗi request; cap 8.191 token/văn bản để có
  biên bảo thủ so với giới hạn 8.192. Không có truncation.
- Model mặc định 1.536 chiều; không gửi tham số `dimensions`.
- Ước tính Standard **0,1373447 USD**, theo
  [giá OpenAI 0,02 USD/triệu token](https://developers.openai.com/api/docs/models/text-embedding-3-small).
  Chưa tính giảm giá Batch, thuế hoặc lần chạy lại.
- Có mapping query ID → request/index, hash nguồn và hash file đầu ra. Công cụ
  không có submit, không đọc key và không cài provider SDK.

Đây là input snapshot tám model được chấp nhận. Cache query có thể dùng lại
khi pool mở rộng, nhưng prompt preference mới cần được bổ sung và tính lại dự
toán. File request có prompt thật, nằm trong `outputs/` bị Git ignore; không
đưa chúng vào gói báo cáo công khai.

## 6. Huấn luyện và thứ tự bước tiếp

Nguồn đã kiểm xác nhận hai giai đoạn preference training và grid alpha/window;
chưa tìm được epoch, optimizer/sampling schedule, learning rate hay projection
dimension đầy đủ của tác giả. Protocol đề xuất ba lựa chọn local cùng 64D:
15/20, 30/40, 60/80 projection/prior epochs; seed chọn 42; chọn bằng calibration
accuracy, hòa thì chọn ít epoch hơn. Đánh giá cuối không chọn tham số. Khi một
arm không có accuracy dương để tính lambda thì báo blocker, không tạo prior giả.

Sau khi chốt gate khoa học: xác nhận input/pool cuối → duyệt đúng số request và
ngân sách → embedding một lần → học lại projection/prior và tuning → chạy hai
thí nghiệm core → cập nhật đối chứng embedding/pool → bổ sung binary/baseline
khi có dữ liệu đủ bằng chứng. BGE cache và mọi kết quả cũ tiếp tục là evidence
của nghiên cứu local, không đổi nhãn thành OpenAI.

## Chạy lại phần miễn phí

Chạy từng lệnh một, dùng thư mục đầu ra mới; đặt `OMP_NUM_THREADS`,
`MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS` bằng 1. CLI tự kiểm RAM >=8 GiB và
không có job nặng cạnh tranh.

```powershell
.venv/Scripts/python.exe -X utf8 -m tools.paper_preflight.audit --output-dir outputs/my-input-audit
.venv/Scripts/python.exe -X utf8 -m tools.paper_preflight.embedding_plan --inputs outputs/course-20261030/rebuilt-inputs/inputs/routing-inputs.json outputs/course-20261030/rebuilt-inputs/inputs/preference-inputs.json --output-dir outputs/my-embedding-plan
.venv/Scripts/python.exe -X utf8 -m tools.paper_preflight.compare --output-dir outputs/my-budget-comparison
```

Audit và embedding preparation đã chạy hai lần vào hai thư mục mới; nội dung
đầu ra giống byte. So sánh ngân sách có toàn bộ input số, trace chọn arm và hash
để kiểm lại độc lập. Báo cáo kiểm chứng cuối và những phần chưa chạy nằm trong
gói `reports/paper-preflight-2026-10-30/`.
