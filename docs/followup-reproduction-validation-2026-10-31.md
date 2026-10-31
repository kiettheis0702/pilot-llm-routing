# Kiểm chứng bổ sung khả năng chạy lại — 31/10/2026

**Đã kiểm chứng: 714 test đạt, toàn bộ chín giai đoạn CLI đạt.**
[Lượt kiểm chứng cuối](https://github.com/kiettheis0702/pilot-llm-routing/actions/runs/41201429682)
hoàn tất thành công, gồm đối chiếu cuối với bảng kết quả đã công bố.
Đây là bằng chứng chạy lại, không phải nghiên cứu mới.

## Phần thiếu đã bổ sung

Checkout đã khôi phục gói core vẫn thiếu split preference ba phần và bản ghi
quyết định batch mà adapter Hybrid yêu cầu. Một máy Windows hoàn toàn mới còn
thiếu bảng từ vựng tiktoken mà máy cũ đã cache toàn hệ thống. Các bước chuẩn bị
nay có lệnh tái tạo/khôi phục và kiểm SHA-256; không lấy một JSON giả để vượt
qua điều kiện của adapter.

Thêm lệnh fit Preference500 theo đúng quy tắc chọn trên calibration của seed42,
tái sử dụng lựa chọn cho seed7/2026, refit sau khi đảo riêng nhãn test, và phân
tích lại 18 trường hợp sốc chi phí. Tài liệu chỉ rõ đầu vào và thứ tự chạy cho
cả 11 nhóm nghiên cứu, gồm các nhánh neural đã công bố.

Hai lỗi review đã được sửa: kiểm lại checksum các đầu vào đã pin và báo cáo
tham chiếu **trước khi fit**; từ chối ghi đè summary khác nội dung. Năm test hồi
quy đã tái hiện các lỗi này trước bản sửa. Orchestration binary cũng được sửa:
giới hạn một nhóm tạo epoch-greedy trước, nên không thể đọc kết quả PILOT ngay;
lượt cuối chạy và kiểm tra đủ bảy phương pháp ở seed7.

## Máy và phiên bản kiểm chứng

- Code được chạy: `7fcfe5b4bbae5a62e384e6f31272e605e7a20210`.
- Một GitHub standard `windows-2025` runner mới, Python3.11.18, môi trường tạo
  từ `uv.lock`; không dùng cache Actions. Một pytest/BLAS worker, tải/cài tuần tự.
- Kiểm RAM vật lý trước cài đặt và từng giai đoạn; yêu cầu tối thiểu8GiB.
  Máy Windows của người dùng dưới ngưỡng này, nên không chạy local test/fit nặng
  và không dừng công việc khác để giải phóng RAM.
- Repository công khai; workflow từ chối chạy nếu repository private.
  [GitHub xác nhận standard runner cho public repo miễn phí](https://docs.github.com/en/billing/concepts/product-billing/github-actions).
  Không upload artifact hoặc lưu Actions dependency cache. Không gọi API model.
- Bản core cũ/tag/ZIP giữ nguyên. Tag bổ sung
  `openai-followup-reproduction-2026-10-31` cố định code cùng tài liệu sau tích hợp.
  Chỉ thay tài liệu sau commit đã chạy CI; đối chiếu hash source ở bảng dưới.

## Kết quả thực chạy

| Phần kiểm chứng | Kết quả thực tế |
|---|---|
| Regression tuần tự | `python -m pytest -q -rs --basetemp=.cache/ci-full`: **714 passed trong 38,82 giây**, không skip |
| Gói đầu vào |139 file đúng checksum; restore hai lần; chuẩn bị split/profile hai lần |
| Preference500 | Refit seed7/42/2026: **319/313/321 trên500**, khớp từng dòng công bố |
| Đảo nhãn test | Refit cả ba candidate của seed42; checkpoint được chọn byte-identical; **313→187/500** |
| Core PILOT |6 nhóm,133.917 vòng học,66 điều kiện deployment; replay độc lập và khớp CSV công bố |
| Ablation |5 biến thể RouterBench/seed7,40 điều kiện; replay độc lập và khớp CSV công bố |
| Ngân sách |40 điều kiện của cả5 policy ở RouterBench/seed7, replay độc lập; không nhận là đủ240 |
| Noise |2 mức0%/1% ở seed7, replay độc lập; không nhận là đủ12 |
| Binary | Tuning mới18 candidate;7 phương pháp seed7,225.890 vòng học,77 điều kiện; replay độc lập và khớp CSV công bố |
| Sốc chi phí |18 trường hợp khớp báo cáo; kiểm chéo bằng Decimal, giữ nguyên quyết định/reward |
| Báo cáo gốc | Cả16 package giữ nguyên checksum |

Reader ablation đọc129.080 vòng học, trong đó có nhóm dùng lại snapshot core;
không cộng con số đó thành tất cả các vòng huấn luyện mới. Ba seed preference
dùng cùng500 cặp test, không phải1.500 mẫu độc lập.

Baseline trên runner mới:701 test cũ đạt khi loại file test mới; source/test
khoa học cũ không thay đổi so với main `8ec760c`. Tám test entrypoint mới đã
[đỏ đúng lỗi thiếu module](https://github.com/kiettheis0702/pilot-llm-routing/actions/runs/41199016458)
trước implementation. Năm case review sau đó
[đỏ trước bản sửa](https://github.com/kiettheis0702/pilot-llm-routing/actions/runs/41199835687),
rồi cả13 case cùng toàn bộ regression đạt ở lượt cuối. Skip quyền symlink của
máy Windows cũ không xảy ra trên runner mới.

`REPRODUCTION_EVIDENCE_JSON` trong log CI chứa commit, source hashes, phạm vi
thực chạy và các kết quả đối chiếu. Các reader tính lại trạng thái, action,
reward và chi phí từ trace, độc lập với phép toán routing của producer.
So với CSV công bố dùng sai số tương đối1e-10/tuyệt đối1e-9 cho metric routing;
không gọi đó là cam kết mọi hệ điều hành/BLAS đều cho byte float giống nhau.

## Định danh source

| File | SHA-256 |
|---|---|
| `tools/openai_study_reproduce.py` | `bcd8ae929c17735c4af8b6175b83f1d814bb7cee5b52685b3a5997025df5a5e6` |
| `tools/reproduction-ci.ps1` | `a011b81c299427381e5f3ae25613c62d926ebae12efe373227ab384cb15956dd` |
| `configs/hybrid-batch-profile-2026-10-31.json` | `a62acd18cdc179d25f871c578f0b8ac8bf6f38713dbc9a0f288b62e42d7f1d51` |
| `uv.lock` | `f80da54c54aa1b4f7ac85d485c16678b686599e1f2584281ba8b21bffc44084b` |

## Phạm vi và giới hạn

- Core replay dùng prior64D và tuning đã khôi phục từ ZIP. Lượt release trước
  đã refit thật prior seed7 và kiểm byte-identical; lượt bổ sung không được gọi
  là refit toàn bộ prior/tuning core trên máy mới.
- Bản ghi batch Hybrid là **bằng chứng phần cứng lịch sử**, không phải benchmark
  của máy CI. Máy CUDA mới vẫn phải chạy probe thật trước khi train.
- Đã tải/kiểm34 source công khai và freeze đầu vào RouteLLM thật. Tại clone
  Windows riêng, helper cũng tải/kiểm4 asset DeBERTa;12 asset Instructor dùng lại
  các byte local đã kiểm hash. Đây không phải lần tải mới toàn bộ Instructor.
- Chưa cài/chạy lại môi trường CUDA mới, encode toàn bộ Instructor, refit MF
  hoặc hai classifier Hybrid trong lượt này. RAM local vẫn dưới8GiB; CPU CI
  không được dùng để tuyên bố đã kiểm chứng GPU. Recipe và prerequisite được
  ghi trong [hướng dẫn đầy đủ](followup-reproduce.md).
- Không chạy lại mọi điều kiện của tất cả các grid. Các nhóm/điều kiện đã chạy
  được liệt kê chính xác; không dùng smoke để tuyên bố full inventory hoàn tất.
- Không đổi số liệu/kết luận khoa học, không tăng số run W&B, không gọi API
  embedding/LLM trả phí. Các báo cáo gốc vẫn là bằng chứng nghiên cứu đã công bố.
- Hai file PPTX hiện tại khớp hash nhật ký cập nhật OpenAI ở chat khác (deck24
  slide, poster1 slide). Lượt này không sửa PPTX/PDF, không lặp lại QA bố cục,
  không xác nhận nộp Moodle hoặc mức hiểu bài của hai thành viên.

## Bàn giao kỹ thuật

```text
Trạng thái: XONG — các bổ sung đã chạy thật; phạm vi/giới hạn ghi ở trên.
Gate: 714 passed / 0 skipped; baseline701 test cũ đạt;13 test mới đạt.
E2E: Windows mới + môi trường khóa + ZIP công khai + source đã pin; code và
source hashes được ghi ở trên.
Probe:13/13 test mới và9/9 giai đoạn CLI với dữ liệu thật đạt.
Đường hỏng: thiếu/sai profile, đích xung đột, thiếu core inputs, chọn candidate
sai seed/tie rule, thay cost/cache sau chuẩn bị, thiếu report, summary xung đột.
Người thứ hai: n/a — CLI một người.
Lưu trữ: khôi phục/chuẩn bị hai lần; reader mở lại artifact độc lập; summary
giữ nguyên dữ liệu cũ khi bị gọi với nội dung khác.
Chưa chạy: CUDA/neural toàn bộ, full-grid thứ hai và QA/nộp bài/bảo vệ, như trên.
Bug đã biết: không còn finding chưa xử lý trong diff; hai finding review đã sửa.
Quyết định: bổ sung các dependency/entrypoint thực sự thiếu và kiểm chứng CPU
đại diện; giữ producer khoa học cùng protocol/checksum gốc.
Tôi sẽ lo: các phần code, kiểm chứng và hướng dẫn thuộc lượt này đã hoàn tất.
Cần bạn: không có quyết định hoặc phê duyệt chi phí nào cho các bước trên.
```
