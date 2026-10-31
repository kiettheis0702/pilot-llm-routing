# Follow-up QA and handoff —31 October2026

Trạng thái: XONG phần code, nghiên cứu local và công bố W&B — đã chạy thật.
Sẵn sàng bàn giao nhánh/PR với bằng chứng ở dưới. Slide/poster giữ nguyên;
Claude nhận bản cập nhật riêng. Những giới hạn author fidelity vẫn được giữ.

Gate: `.venv/Scripts/python.exe -m pytest -q -rs --basetemp=outputs/qhy23`
->681 passed,1 skipped36.47s. Baseline origin/main a0551aa trong checkout riêng
->490 passed, cùng1 skip15.23s. Skip do Windows không cho tạo symlink
(WinError1314); không tự đổi quyền hệ thống. Lượt baseline đầu thiếu thư mục
outputs cha, gây168 lỗi fixture; đã giữ log, tạo thư mục và chạy lại ở temp mới.
Không dùng lỗi chuẩn bị đó để kết luận baseline có168 lỗi sản phẩm.

E2E: OpenAI main858+ablation240,240 policy simulations, preference500 ba seed,
MF ba seed với refit đảo nhãn test,12 điều kiện noise và231 binary conditions
đã chạy trên cache/dữ liệu thật. Instructor38018 vector/38055 mapped rows,
37 official-source neural samples,858 conditions/78 full replays. Hai Hybrid
classifiers riêng biệt: drift5000/398/10000 và binary32270/1000/3227; đều5 epoch,
đọc lại tất cả5 validation candidates và toàn bộ test logits,maxdiff0. PILOT
shift15000 selected-feedback rounds được replay đủ action/score/state. Các
biểu đồ khoa học mới đã mở xem; số liệu bất lợi được giữ nguyên.

Probe:22/22 targeted checks đạt:14 focused publication tests,6 runtime CLI
probes và2 real-backend recovery/idempotency checks. Trong14 test có5 actual-CLI simulations
cho artifact initialization/submission interruption và GraphQL-only SDK bootstrap.
6/6 runtime CLI probes MF/noise/binary ở checkpoint trước đạt. Bộ14 cuối gồm receipt persistence,
capacity guard, exact artifact inventory/content và phase-aware recovery.
Không gọi fake SDK là kiểm chứng network thật. Một actual verified record đã
được server nhận rồi driver cố ý bỏ response: receipt còn planned; resume dùng
đúng server_id cũ, finished. Lặp limit2 báo pending0, không thêm run. Toàn bộ
2498 records/35402 metrics và1128 historical records đã được đọc lại anonymous;
project total3626. Artifact70 entries khớp authenticated/anonymous manifest. Đổi nhãn test qua actual extractor không đổi Hybrid
train/validation arrays; đây không phải lượt refit neural thứ hai.

Đường hỏng: trước sửa, retry mạng làm mất server_id đã nhận (1 RED), artifact
planned receipt tìm artifact chưa gửi và source run còn running chặn recovery
(4 RED). Đã sửa, quan sát GREEN; source receipt giữ ID cũ, ghi phase trước
submission, không gửi lại khi acceptance chưa rõ, khôi phục đúng aggregate và
đọc lại finished bằng API mới. Lỗi/receipt được giữ để resume, không xoá dữ liệu.

Người thứ hai: n/a — pipeline local/serial cho một người; reviewer độc lập chỉ
đọc source/test/manifest/CSV, không chạy job hay thay dữ liệu. Review tìmP2
recovery ở trên; review lại bản sửa không còn material findings.

Lưu trữ:14 frozen report packages,10 protocols,235 pinned references đã khớp;
7/7 file presentation/PDF/reference giữ nguyên hash. Source snapshot chính xác
của task-shift report producer được lưu riêng, không repin report cũ. Payload
2498 summary views được dựng lại byte-identical với PYTHONHASHSEED1/2;70 file
artifact11,037,886 bytes, scan không có key/private-path matches.48 budget
controls đã reuse không bị import thành240 điều kiện mới lần nữa.

Chưa chạy: không dựng lại môi trường neural sạch lần thứ hai; checkout không chứa paid
vectors/raw checkpoints. Full author11-arm/release/split/order/settings và đúng
500 preference pairs vẫn thiếu bằng chứng; không diễn giải như đã tái lập.
Không sinh difficulty labels trả phí, không nộp Moodle hoặc sửa presentation.

Bug đã biết: không còn numeric mismatch hoặc material code finding chưa xử lý
trong các phần đã kiểm. Giới hạn khoa học/skip môi trường ở trên không bị che.
Binary AMP có6 overflow skips bình thường:1265 attempts/1259 applied; terminal
log cũ ghi optimizer_steps theo attempts, selection.json/report tách đủ hai số.

Quyết định: giữ8 model đã xác minh; không đoán3 alias hoặc xoá nhãn để khớp64.
Giữ main guarded-stop, đặt tên riêng các policy extensions. Giữ shared8-arm
prior cho verified2-arm study; H native thresholds khác budget controller.
PILOT50015/20–64D và MF100epochs–128D có criterion/training effort khác nhau.
Instructor instruction/slowT5/cap512 là lựa chọn local. Binary8/16/eval8 được
chọn bằng hardware-only check trước outcomes; drift2/64 giữ nguyên. Một seed
Hybrid/shift không tạo kết luận thống kê hoặc thuật toán thắng ở mọi điểm.

Tôi sẽ lo: bàn giao commit/nhánh/PR và lưu GitHub readback receipt sau khi đẩy;
registry, README và Claude map đã cập nhật từ kết quả công khai được xác minh.
Không có research job hoặc code finding đang bỏ dở.
Cần bạn: không có quyết định local mới cần hỏi. Không có phê duyệt API model
trả phí mới; không phát sinh request loại đó trong follow-up.

Evidence local (ignored operational files):

- outputs/openai-followup-20261031/full-tests-final-reviewed.log
- outputs/openai-followup-20261031/origin-main-baseline-tests-success.log
- outputs/openai-followup-20261031/final-evidence-audit.json
- outputs/openai-followup-20261031/receipt-retry-red.log
- outputs/openai-followup-20261031/artifact-recovery-red.log
- outputs/openai-followup-20261031/artifact-recovery-green.log
- outputs/openai-followup-20261031/publication/deterministic-preparation.json
- outputs/openai-followup-20261031/hybrid-default-batch/binary/independent-verification.json
- outputs/openai-followup-20261031/hybrid/drift/independent-verification.json
- outputs/openai-followup-20261031/drift/independent-verification.json

Real artifact integration initially failed because resume='must' requires SDK
initialization metadata absent from a GraphQL-only imported run. This occurred
BEFORE artifact submission; schema2 receipt correctly kept False. Added one
regression RED->GREEN, followed the official allow recipe after exact-ID proof,
carried canonical config/labels/summary, reviewed again and ran full681/1. Real
upload then succeeded:70 files11,037,886bytes, source aggregate still finished
with exact expected values. Anonymous readback after that verified all35402
new and13314 old expected metrics/configs; SDK run-file audit found no
wandb-metadata.json. No synthetic public test run or extra artifact was created.

Public evidence: [registry](../reports/openai-publication-2026-10-31/README.md),
[all2498 mappings](../reports/openai-publication-2026-10-31/experiment-map.csv),
anonymous run/artifact proofs and derived reconstruction proof in that folder.
A fresh Git-index export verifies14 packages and rebuilds byte-identical2498
payloads without ignored caches; this is derived-evidence reproduction, not a
claim to have repeated all neural fits in a second clean environment.
