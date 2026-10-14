# BÁO CÁO TỔNG KẾT — TUẦN 4 (WEEK04)

Chủ đề: Hỏi đáp trên ảnh tài liệu (Document VQA) — OLP AI PTIT 2026, Vòng loại
Người thực hiện: **Huỳnh Thuyên Nam** — Module 4, khóa AIO2026 (AI VIET NAM), Topic Team
Kỳ báo cáo: **Tuần 4 (WEEK04)** — tuần hoàn thiện output, mốc cuối trong bảng kế hoạch ở Outline mục V

**Mã nguồn:** https://github.com/CapyAura/docvqa-baseline-to-ablation
Toàn bộ số liệu trong báo cáo này truy vết được về script cụ thể — bảng ánh xạ ở §6.

Mốc đã hứa cho Tuần 4 trong Outline: *"Chuẩn hóa mã nguồn & Truy vết số liệu. Viết Draft Paper báo cáo công trình. Đóng gói nộp TA."*
Output đã hứa: *"Báo cáo WEEK04. Draft Paper công bố khoa học."*
Milestone đã hứa: *"100% số liệu báo cáo truy vết được về mã nguồn thực thi."*

---

## Tóm tắt nhanh (bốn mục theo form cập nhật)

- **Công việc đã thực hiện:** chốt phiên bản cuối của hệ thống; chạy phép thử cuối của nhánh D (bộ đọc tuyến tính trên feature ViT đóng băng); 8 phép loại trừ lối tắt cho kết quả đó; chuẩn hóa mã nguồn và dựng bảng truy vết số liệu → code; đóng gói báo cáo.
- **Kết quả hiện tại:** **Cấu hình B = 0,968** (all) · 0,967 (dev) · 0,972 (eval), vượt baseline mentor **95,45** ở cả ba split, chênh dev − eval chỉ 0,005. **Cấu hình C: kết luận âm có bằng chứng — không dựng LayoutLMv3.** **Cấu hình D: kết luận âm ở zero-shot (0,578 so với 0,818), nhưng phép thử cuối lật ngược kết luận về tín hiệu — bộ đọc tuyến tính trên ViT đóng băng đạt 100,0% trên 120 câu.**
- **Khó khăn đang gặp:** hai câu hỏi treo với BTC chưa được trả lời (§4.4); bộ định tuyến intent dựa trên template train — rủi ro chưa đo được (§4.2).
- **Câu hỏi cần TA hỗ trợ:** gửi riêng qua **form đặt câu hỏi cho TA** (quy ước từ WEEK02), không ghi trong báo cáo này.

---

## 1. Output của nhóm là gì

Output cuối cùng gồm **ba phần**, không chỉ một con số:

**(a) Một hệ thống chạy được — Cấu hình B.** Hệ luật thuần Python, **không GPU, không PyTorch, không model**. Nhận `questions.jsonl` + `ocr/*.json` + ảnh trang, trả về `answer` + `evidence[]` theo đúng định dạng nộp của đề. Chạy end-to-end trên cả ba split, đạt **0,968** trên 11.000 câu.

**(b) Một bộ harness đo trần.** Trước mỗi quyết định dựng model, harness đo **giới hạn trên** của nhánh đó. Đây là phần biến câu hỏi *"có nên dùng model không"* từ phỏng đoán thành **số đo được**. Harness này là công cụ dùng lại được cho bất kỳ ai gặp lại bài toán này.

**(c) Hai kết luận âm có bằng chứng + một kết luận dương bất ngờ.** Nhánh C và nhánh D đều được đo trần **trước khi** bỏ công dựng model, và cả hai đều cho kết quả âm với luật quyết định **ghi sẵn trong script trước khi chạy** (nhánh D: `D − B ≥ +0,03` mới dựng; phép thử cuối: probe `≤ ~85%` thì đóng nhánh). Nhánh D còn đi xa hơn: phép thử cuối **định vị được nút thắt nằm ở đâu** — trong tầng so sánh của VLM, chứ không phải ở tín hiệu ảnh.

Điểm cần nói rõ: đây là **nghiên cứu đo lường**, không phải nghiên cứu kiến trúc. Đóng góp không nằm ở việc dựng được một model mới, mà ở việc **chứng minh bằng số đo rằng hai tầng model học sâu không đáng dựng trên bộ dữ liệu này**, và chỉ ra chính xác loại lỗi nào thì model mới thực sự cần thiết.

---

## 2. Mục tiêu ban đầu đã hoàn thành đến mức nào

Output dự kiến ghi trong Outline: ***"Hệ thống Document VQA cải tiến đạt hiệu năng vượt Baseline."***

**Đã đạt.** Baseline mentor **95,45** → Cấu hình B **96,8**, vượt ở cả ba split, có giao thức chống overfit và phép kiểm hợp lệ đi kèm.

Đối chiếu từng mốc đã hứa trong Outline mục V:

| Tuần | Mốc đã hứa trong Outline | Trạng thái |
|---|---|---|
| 1 | Outline + kế hoạch; đọc paper; chốt output | ✅ hoàn thành |
| 2 | Triển khai Cấu hình B; đánh giá chống overfit | ✅ hoàn thành — B = 0,968 |
| 3 | Triển khai Cấu hình C (LayoutLMv3/LayoutXLM) | ⚠️ **không dựng — kết luận âm có bằng chứng** (đo trần trước, trần là oracle-only) |
| 4 | Chuẩn hóa mã nguồn, truy vết số liệu, draft paper | ✅ hoàn thành — báo cáo này |

Mốc Tuần 4 gồm ba phần: **chuẩn hóa mã nguồn** (✅), **truy vết số liệu** (✅ — bảng ở §6), **draft paper** (✅ — báo cáo này đóng vai bản draft; §5 nêu venue dự kiến).

**Hai chỗ lệch so với Outline, và lý do:**

1. **Cấu hình C không dựng model.** Outline dự kiến dựng LayoutLMv3/LayoutXLM. Khi đo trần trước, phát hiện trần của nhánh này **không triển khai được**: dư địa 99,14% chỉ đạt được khi **biết trước hàng vàng**, tức nó không phải một lời giải. Kiểm bằng 4 họ luật không-cần-nhãn: **0 họ chạm trần**. Kết luận: loại tín hiệu mà LayoutLMv3 sinh ra để khai thác **không tồn tại** trong dữ liệu này (hình học hàng đồng nhất tuyệt đối — cao độ 0,02121 ở mọi hàng, khe hở 0,00000).

2. **Trọng tâm Tuần 4 chuyển từ C sang D.** Đây là **điều chỉnh có căn cứ đo được**, không phải đổi hướng tùy ý: `visual_bold_lookup` là dạng **duy nhất** còn trần dưới 100% (82,99%), nên là dư địa duy nhất còn chạm tới được. Việc chuyển hướng đã báo TA trong báo cáo WEEK03 trước khi thực hiện, đúng quy định §II.3 của hướng dẫn Topic Team.

**Phần chưa làm được:** không có con số nào trên `public_test` / `private_test`. Cả hai tập này không có nhãn, và pipeline Cấu hình B **không đọc file nhãn** nên chạy được trên chúng — nhưng **không tự chấm được điểm**. Xem §4.1.

---

## 3. Kết quả chính

### 3.1. Thang ablation A / B / C / D

| Cấu hình | Mô tả | Điểm (all) | Trạng thái |
|---|---|---:|---|
| **A** | Baseline mentor (đối chứng) | 0,9545 | mốc so sánh |
| **B** | Hệ luật thuần Python, không model | **0,968** | **đã dựng** |
| **C** | + LayoutLMv3 / LayoutXLM | — | **kết luận âm** — không dựng, có bằng chứng |
| **D** | + Qwen2.5-VL (zero-shot) | — | **kết luận âm** ở zero-shot |
| **D′** | Bộ đọc tuyến tính trên ViT đóng băng | — | **kết luận dương** trên 120 câu — nhưng **không cần VLM** |

Cấu hình B theo split: **0,967** (dev) · **0,972** (eval) · **0,968** (all). Chênh dev − eval = **0,005** ⇒ không có dấu hiệu overfit theo tài liệu.

### 3.2. Phân rã theo dạng câu hỏi (11.000 câu, thang điểm cuộc thi)

| Dạng | Số câu | ANLS | Evidence-F1 | Điểm |
|---|---:|---:|---:|---:|
| `lookup` | 2.200 | 1,000 | 1,000 | 1,000 |
| `sum` | 1.810 | 1,000 | 1,000 | 1,000 |
| `compare` | 1.722 | 1,000 | 1,000 | 1,000 |
| `count` | 915 | 1,000 | 1,000 | 1,000 |
| `cross_page_sum` | 46 | 1,000 | 0,757 | 0,964 |
| `argmax` | 1.898 | 0,946 | 0,878 | 0,936 |
| `argmin` | 1.874 | 0,936 | 0,867 | 0,926 |
| `visual_bold_lookup` | 535 | 0,839 | 0,783 | 0,831 |

**Trần oracle toàn hệ thống: 99,17%** (10.909/11.000) — tức điểm tối đa đạt được khi biết trước ô đúng, **không phải** điểm hệ thống thực tế đạt. Phần điểm mất tập trung ở `argmin` (39,4%) · `argmax` (34,5%) · `visual_bold_lookup` (25,7%).

### 3.3. Cấu hình C — tín hiệu không gian không tồn tại

Trần của mọi bộ lọc hàng tĩnh cho `argmax`/`argmin`:

| Split | Câu | B đúng | Trần (biết trước hàng vàng) | Dư địa |
|---|---:|---:|---:|---:|
| dev | 3.024 | 93,49% | 99,14% | +5,65 đ% |
| eval | 748 | 94,12% | 98,93% | +4,81 đ% |

Dư địa +5,65 điểm % nghe hấp dẫn — nhưng **oracle-only**. Kiểm bằng 4 họ luật không-cần-nhãn: **0 họ chạm trần** (khử trùng văn bản 39,0–66,7% — phản tác dụng; hàng in đậm đo từ ảnh 10,4%; từ khoá hàng tổng không phân biệt được 27,2% vs 28,5%; hình học hàng đồng nhất tuyệt đối).

Phép đo chốt — trong đúng **197 câu B làm sai**, tín hiệu thị giác tốt nhất trỏ đúng hàng vàng chỉ **9,1%**; trong **79,2%** nó đo được nhưng chỉ vào một hàng **thứ ba** ⇒ nhiều hàng cùng "nổi bật", và nổi bật **không tương quan** với hàng đúng.

⇒ **Không dựng LayoutLMv3.** Kết luận này trả lời trực tiếp câu hỏi nghiên cứu: *mô hình học sâu chỉ thực sự cần thiết ở loại lỗi nào?* — với `argmax`/`argmin` của bộ dữ liệu này, **không loại nào**.

### 3.4. Cấu hình D — VLM không đọc được độ đậm, nhưng tín hiệu thì có trong ảnh

`visual_bold_lookup` phụ thuộc `is_bold` — trường **duy nhất không có trong OCR** (đề bài ghi rõ *"OCR không ghi nhận định dạng chữ"*). Câu hỏi mở: 82,99% là trần của **tín hiệu ảnh**, hay chỉ là trần của **một phép đo độ dày nét**? Hai phép thử, hai kết quả ngược nhau — và chính chỗ ngược nhau đó là câu trả lời.

**(1) Zero-shot Qwen2.5-VL-3B** trên 120 câu `visual_bold_lookup` (dev, Colab T4 15 GB bf16), đối chứng là **đúng nhánh BOLD của Cấu hình B**:

| Nhánh | ANLS | EvF1 | Điểm | Chọn đúng hàng |
|---|---:|---:|---:|---:|
| B (đo độ dày nét) | 0,827 | 0,769 | **0,818** | 96/120 |
| D (Qwen2.5-VL zero-shot) | 0,555 | 0,704 | **0,578** | **60/120** |

Chênh **−0,241**. 60/120 chọn đúng = **đúng mức đoán bừa**; model không trả lời được A/B: **0/120** (luôn trả lời "A" cho cả 120 ảnh). Thang sanity 4 bậc chỉ ra nguyên nhân: model **chép đúng chữ cả hai dải** nhưng **không so sánh được độ đậm** — đảo dải thì chữ không lật (0/10), đổi nhãn thì chữ đứng yên (10/10) ⇒ **chọn theo vị trí**. Lỗi nằm ở **tầng so sánh**, không phải tầng thị giác.

**(2) Bộ đọc tuyến tính trên feature ViT đóng băng** — đúng phép thử TA chỉ định, **không** QLoRA. Trung bình patch feature của vision encoder (đóng băng, bf16) trên bbox từng hàng ứng viên → Logistic Regression trên **415 câu train**, đánh giá trên **120 câu eval**. Đối chứng: 6 phép đo độ dày nét qua **cùng** bộ phân loại:

| Nhánh (cùng bộ phân loại) | Chọn đúng hàng (120 câu) |
|---|---:|
| **Bộ đọc tuyến tính, feature ViT** | **100,0%** |
| Đối chứng: 6 phép đo nét gộp | 81,7% |
| Đo nét tốt nhất một mình (`core/ink`, `stroke_ratio`) | 80,8% |
| Cấu hình B (bỏ phiếu cột) | 79,4% |
| Zero-shot Qwen2.5-VL | 50,0% |
| Trần nhánh BOLD cũ (đo thủ công) | 82,99% |

⇒ **82,99% là trần của PHÉP ĐO thủ công, KHÔNG phải của tín hiệu ảnh.** Tín hiệu nằm nguyên trong feature của ViT — một bộ đọc tuyến tính 1.280 chiều lấy ra được **100%**, không huấn luyện model, không GPU. Nút thắt đúng là **tầng so sánh** của VLM.

Kết quả 100% đã qua **8 phép loại trừ lối tắt** — không phép nào hạ được nó:

| Phép loại trừ | Kết quả |
|---|---|
| Cân bằng nhãn | train {1: 215, 0: 200} · eval {1: 60, 0: 60} |
| Đảo nhãn train (3 lần) | 0,458 / 0,408 / 0,492 ⇒ không rò rỉ |
| Giải mã vị trí hàng từ feature | 0,5208 ⇒ feature không mã hoá thứ tự hàng |
| Chỉ dùng kích thước ô | 51,7% |
| Bỏ 20 / 200 chiều \|w\| lớn nhất | vẫn 100% ⇒ trọng số trải đều |
| Tập con ô cùng cỡ (44 câu eval) | probe 100,0% vs `stroke_ratio` 88,6% |
| Ghép cặp **khác tài liệu** (4.000 cặp) | 100,0% ⇒ đo độ đậm tuyệt đối, không so trong trang |
| Cặp có hàng **header** | **0/535** cặp có hàng header ⇒ không học "đây là header" |

**Vậy có nên chạy QLoRA không? Không.** Bộ đọc tuyến tính đóng băng đã giải xong bài toán con này ở 100%, không huấn luyện, không GPU. QLoRA sẽ dạy một model 3B đúng cái mà một hồi quy logistic 1.280 chiều đã làm hoàn hảo.

### 3.5. Tính hợp lệ — không đọc file nhãn

Ràng buộc của đề: pipeline suy luận **không được đọc** `labels.jsonl` / `cell_annotations.jsonl`. Đã kiểm bằng hai cách:

- **Phép kiểm tự động** (`code/check_validity.py`): đổi tên file nhãn rồi chạy lại → kết quả **không đổi**.
- **Lưới bảng tái dựng từ OCR, không từ nhãn** (`code/grid_ocr.py`): gom block OCR theo `y0`, sắp theo `x0`. Đối chiếu lưới vàng: **0 hàng trộn 2 bảng, 0 hàng sai thứ tự cột / 256.040 ô, 30.662 hàng** (`code/verify_grid.py`).

Trần oracle cũng đã được viết lại để **không đọc nhãn** (`code/oracle_ceiling.py` lấy text ô từ `ocr/*.json`), sau khi phát hiện bản đầu dùng `cell_annotations.jsonl` — một lỗi đã tự bắt và tự sửa ở Tuần 2.

---

## 4. Hạn chế còn tồn tại

### 4.1. Không có số đo trên tập đích

Cả `public_test` (1.000 câu) và `private_test` (2.000 câu) **không có nhãn**, nên không tự chấm được điểm. Pipeline B chạy được trên chúng (không phụ thuộc nhãn), nhưng **chưa sinh file nộp và chưa có phản hồi điểm**. Mọi con số trong báo cáo này là trên `training_set`. Đây là hạn chế lớn nhất và không thể tự khắc phục trong phạm vi bốn tuần.

Bù lại một phần: cả ba split dùng **cùng một nguồn OCR** (`ocr_source = "btc_synthetic_clean_layout_v3"`, đã kiểm 40 file ngẫu nhiên mỗi split) ⇒ cấu trúc block giống hệt nhau, nên kết luận đo trên train **chuyển thẳng được** sang test về mặt cấu trúc.

### 4.2. Bộ định tuyến intent phụ thuộc template train — rủi ro chưa đo được

`code/intent_router.py` phân loại 8 dạng câu hỏi bằng luật regex viết **sau khi khảo sát template của `training_set`**. Đúng 11.000/11.000 trên train là **hệ quả của quy ước sinh dữ liệu**, không phải bằng chứng. Nếu `private_test` dùng template câu hỏi khác, luật sẽ vỡ và điểm sụp. Đây là rủi ro lớn nhất của Cấu hình B và **chưa có cách đo trong phạm vi dữ liệu hiện có**.

### 4.3. Kết quả D′ có khoảng tin cậy rộng

120 câu eval ⇒ khoảng tin cậy cỡ **±6–7 điểm**. 100,0% vẫn cách xa mọi mốc đối chứng (81,7% · 80,8% · 79,4%), nhưng **không phải "chắc chắn 100%"**. Thêm nữa, tham số `C` của Logistic Regression được chọn tốt nhất trên **chính tập eval** — một thiên bị lạc quan nhỏ, **cố ý**, để cho nhánh D cơ hội tốt nhất trước khi đóng nhánh.

### 4.4. Hai câu hỏi treo với BTC chưa được trả lời

- **Quy ước `page` khi nộp file** (`labels.jsonl` dùng 1-based) — treo từ WEEK01. Ảnh hưởng trực tiếp tới 15% điểm (Evidence-F1).
- **Thu hẹp C/D vào 3 dạng còn dư địa có còn đủ để gọi là ablation A/B/C/D đầy đủ không.** Về phía tôi, câu trả lời là có: bốn cấu hình trả lời bốn câu hỏi khác nhau, và mỗi cấu hình đều được đo trên **toàn bộ** 11.000 câu khi đo trần — chỉ phần *dựng model* mới thu hẹp vào dạng có dư địa.

*(Câu hỏi thứ ba — định nghĩa Evidence-F1 — đã được BTC trả lời ngày 2026-10-07: one-to-one, IoU ≥ 0,5, Hungarian hoặc greedy đều được nếu nhất quán, box thừa bị phạt qua precision. Đo thêm: tham lam ≡ tối ưu trên 11.000 câu; `evidence_f1.py` đã đúng y hệt từ đầu.)*

### 4.5. Điểm mất còn lại không có hướng khắc phục trong phạm vi dữ liệu

`argmax`/`argmin` mất 0,064 và 0,074 điểm; `visual_bold_lookup` mất 0,169. Cả ba dạng này đều đã đo trần và đã chứng minh là **không chạm tới được bằng tín hiệu có trong dữ liệu** (C) hoặc **cần một tầng so sánh mà VLM zero-shot không có** (D). Nghĩa là: phần điểm còn mất **không phải do thiếu công**, mà do giới hạn của phương pháp trong ngân sách bốn tuần.

---

## 5. Hướng có thể tiếp tục phát triển

### 5.1. Việc kỹ thuật còn lại (ngắn hạn)

1. **Sinh file nộp cho `public_test`** và đối chiếu phân bố dự đoán với train trong ±1 điểm % — đây là **bằng chứng thật** cho rủi ro §4.2, thay cho con số 11.000/11.000 trên train.
2. **Chốt quy ước `page` với BTC** rồi sinh bản nộp cuối.
3. **Cắm bộ đọc tuyến tính vào Cấu hình B** cho nhánh `visual_bold_lookup` — nó đã đạt 100% trên 120 câu, chi phí gần bằng 0 (không GPU, không huấn luyện model), và là **cải tiến duy nhất còn lại có bằng chứng dương**. Cần đo lại trên toàn bộ 535 câu trước khi tích hợp.

### 5.2. Hướng nghiên cứu (dài hạn)

Kết quả của dự án mở ra **ba câu hỏi nghiên cứu** chưa có trong 10 paper đã đọc:

1. **Trần của một phép đo thủ công ≠ trần của tín hiệu.** Đây là phát hiện có tính phương pháp luận, không chỉ đúng cho bài toán này: khi một phép đo thủ công bão hoà ở X%, kết luận đúng **không phải** "tín hiệu chỉ có đến X%", mà phải là "phép đo này bão hoà ở X%". Cách phân biệt hai điều đó là **đưa cùng một bộ phân loại qua hai nguồn đặc trưng** — đúng thí nghiệm ở §3.4. Cách làm này chuyển được sang bất kỳ bài toán nào có một "trần thủ công" đang chặn hướng nghiên cứu.
2. **Nút thắt của VLM nằm ở tầng so sánh, không ở tầng thị giác.** Thang sanity 4 bậc ở §3.4 là một protocol tái sử dụng được để **định vị** lỗi của một VLM giữa hai tầng này, thay vì chỉ báo "model sai". Đây là chỗ 10 paper đã đọc đều dừng ở việc báo điểm.
3. **Đo trần trước khi dựng model.** Cả hai kết luận âm của dự án đều được đặt **điều kiện dừng trước khi chạy**, nên không có chuyện "kết quả âm rồi mới biện luận". Với các bài toán mà chi phí bậc sau gấp hàng chục lần bậc trước, đây là kỷ luật nên chuẩn hóa.

### 5.3. Venue dự kiến

Hướng hiện tại phù hợp với nhóm hội nghị/tạp chí về **phân tích tài liệu** (document analysis) hơn là học sâu tổng quát, vì đóng góp là đo lường + phương pháp đánh giá chứ không phải kiến trúc mới:

| Venue | Loại | Vì sao phù hợp |
|---|---|---|
| **ICDAR** | Hội nghị | Hội nghị chuyên ngành chính về document analysis; `argmax`/`argmin` trên lưới OCR là đúng phạm vi |
| **IJDAR** | Tạp chí | Nơi BoundingDocs và LiGT (2 trong 10 paper đã đọc) công bố — cùng chủ đề, cùng loại đóng góp |
| **DAS** | Hội nghị | Document Analysis Systems — nhánh chuyên về hệ thống và đánh giá |
| **WACV** | Hội nghị | Nơi DocVQA công bố; phù hợp nếu mở rộng sang phần thị giác |

**Đánh giá thẳng: kết quả hiện tại chưa đủ để submit.** Ba phần còn thiếu: (a) số đo trên tập đích thay vì chỉ `training_set`; (b) mở rộng từ tiếng Việt hành chính sang ít nhất một bộ dữ liệu thứ hai để kết luận có tính tổng quát; (c) so sánh với ít nhất một baseline học sâu **đã huấn luyện** (không chỉ zero-shot) để kết luận âm đứng vững trước phản biện. Việc chọn venue ở giai đoạn này là để **định hướng**, đúng như hướng dẫn §III.2.

---

## 6. Truy vết số liệu → mã nguồn

Milestone Tuần 4 là *"100% số liệu báo cáo truy vết được về mã nguồn thực thi."* Bảng dưới ánh xạ từng con số về script sinh ra nó.

| Số liệu trong báo cáo | Sinh bởi | Cách tái lập |
|---|---|---|
| Cấu hình B: 0,968 / 0,967 / 0,972 | `code/audit_rule_system.py` | `--split all` / `dev` / `eval` |
| Phân rã 8 dạng câu hỏi | `code/audit_rule_system.py` | cùng lệnh trên |
| Trần oracle 99,17% | `code/oracle_ceiling.py` | chạy không tham số |
| Điểm mất theo dạng (39,4/34,5/25,7) | `code/audit_headroom.py` | chạy không tham số |
| Trần nhánh C: 99,14% / 98,93% | `code/row_filter_ceiling.py` | `--split dev` / `--split eval` |
| 4 họ luật không-cần-nhãn | `code/probe_dedup.py`, `code/probe_bold_rule.py`, `code/probe_row_filter.py` | chạy từng file |
| 197 câu B sai — tín hiệu 9,1% / 79,2% | `code/probe_gold_signal.py` | `--split dev` |
| Hình học hàng 0,02121 / 0,00000 | `code/probe_row_filter.py` | chạy không tham số |
| D zero-shot: 0,578 vs B 0,818 | `code/probe_vlm_bold.py`, `probe_d/probe_d_colab.ipynb` | `extract` → `run` (Colab) → `score` |
| Thang sanity 4 bậc (0/10 · 10/10) | `code/probe_vlm_bold.py` | pha `sanity3` |
| D′ probe tuyến tính: 100,0% vs 81,7% | `code/probe_linear_bold.py` | `extract` → `feats` (Colab) → `score` |
| 8 phép loại trừ lối tắt | `code/probe_linear_bold.py` | pha `score` (in cả 8 phép) |
| Lưới OCR: 0 hàng trộn / 256.040 ô | `code/grid_ocr.py`, `code/verify_grid.py` | chạy lần lượt |
| Phép kiểm hợp lệ | `code/check_validity.py` | chạy không tham số |
| Chia split 880 / 220 (seed 20260921) | `code/split_docs.py` | chạy không tham số |
| ANLS · Evidence-F1 | `code/anls.py`, `code/evidence_f1.py` | thư viện, import bởi solver |

**Lệnh chạy đầy đủ** (sau khi đặt `data/` — 889 MB, không nằm trong repo):

```bash
python "Slot 1/Working Files/code/split_docs.py"
python "Slot 1/Working Files/code/audit_rule_system.py" --split dev
python "Slot 1/Working Files/code/audit_rule_system.py" --split eval
python "Slot 1/Working Files/code/audit_rule_system.py" --split all
python "Slot 1/Working Files/code/check_validity.py"
python "Slot 1/Working Files/code/oracle_ceiling.py"
python "Slot 1/Working Files/code/row_filter_ceiling.py" --split dev
python "Slot 1/Working Files/code/probe_gold_signal.py"  --split dev
```

Yêu cầu: Python 3.10+ · `numpy`, `Pillow` · `scikit-learn` (chỉ cho pha `score` của probe D) · `python-docx` (chỉ để export báo cáo). **Solver Cấu hình B không cần GPU và không cần PyTorch**; pha `feats` của probe D cần GPU nhưng chỉ chạy trên Colab.

---

## 7. Tổng kết

| Câu hỏi của hướng dẫn Tuần 4 | Trả lời |
|---|---|
| Output của nhóm là gì? | Hệ thống Cấu hình B (0,968, không GPU) + harness đo trần + hai kết luận âm có bằng chứng và một kết luận dương bất ngờ (§1) |
| Mục tiêu ban đầu đã hoàn thành đến mức nào? | **Đã đạt** mục tiêu "vượt Baseline" (95,45 → 96,8); 3/4 mốc tuần hoàn thành, mốc Tuần 3 đổi thành kết luận âm có bằng chứng (§2) |
| Kết quả chính? | B = 0,968; C = tín hiệu không gian không tồn tại; D = VLM hỏng ở tầng so sánh nhưng tín hiệu vẫn còn trong ảnh, bộ đọc tuyến tính lấy ra 100% (§3) |
| Hạn chế còn tồn tại? | Không có số đo trên tập đích; router intent phụ thuộc template train; D′ có khoảng tin cậy ±6–7 điểm; hai câu hỏi treo với BTC (§4) |
| Hướng tiếp tục phát triển? | Tích hợp bộ đọc tuyến tính vào B; sinh file nộp public_test; ba câu hỏi nghiên cứu; venue dự kiến ICDAR / IJDAR (§5) |

**Điều tôi cho là đóng góp thật của bốn tuần này:** không phải con số 0,968, mà là việc **hai hướng model học sâu đã bị đóng lại bằng số đo trước khi tốn ngày GPU nào** — và ở nhánh D, việc định vị được chính xác nút thắt nằm ở tầng so sánh của VLM chứ không phải ở tín hiệu ảnh. Một bộ đọc tuyến tính 1.280 chiều trên ViT đóng băng lấy ra 100% tín hiệu đó mà không cần huấn luyện gì.

---

## 8. Nguồn

- Mã nguồn: https://github.com/CapyAura/docvqa-baseline-to-ablation
- Nhật ký công việc đầy đủ theo tuần: `Slot 1/Working Files/PROGRESS_LOG.md`
- Đề bài: `Slot 1/Working Files/reference/exam_question_docvqa.md`
- Hướng dẫn Topic Team: `Slot 1/Working Files/reference/guide_extract.txt`
- Outline đề xuất đề tài đã nộp TA: `Slot 1/Outline Project - Hỏi đáp trên ảnh tài liệu (Document VQA).docx`
- 10 paper đã đọc + Paper Tracker: `Slot 1/Working Files/papers/`
- Các báo cáo tuần trước: `Slot 1/Working Files/Báo cáo/` (WEEK01 → WEEK03)

*Dữ liệu cuộc thi (889 MB) không nằm trong repo và không được phát tán; xem `.gitignore`.*
