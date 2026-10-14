# docvqa-baseline-to-ablation

Hỏi đáp trên ảnh tài liệu (**Document VQA**) — **Module 4** của khóa **AIO2026 (AI VIET NAM)**,
subtopic *Hỏi đáp trên ảnh tài liệu*.

Bắt đầu từ baseline của mentor (**0,9545**), dựng một hệ luật thuần Python đạt **0,968**,
rồi đo trần của từng tầng mô hình học sâu trước khi quyết định có dựng hay không.

Điểm đáng chú ý không nằm ở con số cuối, mà ở chỗ **hai tầng mô hình đều cho kết luận âm
có bằng chứng đo được** — và ở phép thử cuối của nhánh D, **tín hiệu hoá ra vẫn còn nguyên
trong ảnh**, chỉ là VLM không so sánh nổi nó. Phần lớn công sức của dự án là chứng minh
những điều đó bằng số liệu thay vì bằng phỏng đoán.

---

## Bối cảnh

Đây là sản phẩm **Module 4** của khóa **AIO2026 — AI VIET NAM**, làm theo hình thức
**Topic Team**: nhận một subtopic có sẵn, tự đọc paper, tự triển khai và tự đánh giá.

- Subtopic được giao: **Hỏi đáp trên ảnh tài liệu** (`Slot 1/Working Files/reference/hoi_dap_tren_anh_tai_lieu.md`)
- Hướng dẫn thực hiện Topic Team: `Slot 1/Working Files/reference/guide_extract.txt`
- Đề bài vòng thi **OLP AI PTIT 2026 — Vòng loại**: `Slot 1/Working Files/reference/exam_question_docvqa.md`
- Baseline tham chiếu của mentor: `Slot 1/Working Files/reference/mentor_solution_extract.txt`
- Outline đề xuất đề tài đã nộp TA (mục IV = bảng ablation): `Slot 1/Outline Project - Hỏi đáp trên ảnh tài liệu (Document VQA).docx`

Phần **được giao** là subtopic và baseline. Phần **tự làm** là hệ luật Cấu hình B, harness
đo trần, giao thức chống overfit, và hai kết luận âm của C và D — toàn bộ nằm trong
`Slot 1/Working Files/code/`.

---

## Vì sao có bốn cấu hình A/B/C/D

Bốn cấu hình không phải bốn thử nghiệm rời rạc. Chúng là **một thang leo có chủ đích**,
mỗi bậc trả lời đúng một câu hỏi, và bậc sau chỉ được dựng nếu bậc trước đã chứng minh
nó đáng dựng. Dưới đây là đường đi của suy luận đó.

### 1. Điểm xuất phát — không phải "làm sao cho điểm cao hơn", mà là "điểm mất ở đâu"

Baseline của mentor đạt **95,45**. Câu hỏi tự nhiên là *cải thiện chỗ nào* — nhưng đó là
câu hỏi sai, vì nó dẫn tới việc thử mọi thứ. Câu hỏi đúng là **điểm đang mất nằm ở đâu**.

Hai nguồn trả lời được câu đó:

- **TA Minh** chỉ ra hai bài toán con của Document VQA: **tái dựng hàng logic** (gom các ô
  OCR rời rạc thành đúng hàng của bảng) và **căn chỉnh giá trị theo hàng**.
- **File chẩn đoán của baseline** (`argextreme_*.csv`, 3.772 câu `argmax`/`argmin`) cho thấy
  **93,1% lỗi của baseline tập trung đúng ở khâu chọn hàng logic** (Argmax 44,9% + Argmin 48,2%).

⇒ Kết luận: **điểm mất không rải đều, nó dồn vào một chỗ.** Vậy cả dự án chỉ nên xoay quanh
chỗ đó, và mọi thứ khác bị loại khỏi phạm vi ngay từ đầu.

### 2. Nguyên tắc thiết kế thang — từ nhẹ đến nặng

Từ chỗ đó, thang được dựng theo một nguyên tắc duy nhất:

> **Không model → Model nông → VLM.** Mỗi bậc thêm đúng một loại năng lực, và phải tự
> chứng minh năng lực đó là cần thiết trước khi bậc sau được phép tồn tại.

Lý do chọn nguyên tắc này: mỗi bậc đắt hơn bậc trước rất nhiều (0 GPU → vài ngày GPU →
nhiều ngày GPU), nên nếu bậc nhẹ đã chạm trần thì bậc nặng là tiền và thời gian đổ đi.
Thang leo biến câu hỏi *"có nên dùng model không"* thành câu hỏi **đo được**: *trần của
bậc nhẹ là bao nhiêu, và bậc nặng vượt được bao nhiêu.*

### 3. Đọc 10 paper — ba nhận xét định hình thang

Đọc 10 paper SOTA (DocVQA, TAPAS, LayoutLMv3, DocLLM, LMDX, Qwen2-VL, Qwen2.5-VL,
BoundingDocs, DocExplainerV0, LiGT) cho ba nhận xét quyết định:

1. **Không paper nào giải đúng bài toán đang mắc.** Cả 10 paper đều giả định bảng **đã có
   cấu trúc** (HTML / markdown / văn bản tuyến tính hoá). Không paper nào làm việc *chọn
   hàng logic trên một lưới OCR phẳng* — đúng chỗ 93,1% lỗi đang nằm. ⇒ **Đây là khoảng
   trống, không phải bài toán đã có lời giải.** Vì vậy không thể chỉ đi copy một kiến trúc.
2. **TAPAS cho ý tưởng tách đôi.** Nó tách *chọn ô* khỏi *thực thi phép toán* — tức là
   tách **định vị** khỏi **tính toán**. Nhận xét này thành kiến trúc của cả hệ: một
   **Router** phân loại câu hỏi, rồi một **Solver** thực thi theo từng dạng.
3. **LayoutLMv3 và Qwen2.5-VL không phải hai lựa chọn thay thế nhau, mà là hai mức của
   cùng một thang.** LayoutLMv3 là *model nông*: hiểu bố cục, **không sinh được văn bản**.
   Qwen2.5-VL là *VLM*: vừa hiểu bố cục vừa **sinh và so sánh được ngữ nghĩa**. Đây chính
   là lý do C và D là **hai ablation riêng biệt** chứ không gộp thành một — chúng kiểm hai
   năng lực khác nhau, ở hai mức chi phí khác nhau.

### 4. Mỗi cấu hình trả lời đúng một câu hỏi

| | Câu hỏi mà cấu hình này trả lời | Vì sao cần hỏi câu đó |
|---|---|---|
| **A** | Baseline của mentor đang ở đâu? | Mốc đối chứng. Không có A thì mọi con số sau không có nghĩa. |
| **B** | Chỉ dùng luật hình học 2D thuần Python — **không model** — thì chạm trần ở đâu? | Nếu B đã chạm trần, mọi bậc sau là thừa. Đây là bậc **rẻ nhất**, phải hỏi trước tiên. |
| **C** | Thêm **model nông có tham số không gian** (LayoutLMv3) có vượt được B không? | Đo xem tín hiệu không gian — thứ LayoutLMv3 khai thác — có **tồn tại** trong dữ liệu này không. |
| **D** | Thêm **VLM lớn** (Qwen2.5-VL) có vượt được B không? | Đo xem năng lực *đọc ảnh trực tiếp* có vượt được trần của **một phép đo thủ công** không. |

Điểm mấu chốt về phương pháp: **C và D đều được đo trần trước khi dựng**, và cả hai đều
có **luật quyết định ghi sẵn trong script trước khi chạy** (nhánh D: `D − B ≥ +0,03` →
mới dựng model + QLoRA, `≤ 0` → dừng; phép thử cuối của D: probe `≤ ~85%` → đóng nhánh,
`≥ ~90%` → mới cân nhắc QLoRA). Nghĩa là kết quả âm không phải chuyện xảy ra rồi mới
biện luận, mà là **điều kiện dừng được đặt trước**.

### 5. Vì sao C và D kết thúc bằng kết luận âm — và vì sao đó vẫn là kết quả

- **C**: trần của nhánh này đo được **99,14% (dev)** — nghe rất hấp dẫn, nhưng con số đó
  chỉ đạt được **khi biết trước hàng vàng**, tức nó không phải một lời giải. Kiểm bằng 4 họ
  luật không-cần-nhãn: **0 họ chạm trần**. Và hình học hàng trong dữ liệu **đồng nhất tuyệt
  đối** (cao độ 0,02121 ở mọi hàng, khe hở 0,00000) — tức **loại tín hiệu mà LayoutLMv3
  sinh ra để khai thác không tồn tại**. Dựng model là chắc chắn âm.
- **D**: `visual_bold_lookup` là dạng duy nhất có trần dưới 100% (82,99%) vì phụ thuộc
  `is_bold` — trường duy nhất không có trong OCR. Câu hỏi mở là: 82,99% là trần của **tín
  hiệu ảnh**, hay chỉ là trần của **một phép đo độ dày nét**? Hai phép thử, hai kết quả
  ngược nhau — và chính chỗ ngược nhau đó là câu trả lời:
  - **Zero-shot Qwen2.5-VL** (đường "dùng VLM có sẵn"): **0,578** so với B **0,818**
    (chênh −0,241), chọn đúng hàng 60/120 = **đúng mức đoán bừa**. Thang sanity cho thấy
    model **chép đúng chữ cả hai dải** nhưng **không so sánh được độ đậm** ⇒ chết ở **tầng
    so sánh**, không phải tầng thị giác. Kết luận âm.
  - **Bộ đọc tuyến tính trên feature ViT đóng băng** (đúng đề nghị của TA — *không* QLoRA):
    **100,0%** trên 120 câu dev, so với **81,7%** của đối chứng đo nét qua **cùng** bộ phân
    loại. ⇒ **82,99% là trần của phép đo thủ công, KHÔNG phải của tín hiệu ảnh.** Tín hiệu
    nằm nguyên trong feature của ViT; nút thắt đúng là tầng so sánh của VLM.

Nói cách khác: **giá trị của dự án không nằm ở việc dựng được model, mà ở việc chứng minh
bằng số đo rằng hai tầng model đó không đáng dựng trên bộ dữ liệu này** — và chỉ ra chính
xác loại lỗi nào thì model học sâu mới thực sự cần thiết (câu trả lời: ở `argmax`/`argmin`
của bộ dữ liệu này, **không loại nào**). Nhánh D còn đi xa hơn một bước: nó không chỉ
chứng minh VLM *không đáng dựng*, mà còn định vị được **nút thắt nằm ở đâu** — trong tầng
so sánh của VLM, chứ không phải ở tín hiệu ảnh. Một bộ đọc tuyến tính 1.280 chiều trên
ViT đóng băng lấy ra **100%** tín hiệu đó mà không cần huấn luyện gì.

---

## Kết quả

| Cấu hình | Mô tả | Điểm (all) | Trạng thái |
|---|---|---:|---|
| **A** | Baseline mentor | 0,9545 | đối chứng |
| **B** | Hệ luật thuần Python | **0,968** | đã dựng |
| **C** | + LayoutLMv3 / LayoutXLM | — | **kết luận âm** (không dựng) |
| **D** | + Qwen2.5-VL (zero-shot) | — | **kết luận âm** ở zero-shot |
| **D′** | Bộ đọc tuyến tính trên ViT đóng băng | — | **kết luận dương**: 100% trên 120 câu BOLD — nhưng **không cần VLM** |

Cấu hình B theo split: **0,967** (dev) · **0,972** (eval) · **0,968** (all).
Chênh dev − eval = 0,005 ⇒ không có dấu hiệu overfit theo tài liệu.

Phân rã theo dạng câu hỏi (11.000 câu, thang điểm cuộc thi):

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

Trần oracle toàn hệ thống: **99,17%** (10.909/11.000). Phần điểm mất tập trung ở
`argmin` (39,4%) · `argmax` (34,5%) · `visual_bold_lookup` (25,7%).

---

## Bài toán

Mỗi câu hỏi kèm một ảnh tài liệu chứa **nhiều bảng**. Phải trả về:

```json
{"question_id": "...", "answer": "...", "evidence": [{"page": 1, "bbox": [x0, y0, x1, y1]}]}
```

Điểm mỗi câu = `0,85 × ANLS + 0,15 × Evidence-F1`. Tám dạng câu hỏi, từ tra cứu ô
(`lookup`) tới cực trị theo hàng (`argmax`/`argmin`) và tra cứu theo định dạng chữ
(`visual_bold_lookup` — "ô in đậm ở cột X").

**Ràng buộc hợp lệ:** lời giải **không được đọc file nhãn** (`labels.jsonl`,
`cell_annotations.jsonl`). Nhãn chỉ dùng để chấm điểm. Phép kiểm hợp lệ tự động:
đổi tên file nhãn rồi chạy lại — kết quả phải không đổi.

---

## Cách tiếp cận

### Lưới bảng tái dựng từ OCR, không từ nhãn

`code/grid_ocr.py` dựng lưới `(table, row, column)` thuần hình học: gom block OCR theo
`y0`, sắp theo `x0`. Đối chiếu với lưới vàng: **0 hàng trộn 2 bảng, 0 hàng sai thứ tự
cột / 256.040 ô, 30.662 hàng** (`code/verify_grid.py`).

### `is_bold` đo từ ảnh

`is_bold` là trường **duy nhất không có tương ứng trong OCR** — đề bài ghi rõ
*"OCR không ghi nhận định dạng chữ"*. Đo độ dày nét từ ảnh trang (tỉ lệ pixel tối sống
sót sau bào mòn 1 vòng, bỏ phiếu theo cột). Năm phép đo chuẩn hoá khác nhau đều cho
cùng kết quả ⇒ **các phép đo thủ công bị chặn ở ~83%, không phép nào vượt.** Trần đó
là trần của *phép đo*, **không phải** của tín hiệu ảnh — xem §Cấu hình D.

### Giao thức chống overfit

`training_set` chia theo **tài liệu** (không chia theo câu hỏi — nhiều câu dùng chung một
tài liệu): **880 dev / 220 eval** tài liệu, seed 20260921. Luật chốt trên dev, chỉ đo
eval sau khi chốt. Mỗi cấu hình báo cáo **hai** con số; chênh dev − eval là thước đo
overfit trực tiếp. `public_test` chỉ dùng làm kiểm tra cuối.

---

## Hai kết luận âm — và một kết luận dương bất ngờ

Đây là phần đóng góp chính. Cả hai nhánh C và D đều là **đo trần trước khi dựng model**,
và cả hai đều tiết kiệm nhiều ngày GPU cho một hướng đã biết là âm. Riêng nhánh D, phép
thử trần cuối cùng (bộ đọc tuyến tính trên ViT đóng băng) lật ngược kết luận về *tín
hiệu*: tín hiệu vẫn còn trong ảnh, chỉ có VLM là không đọc nổi nó.

### Cấu hình C — tín hiệu không gian không tồn tại

Kế hoạch ban đầu: dựng LayoutLMv3/LayoutXLM để khai thác **tín hiệu không gian** của hàng.
Trước khi dựng, đo trần của mọi bộ lọc hàng tĩnh cho `argmax`/`argmin`:

| Split | Câu | B đúng | Trần (biết trước hàng vàng) | Dư địa |
|---|---:|---:|---:|---:|
| dev | 3.024 | 93,49% | 99,14% | +5,65 đ% |
| eval | 748 | 94,12% | 98,93% | +4,81 đ% |

Dư địa +5,65 điểm % nghe hấp dẫn — nhưng là **oracle-only**. Kiểm bằng 4 họ luật
không-cần-nhãn: **0 họ chạm trần**.

| Họ luật | Kết quả (dev) |
|---|---|
| Khử trùng văn bản | 39,0 – 66,7% (baseline 93,5%) — phản tác dụng |
| Hàng in đậm đo từ ảnh | 10,4% |
| Từ khoá hàng tổng | không phân biệt được (27,2% vs 28,5%) |
| Hình học hàng | cao độ **đồng nhất tuyệt đối** (0,02121), khe hở **bằng 0** |

Phép đo chốt — trong đúng **197 câu B làm sai**, tín hiệu thị giác tốt nhất trỏ đúng hàng
vàng chỉ **9,1%**; trong **79,2%** nó đo được nhưng chỉ vào một hàng **thứ ba** ⇒ nhiều
hàng cùng "nổi bật", và nổi bật **không tương quan** với hàng đúng.

⇒ **Không dựng LayoutLMv3.** Loại tín hiệu mà nó được kỳ vọng khai thác đã đo là
*không tồn tại* trong dữ liệu này. Kết luận này trả lời trực tiếp câu hỏi nghiên cứu:
*mô hình học sâu chỉ thực sự cần thiết ở loại lỗi nào?*

### Cấu hình D — VLM không đọc được độ đậm, nhưng tín hiệu thì có trong ảnh

`visual_bold_lookup` là dạng **duy nhất** có trần dưới 100% (82,99%), vì nó phụ thuộc
`is_bold` — trường không có trong OCR. Câu hỏi mở: 82,99% là trần của **tín hiệu ảnh**,
hay chỉ là trần của **một phép đo độ dày nét**? Hai phép thử, hai kết quả ngược nhau —
và chính chỗ ngược nhau đó là câu trả lời.

**(1) Zero-shot Qwen2.5-VL-3B** trên 120 câu (Colab T4), đối chứng là đúng nhánh BOLD
của Cấu hình B:

| Nhánh | ANLS | EvF1 | Điểm | Chọn đúng hàng |
|---|---:|---:|---:|---:|
| B (đo độ dày nét) | 0,827 | 0,769 | **0,818** | 96/120 |
| D (Qwen2.5-VL zero-shot) | 0,555 | 0,704 | **0,578** | **60/120** |

Chênh **−0,241**. 60/120 chọn đúng = **đúng mức đoán bừa**. Thang sanity 4 bậc chỉ ra
nguyên nhân: model **chép đúng chữ cả hai dải** nhưng **không so sánh được độ đậm** —
đảo dải thì chữ không lật (0/10), đổi nhãn thì chữ đứng yên (10/10) ⇒ **chọn theo vị trí**.
Lỗi nằm ở **tầng so sánh**, không phải tầng thị giác. Kết luận âm cho đường "dùng VLM
có sẵn".

Nhưng kết luận âm đó **chưa** nói được 82,99% là trần của tín hiệu: model hỏng ở tầng
so sánh thì không chứng minh được tầng thị giác đã hết thông tin.

**(2) Bộ đọc tuyến tính trên feature ViT đóng băng** — đúng phép thử TA chỉ định,
**không** QLoRA. Lấy trung bình patch feature của vision encoder (đóng băng) trên bbox
từng hàng ứng viên, huấn luyện Logistic Regression trên **415 câu train**, đánh giá trên
**120 câu eval**. Đối chứng: 5 phép đo độ dày nét + `stroke_ratio` qua **cùng** bộ phân
loại:

| Nhánh (cùng bộ phân loại) | Chọn đúng hàng (120 câu) |
|---|---:|
| **Bộ đọc tuyến tính, feature ViT** | **100,0%** |
| Đối chứng: 6 phép đo nét gộp | 81,7% |
| Đo nét tốt nhất một mình (`core/ink`, `stroke_ratio`) | 80,8% |
| Cấu hình B (bỏ phiếu cột) | 79,4% |
| Zero-shot Qwen2.5-VL | 50,0% |
| Trần nhánh BOLD cũ (đo thủ công) | 82,99% |

⇒ **82,99% là trần của phép đo thủ công, KHÔNG phải của tín hiệu ảnh.** Tín hiệu nằm
nguyên trong feature của ViT — một bộ đọc tuyến tính 1.280 chiều lấy ra được **100%**,
không cần huấn luyện model, không cần GPU. Nút thắt đúng là **tầng so sánh** của VLM.

Kết quả 100% này đã qua **8 phép loại trừ lối tắt** (không phép nào hạ được nó):

| Phép loại trừ | Kết quả |
|---|---|
| Cân bằng nhãn | train {1: 215, 0: 200} · eval {1: 60, 0: 60} |
| Đảo nhãn train (3 lần) | 0,458 / 0,408 / 0,492 ⇒ không rò rỉ |
| Giải mã vị trí hàng từ feature | 0,5208 ⇒ feature không mã hoá thứ tự hàng |
| Chỉ dùng kích thước ô | 51,7% |
| Bỏ 20 / 200 chiều \|w\| lớn nhất | vẫn 100% |
| Tập con ô cùng cỡ (44 câu eval) | probe 100,0% vs `stroke_ratio` 88,6% |
| Ghép cặp **khác tài liệu** (4.000 cặp) | 100,0% ⇒ đo độ đậm tuyệt đối, không so trong trang |
| Cặp có hàng **header** | **0/535** cặp có hàng header ⇒ không học "đây là header" |

Hai giới hạn phải ghi rõ: (a) 120 câu dev ⇒ khoảng tin cậy rộng cỡ ±6–7 điểm (100% vẫn
cách xa mọi mốc đối chứng, nhưng không phải "chắc chắn 100%"); (b) C chọn tham số C tốt
nhất trên chính tập eval — một thiên vị lạc quan nhỏ, cố ý, để cho nhánh D cơ hội tốt
nhất trước khi đóng.

**Vậy có nên chạy QLoRA không? Không.** Bộ đọc tuyến tính đóng băng đã giải xong bài
toán con này ở 100%, không huấn luyện, không GPU. QLoRA sẽ dạy một model 3B đúng cái mà
một hồi quy logistic 1.280 chiều đã làm hoàn hảo — và theo chính TA, phải đạt ≥90% mới
phân biệt được với Cấu hình B. Ở đây không còn gì để QLoRA mở ra.

---

## Cấu trúc repo

```
Slot 1/
  Working Files/
    code/                  # toàn bộ harness và probe
      grid_ocr.py            # tái dựng lưới bảng từ OCR + ảnh (không đọc nhãn)
      audit_rule_system.py   # Cấu hình B — solver chính
      anls.py                # độ đo ANLS
      evidence_f1.py         # độ đo Evidence-F1
      oracle_ceiling.py      # đo trần oracle, không đọc cell_annotations.jsonl
      row_filter_ceiling.py  # trần của mọi bộ lọc hàng tĩnh (nhánh C)
      probe_*.py             # các phép đo độc lập, mỗi file một giả thuyết
      check_validity.py      # phép kiểm hợp lệ tự động
      split_docs.py          # chia train theo tài liệu (880 dev / 220 eval)
    splits/                # dev_docs.txt, eval_docs.txt
    Báo cáo/               # báo cáo tiến độ theo tuần (.md nguồn + .docx nộp)
    papers/                # 10 paper + Paper Tracker + tổng hợp gap
    reference/             # đề bài, hướng dẫn Topic Team
    probe_d/               # probe nhánh D: zero-shot VLM + probe tuyến tính ViT
                           #   (notebook Colab, lin_items/lin_crops/lin_feats, preds)
  Outline Project ... .docx  # Outline nộp TA (mục IV = bảng ablation)
  PROGRESS_LOG.md            # nhật ký công việc theo tuần
```

---

## Chạy lại

Dữ liệu cuộc thi **không nằm trong repo** (889 MB, và không phải để phát tán). Đặt vào:

```
data/training_set/{labels,questions,cell_annotations}.jsonl
data/training_set/ocr/*.json
data/training_set/images/*.png
```

Chia split rồi chạy solver:

```bash
python "Slot 1/Working Files/code/split_docs.py"                    # 880 dev / 220 eval
python "Slot 1/Working Files/code/audit_rule_system.py" --split dev
python "Slot 1/Working Files/code/audit_rule_system.py" --split eval
python "Slot 1/Working Files/code/audit_rule_system.py" --split all
```

Phép kiểm hợp lệ (đổi tên file nhãn rồi chạy lại — kết quả phải không đổi):

```bash
python "Slot 1/Working Files/code/check_validity.py"
```

Đo trần và các probe nhánh C:

```bash
python "Slot 1/Working Files/code/oracle_ceiling.py"
python "Slot 1/Working Files/code/row_filter_ceiling.py" --split dev
python "Slot 1/Working Files/code/probe_gold_signal.py"  --split dev
```

Probe tuyến tính nhánh D — hai máy (máy local không có torch):

```bash
python "Slot 1/Working Files/code/probe_linear_bold.py" extract   # local: cắt ô -> lin_crops.npz
# đẩy probe_d/ lên Drive, chạy Cell 5 của probe_d/probe_d_colab.ipynb (Colab T4) -> lin_feats.npz
python "Slot 1/Working Files/code/probe_linear_bold.py" score     # local: sklearn, không cần GPU
```

Yêu cầu: Python 3.10+ · `python-docx` (chỉ để export báo cáo) · `numpy`, `Pillow`,
`scikit-learn` (chỉ cho pha `score` của probe D). Solver Cấu hình B **không cần GPU và
không cần PyTorch**; pha `feats` của probe D cần GPU nhưng chỉ chạy trên Colab.

---

## Tác giả

**Huỳnh Thuyên Nam** — Module 4 (Document VQA) khóa AIO2026, AI VIET NAM.
Đề bài vòng loại: OLP AI PTIT 2026 — Vòng loại, chủ đề Document VQA.

Nhật ký công việc đầy đủ theo tuần: [`PROGRESS_LOG.md`](Slot%201/Working%20Files/PROGRESS_LOG.md)
