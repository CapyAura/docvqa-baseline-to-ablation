"""Linear probe trên vision encoder ĐÓNG BĂNG — câu hỏi mở cuối cùng của nhánh D.

BỐI CẢNH
Zero-shot Qwen2.5-VL đạt 0,578 so với 0,818 của Cấu hình B (chọn đúng hàng
60/120 = mức đoán bừa) ⇒ đủ để loại phương án "dùng VLM có sẵn". Nhưng chưa đủ
để kết luận 82,99% là trần của TÍN HIỆU: model chọn theo VỊ TRÍ chứ không dùng
tín hiệu ảnh, và chép đúng chữ chỉ chứng minh OCR ổn, không chứng minh model
phân biệt được độ đậm.

PHÉP THỬ (TA chỉ định)
Lấy trung bình patch feature của vision encoder (ĐÓNG BĂNG, không fine-tune)
trên bbox từng hàng ứng viên, huấn luyện Logistic Regression, đánh giá trên 120
câu dev. Đối chứng: 5 phép đo độ dày nét + `stroke_ratio` qua CÙNG bộ phân loại.

Vì sao đây là phép thử tách được hai giả thuyết:
  bộ đọc tuyến tính vượt ~85%  -> tín hiệu CÓ trong feature của ViT; nút thắt
                                  nằm ở tầng so sánh của VLM ⇒ QLoRA mới có nghĩa.
  bộ đọc tuyến tính ~ mức đo nét -> ViT không giữ thêm gì so với một phép đo
                                  thủ công ⇒ đóng nhánh D, ghi kết luận âm.

LUẬT QUYẾT ĐỊNH (chốt TRƯỚC khi chạy)
  probe <= ~85%  -> ĐÓNG nhánh D.
  probe >= ~90%  -> mới cân nhắc QLoRA.
  ở giữa         -> đóng vì ngân sách, ghi rõ giới hạn.

BA PHA, HAI MÁY (máy local KHÔNG có runtime suy luận: torch/transformers/timm/
onnxruntime/tensorflow đều không cài, GPU 4.096 MiB):
  extract (local)  -> probe_d/lin_items.jsonl + probe_d/lin_crops.npz
  feats   (Colab)  -> probe_d/lin_feats.npz
  score   (local, sklearn)

    python code/probe_linear_bold.py extract
    # đồng bộ probe_d/ lên Drive rồi chạy cell Colab
    python code/probe_linear_bold.py score

ponytail: KHÔNG dựng lại đường chọn hàng. `resolve`/`candidates`/`crop_bbox`/
`gold_bold` của `probe_vlm_bold.py` là đúng đường nhánh BOLD của Cấu hình B, và
`cell_gray` của `probe_patch_bold.py` là đúng phép cắt ô. Chép lại là tự tạo
nguồn lệch thứ hai.
"""
import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

# Colab gắn stdout vào OutStream của ipykernel — không có `reconfigure`. Không
# guard thì `import` chết ngay dòng này, trước cả khi kịp chạy pha `feats`.
# `probe_vlm_bold.py` đã guard đúng chỗ này; file này phải theo cùng một lối.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import probe_vlm_bold as P                                          # noqa: E402
from grid_ocr import DATA, build_tables, load_jsonl, stroke_ratio   # noqa: E402
from probe_bold2 import metrics as stroke_metrics                   # noqa: E402
from probe_patch_bold import cell_gray                              # noqa: E402

SEED = P.SEED            # 20261007 — cùng seed với `probe_vlm_bold extract`
N_EVAL = P.N_SAMPLE      # 120 câu dev giữ riêng để đánh giá

# 5 phép đo của `probe_bold2` + phép đo Cấu hình B đang dùng. Cùng bộ phân loại.
CONTROL = ["ink/bbox", "run/h", "run", "core/ink", "core/h", "stroke_ratio"]
# ponytail: quét C thay vì chốt một giá trị. 1.280 chiều với 830 hàng train thì
# C quyết định lớn, và một con số duy nhất không cho người đọc biết trần thật.
CS = (0.003, 0.03, 0.3, 3.0)

# --------------------------------------------------------------- pha extract


def cmd_extract(args):
    """Local, KHÔNG cần model. -> lin_items.jsonl + lin_crops.npz (1070 ảnh hàng).

    ponytail: gói 1070 ảnh hàng vào MỘT .npz thay vì 1070 file PNG. Đồng bộ
    Drive một object thay vì một thư mục nghìn file — cùng dữ liệu, ít đường
    hỏng hơn. Nén tốt vì ảnh chữ đen trên nền trắng.
    """
    keep_dev = set((P.SPLITS / "dev_docs.txt").read_text(encoding="utf-8").split())
    labels = load_jsonl(DATA / "labels.jsonl")
    questions = {q["question_id"]: q["question"] for q in load_jsonl(DATA / "questions.jsonl")}
    ann = defaultdict(list)
    for a in load_jsonl(DATA / "cell_annotations.jsonl"):
        ann[a["document_id"]].append(a)

    vbl = [l for l in labels if l["reasoning_type"] == P.BOLD_TYPE]
    dev = [l for l in vbl if l["question_id"].rsplit("-q", 1)[0] in keep_dev]
    # Đúng 120 câu mà `probe_vlm_bold extract` đã lấy: cùng seed, cùng thứ tự lọc,
    # nên tập eval ở đây TRÙNG tập 120 câu đã chạy zero-shot — so được trực tiếp.
    random.Random(SEED).shuffle(dev)
    eval_ids = {l["question_id"] for l in dev[:N_EVAL]}

    tabs, docs = {}, defaultdict(set)
    for lab in vbl:
        doc = lab["question_id"].rsplit("-q", 1)[0]
        if doc not in tabs:
            mine = build_tables(doc)
            tabs[doc] = mine
            docs[doc] = set(mine)

    items, crops, skip = [], {}, defaultdict(int)
    for lab in vbl:
        qid = lab["question_id"]
        doc = qid.rsplit("-q", 1)[0]
        got = P.resolve(doc, questions[qid], tabs, docs)
        if not got:
            skip["khong chon duoc bang"] += 1
            continue
        table, page, tbl, ac, G = got
        cands = P.candidates(table, G)
        if len(cands) != 2:
            skip[f"{len(cands)} ung vien"] += 1
            continue
        boxes = [P.crop_bbox(table, [c["row"]]) for c in cands]
        if any(b is None for b in boxes):
            skip["khong co bbox"] += 1
            continue
        gold = P.gold_bold(table, cands, ann[doc])
        if gold is None:
            skip["khong co hang dam vang"] += 1
            continue
        for i, b in enumerate(boxes):
            crops[f"{qid}|{i}"] = cell_gray(doc, page, b)
        items.append({"qid": qid, "doc": doc, "page": page,
                      "rows": [list(b) for b in boxes], "gold_bold": gold,
                      "split": "eval" if qid in eval_ids else "train"})

    P.OUT.mkdir(exist_ok=True)
    with (P.OUT / "lin_items.jsonl").open("w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    np.savez_compressed(P.OUT / "lin_crops.npz", **crops)

    ev = [it for it in items if it["split"] == "eval"]
    tr = [it for it in items if it["split"] == "train"]
    print(f"vbl toàn bộ: {len(vbl)} · dùng được: {len(items)} "
          f"(train {len(tr)} / eval {len(ev)})")
    for k, v in sorted(skip.items()):
        print(f"  bỏ {v:4d}  {k}")
    print(f"crop -> {P.OUT/'lin_crops.npz'}  ({len(crops)} ảnh hàng)")

    # Chốt 1: tập eval phải ĐÚNG 120 câu — nếu ít hơn thì phép so với zero-shot
    # lệch mẫu và mọi con số sau đó không so được với 0,578.
    assert len(ev) == N_EVAL, f"eval {len(ev)} câu, phải đúng {N_EVAL}"
    # Chốt 2: không rò rỉ theo TÀI LIỆU. Giao thức chống overfit của dự án chia
    # theo tài liệu (nhiều câu dùng chung một tài liệu), nên chia theo câu là
    # chia sai — và sai im lặng: điểm vẫn in ra đẹp.
    leak = {it["doc"] for it in tr} & {it["doc"] for it in ev}
    assert not leak, f"rò rỉ {len(leak)} tài liệu giữa train và eval: {sorted(leak)[:5]}"
    print(f"chốt: eval đúng {N_EVAL} câu · trùng tài liệu train/eval = 0")

    old = P.OUT / "items.jsonl"
    if old.exists():
        old_ids = {json.loads(l)["qid"] for l in old.read_text(encoding="utf-8").splitlines() if l.strip()}
        same = old_ids == {it["qid"] for it in ev}
        print(f"đối chiếu 120 câu với items.jsonl của zero-shot: "
              f"{'TRÙNG khít' if same else 'LỆCH — xem lại seed/lọc'}")
        assert same, "tập eval không trùng tập 120 câu đã chạy zero-shot"


# --------------------------------------------------------------- pha feats


def _lin_items():
    p = P.OUT / "lin_items.jsonl"
    if not p.exists():
        raise SystemExit(f"thiếu {p} — chạy `extract` ở máy local trước.")
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def _npz(path):
    if not Path(path).exists():
        raise SystemExit(f"thiếu {path} — đồng bộ probe_d/ lên Drive rồi chạy lại.")
    return np.load(path)


def _visual(model):
    """-> module vision của Qwen2.5-VL. 5.x đổi chỗ gắn nên dò cả hai."""
    for holder in (model, getattr(model, "model", None)):
        v = getattr(holder, "visual", None)
        if v is not None:
            return v
    raise SystemExit(f"không thấy module vision. Thuộc tính có: {sorted(vars(model))}")


def cmd_feats(args):
    """Colab: encode từng hàng ứng viên, gộp trung bình patch -> vector.

    ponytail: cắt bbox hàng rồi encode RIÊNG, gộp trung bình mọi patch token.
    Cách còn lại là encode cả trang rồi ánh xạ bbox sang chỉ số patch — đúng
    nghĩa đen hơn nhưng phải tái dựng phép `smart_resize`, thêm một nguồn sai
    số. Hai hàng chịu CÙNG phép xử lý nên so sánh vẫn công bằng.
    """
    import torch
    from PIL import Image

    items = _lin_items()
    if args.n:
        items = items[:args.n]
    z = _npz(P.OUT / "lin_crops.npz")
    have = set(z.files)
    missing = [it["qid"] for it in items
               if f"{it['qid']}|0" not in have or f"{it['qid']}|1" not in have]
    if missing:
        raise SystemExit(
            f"{len(missing)} câu thiếu crop trong lin_crops.npz (Drive còn bản cũ?) "
            f"— vd {missing[:3]}. Đồng bộ lại probe_d/ rồi chạy lại.")

    model, proc, tag = P._load_model(False)
    vis = _visual(model)
    dt = next(vis.parameters()).dtype
    print(f"feats [{tag}] {type(vis).__name__} dtype={dt} · {len(items)} câu "
          f"x 2 hàng\n", flush=True)

    vecs, grid = {}, None
    for k, it in enumerate(items, 1):
        for i in (0, 1):
            pic = Image.fromarray(z[f"{it['qid']}|{i}"]).convert("RGB")
            inp = proc.image_processor(images=[pic], return_tensors="pt")
            thw = inp["image_grid_thw"].to(model.device)
            with torch.no_grad():
                out = vis(inp["pixel_values"].to(model.device, dtype=dt), grid_thw=thw)
            if not torch.is_tensor(out):
                out = out[0]
            vecs[f"{it['qid']}|{i}"] = out.float().mean(dim=0).cpu().numpy().astype(np.float32)
            if grid is None:
                grid = (tuple(int(x) for x in thw[0].tolist()),
                        pic.size, out.shape[0])
                # In NGAY ở hàng đầu: chạy trên Colab, nếu API vision đổi thì hỏng
                # ở đây, và lưới token cho biết `smart_resize` có nuốt ảnh không —
                # để sau 5 phút mới biết thì phí cả lượt GPU.
                print(f"  hàng đầu: lưới {grid[0]} · ảnh {pic.size} px · "
                      f"{out.shape[0]} patch · {out.shape[1]} chiều", flush=True)
        if k % 50 == 0:
            print(f"  {k}/{len(items)}", flush=True)

    np.savez_compressed(P.OUT / "lin_feats.npz", **vecs)
    g, size, dim = grid
    print(f"\nlưới token hàng đầu: {g} (ảnh {size[0]}x{size[1]} px) · {dim} chiều")
    print(f"ghi {len(vecs)} vector -> {P.OUT/'lin_feats.npz'}")
    # Chốt: 0 patch nghĩa là `smart_resize` đã ăn hết ảnh, và mọi vector sau đó
    # là trung bình của tập rỗng — vẫn ra số, vẫn ra bảng, vẫn ra quyết định.
    assert all(v.size for v in vecs.values()), "có vector rỗng"


# --------------------------------------------------------------- pha score


def _rows(its, get):
    """-> (X, y) mức HÀNG. Mỗi câu đóng góp 2 hàng: hàng vàng nhãn 1, hàng kia 0.

    ponytail: phân loại ở mức HÀNG rồi mới gộp lên mức CÂU. Train ở mức câu
    (vector hiệu hai hàng) cũng được, nhưng mức hàng cho bộ phân loại gấp đôi
    mẫu và đối xứng — không thiên vị hàng nào theo thứ tự.
    """
    X, y = [], []
    for it in its:
        for i in (0, 1):
            X.append(get(it, i))
            y.append(int(it["gold_bold"] == i))
    return np.asarray(X, np.float32), np.asarray(y)


def _pair_acc(score, items):
    """Điểm mỗi hàng -> (đúng theo câu, đúng theo hàng).

    Chọn hàng = argmax điểm trong cặp. Đây là con số đem so với 96/120 của
    Cấu hình B và 60/120 của zero-shot.
    """
    s = np.asarray(score, float).reshape(-1, 2)
    gold = np.array([it["gold_bold"] for it in items])
    pick = s.argmax(axis=1)
    return float((pick == gold).mean()), float(((s > 0.5).astype(int).ravel()
                                                == np.array([int(it["gold_bold"] == i)
                                                             for it in items
                                                             for i in (0, 1)])).mean())


def _fit(Xtr, ytr, Xte, items_te, C):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    clf = make_pipeline(StandardScaler(),
                        LogisticRegression(C=C, max_iter=2000))
    clf.fit(Xtr, ytr)
    return clf, _pair_acc(clf.predict_proba(Xte)[:, 1], items_te)


def cmd_score(args):
    """Local: huấn luyện trên train, đánh giá trên 120 câu eval. Không cần GPU."""
    items = _lin_items()
    tr = [it for it in items if it["split"] == "train"]
    ev = [it for it in items if it["split"] == "eval"]
    if not tr or not ev:
        raise SystemExit(f"train {len(tr)} / eval {len(ev)} — thiếu một vế, chạy `extract` lại.")

    F = _npz(P.OUT / "lin_feats.npz")
    z = _npz(P.OUT / "lin_crops.npz")
    have, have_c = set(F.files), set(z.files)
    miss = [it["qid"] for it in items
            if any(f"{it['qid']}|{i}" not in have or f"{it['qid']}|{i}" not in have_c
                   for i in (0, 1))]
    if miss:
        raise SystemExit(f"{len(miss)} câu thiếu feature/crop (vd {miss[:3]}) — "
                         f"chạy `feats` trên Colab với ĐỦ {len(items)} câu.")

    def probe(it, i):
        return F[f"{it['qid']}|{i}"]

    def control(it, i):
        c = z[f"{it['qid']}|{i}"]
        m = stroke_metrics(c)
        return [m.get(k, 0.0) for k in CONTROL[:5]] + [stroke_ratio(c)]

    print(f"train {len(tr)} câu ({2*len(tr)} hàng) · eval {len(ev)} câu ({2*len(ev)} hàng)\n")

    out = {}
    for name, get in (("probe (feature ViT)", probe), ("đối chứng (đo nét)", control)):
        Xtr, ytr = _rows(tr, get)
        Xte, _ = _rows(ev, get)
        print(f"{name}: {Xtr.shape[1]} chiều")
        print(f"  {'C':>8} {'câu đúng':>12} {'hàng đúng':>12}")
        best = None
        for C in CS:
            _, (qa, ra) = _fit(Xtr, ytr, Xte, ev, C)
            print(f"  {C:>8} {qa*100:>10.1f}% {ra*100:>10.1f}%")
            if best is None or qa > best[0]:
                best = (qa, ra, C)
        print()
        out[name] = best

    # Từng phép đo nét RIÊNG qua cùng bộ phân loại — để con số đối chứng không
    # phải một khối đen: phép nào kéo được, phép nào không, thấy ngay.
    Xtr, ytr = _rows(tr, control)
    Xte, _ = _rows(ev, control)
    print("từng phép đo nét, một mình, qua CÙNG bộ phân loại (C tốt nhất):")
    solo = {}
    for j, k in enumerate(CONTROL):
        best = max((_fit(Xtr[:, [j]], ytr, Xte[:, [j]], ev, C)[1][0], C) for C in CS)
        solo[k] = best[0]
        print(f"  {k:12s} {best[0]*100:6.1f}%  (C={best[1]})")
    print()

    pq, pr, pc = out["probe (feature ViT)"]
    cq, cr, cc = out["đối chứng (đo nét)"]
    print(f"TỔNG HỢP — chọn đúng hàng, đánh giá trên {len(ev)} câu eval:")
    print(f"  probe (feature ViT, C={pc})  {pq*100:6.1f}%")
    print(f"  đối chứng (đo nét, C={cc})   {cq*100:6.1f}%")
    print(f"  mốc đã đo: Cấu hình B (bỏ phiếu cột) 79,4% · zero-shot VLM 50,0% "
          f"(60/120) · trần nhánh BOLD 82,99%")
    print()

    # Lấy C tốt nhất trên chính tập eval là một thiên vị lạc quan nhỏ (4 giá trị
    # trên 120 câu). Cố ý: đây là phép đo TRẦN của nhánh D, phải cho D cơ hội
    # tốt nhất rồi mới đóng.
    if pq <= 0.85:
        v = ("ĐÓNG nhánh D. Bộ đọc tuyến tính trên feature của ViT không vượt "
             "được phép đo độ dày nét ⇒ 82,99% là trần của TÍN HIỆU, không phải "
             "của phép đo. QLoRA không có gì để mở ra.")
    elif pq >= 0.90:
        v = ("≥90% — bộ đọc tuyến tính đã vượt trần 82,99% ⇒ tín hiệu CÓ trong "
             "feature của ViT, nút thắt nằm ở tầng so sánh của VLM. Mới đáng cân "
             "nhắc QLoRA.")
    else:
        v = ("Ở GIỮA 85–90% — trên trần phép đo nhưng dưới ngưỡng đáng dựng QLoRA "
             "(±6–7 điểm trên 120 câu). Đóng vì ngân sách; ghi rõ giới hạn này "
             "thay vì kết luận âm dứt khoát.")
    print(f"QUYẾT ĐỊNH (luật chốt trước khi chạy): {v}")


# --------------------------------------------------------------- tự kiểm


def _demo():
    """Chốt phần dễ vỡ im lặng: cặp hàng -> nhãn, và phép chọn theo câu."""
    items = [{"qid": "q1", "gold_bold": 0}, {"qid": "q2", "gold_bold": 1}]
    y = np.array([int(it["gold_bold"] == i) for it in items for i in (0, 1)])
    assert y.tolist() == [1, 0, 0, 1], y
    assert _pair_acc([0.9, 0.1, 0.2, 0.8], items) == (1.0, 1.0)
    assert _pair_acc([0.1, 0.9, 0.8, 0.2], items) == (0.0, 0.0)
    print("demo OK: cặp hàng -> nhãn và chọn theo câu đúng")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["extract", "feats", "score"])
    ap.add_argument("--n", type=int, default=0, help="feats: chỉ chạy n câu đầu (thử)")
    args = ap.parse_args()
    {"extract": cmd_extract, "feats": cmd_feats, "score": cmd_score}[args.mode](args)


if __name__ == "__main__":
    _demo()
    main()
