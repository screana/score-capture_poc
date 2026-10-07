"""Read DAM's big decorative total score.

1. Anchor: the small "全国平均" text is read reliably, so use its box to locate
   the score band right above it (works even if the screen warp is slightly off).
2. Binarise by colour so the 3-D bevel / glow disappears and only the digit faces remain.
3. Keep the tall connected components (digits), drop the small "Ai感性ボーナス" text,
   render them black-on-white with even spacing and let the recogniser read that clean line.
"""
import json, re, sys
import cv2
import numpy as np
from rapidocr_onnxruntime import RapidOCR

eng = RapidOCR()


K = 5


def face_mask(band, mode):
    hsv = cv2.cvtColor(band, cv2.COLOR_BGR2HSV)
    h, s, v = [hsv[..., i].astype(int) for i in range(3)]
    b, g, r = [band[..., i].astype(int) for i in range(3)]
    if mode == "heart":            # white faces on a light-blue panel
        m = (s < 70) & (v > 205)
    else:                          # gold/yellow-white faces on dark blue
        warm = ((h < 45) | (h > 160)) & (v > 110) & (s > 40)   # yellow / orange / pink-ish
        white = (s < 60) & (v > 170)
        m = warm | white
    m = m.astype(np.uint8) * 255
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((K, K), np.uint8))


def read_total(img, boxes, mode):
    one = [b for b in boxes if b["scale"] == 1.0]
    anc = [b for b in one if "平均" in b["t"]]
    if not anc:
        return None, "全国平均が見つからない", None
    a = anc[0]
    ah = a["y1"] - a["y0"]
    y1 = int(a["y0"] + 0.1 * ah)
    y0 = max(0, int(a["y0"] - (12 if mode == "heart" else 10) * ah))
    x0 = max(0, int(a["x0"] - 2 * ah)); x1 = min(img.shape[1], int(a["x0"] + 36 * ah))
    # the score never extends past the right end of the date box on the same row
    dates = [b for b in one if re.search(r"20\d\d/\d", b["t"]) and abs(b["y0"] - a["y0"]) < 2 * ah]
    if dates and mode != "heart":
        x1 = min(x1, int(max(b["x1"] for b in dates) + 0.5 * ah))
    band = img[y0:y1, x0:x1]
    m = face_mask(band, mode)
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, 8)
    BH = band.shape[0]
    comps = [st[i] for i in range(1, n)
             if st[i, cv2.CC_STAT_HEIGHT] > 0.2 * BH          # ignore the small bonus text
             and st[i, cv2.CC_STAT_TOP] > 2                    # the radar chart touches the top edge
             and st[i, cv2.CC_STAT_WIDTH] < 1.3 * st[i, cv2.CC_STAT_HEIGHT]]
    if not comps:
        return None, "数字領域なし", m
    hmax = max(c[3] for c in comps)
    comps = sorted([c for c in comps if c[3] > 0.3 * hmax], key=lambda c: c[0])
    # digits = the first 5 glyph-sized components from the left (2 big + 3 decimal)
    digs = [c for c in comps if c[3] > 0.45 * hmax][:5]
    xa = min(c[0] for c in digs); xb = max(c[0] + c[2] for c in digs)
    ya = min(c[1] for c in digs); yb = max(c[1] + c[3] for c in digs)
    keep = np.zeros_like(m)
    for i in range(1, n):  # include the dot and fragments that fall inside the run's box
        x, y, w, h, a_ = st[i]
        if x >= xa - 2 and x + w <= xb + 2 and y >= ya - 2 and y + h <= yb + 2:
            keep[lab == i] = 255
    line = 255 - keep[ya:yb, xa:xb]
    line = cv2.resize(line, (int(line.shape[1] * 64 / line.shape[0]), 64), interpolation=cv2.INTER_AREA)
    line = cv2.copyMakeBorder(line, 16, 16, 16, 16, cv2.BORDER_CONSTANT, value=255)
    res, _ = eng.text_recognizer([cv2.cvtColor(line, cv2.COLOR_GRAY2BGR)])
    raw = res[0][0]
    mm = re.search(r"(\d{2})\D?(\d{3})", raw)
    if mm:
        return round(int(mm.group(1)) + int(mm.group(2)) / 1000, 3), f"認識={raw!r}", line
    return None, f"認識={raw!r}", line


if __name__ == "__main__":
    boxes = json.load(open("boxes.json")); gt = json.load(open("gt.json"))
    ok = 0; n = 0
    for k, g in gt.items():
        if g["machine"] != "DAM":
            continue
        img = cv2.imread(f"warped/{k}.jpg")
        mode = "heart" if "Heart" in g["mode"] else "ai"
        val, why, dbg = read_total(img, boxes[k], mode)
        if dbg is not None:
            cv2.imwrite(f"crops/digits_{k}.png", dbg)
        n += 1; ok += val == g["total"]
        print(k, mode, "正解", g["total"], "読取", val, "|", why)
    print(f"{ok}/{n}")
