"""Turn raw OCR boxes into structured karaoke records, with self-checks."""
import re, json, sys
import numpy as np
import cv2

IMG = None
HEART = False

NUM = r"(?<!\d)(\d{1,2})[\.,]?(\d{3})"   # "32.200", "32,200" and the dot-less "32200"


def f3(a, b):
    return round(int(a) + int(b) / 1000, 3)


def classify(boxes, img):
    text = "".join(b["t"] for b in boxes)
    if re.search(r"JOY|分析[採探]点|/40", text):
        return "JOYSOUND", "分析採点AI"
    if re.search(r"月次|人中|ハート", text):
        return "DAM", "精密採点Ai Heart"
    if re.search(r"感性|DAM|全国平均|全.平均", text):
        return "DAM", "精密採点Ai"
    # fall back on colour: JOYSOUND screens are pink/magenta, DAM screens blue
    hue = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[..., 0]
    pink = np.mean((hue > 140) | (hue < 10))
    return ("JOYSOUND", "分析採点AI") if pink > 0.3 else ("DAM", "?")


def parse_datetime(text):
    m = re.search(r"(20\d\d)/(\d{1,2})/(\d{1,4}):(\d{2})", text.replace(" ", ""))
    if m:
        y, mo, dh, mi = m.groups()
        for k in (1, 2):  # split "817" / "1014" into day + hour
            d, h = dh[:k], dh[k:]
            if d and 1 <= len(h) <= 2 and 1 <= int(d) <= 31 and 0 <= int(h) <= 23:
                return f"{y}-{int(mo):02d}-{int(d):02d} {int(h):02d}:{mi}"
    m = re.search(r"(20\d\d)/(\d{1,2})/(\d{1,2})", text)
    return f"{m.group(1)}-{int(m.group(2)):02d}-{int(m.group(3)):02d}" if m else None


def joysound(boxes):
    r = {}
    keys = {"40": "pitch", "30": "stability", "15": "dynamics", "10": "longtone", "5": "technique"}
    for b in sorted(boxes, key=lambda b: -b["c"]):
        if re.search(r"20\d\d/\d", b["t"]):
            continue
        m = re.search(NUM + r"\s*/\s*(40|30|15|10|5)(?!\d)", b["t"])
        if m and keys[m.group(3)] not in r:
            r[keys[m.group(3)]] = f3(m.group(1), m.group(2))
    # AI bonus: "+10.000点" (sometimes split as "+10." and "0.000点")
    right = [b for b in boxes if b["x0"] > 900 and b["y0"] < 500]
    for b in sorted(right, key=lambda b: -b["c"]):
        m = re.search(r"\+\s*" + NUM + r"点", b["t"])
        if m:
            r["bonus"] = f3(m.group(1), m.group(2)); break
    else:  # split as "+9." and ".583点"
        joined = "".join(b["t"] for b in sorted(right, key=lambda b: b["x0"]) if b["scale"] == 1.0)
        m = re.search(r"\+\s*(\d{1,2})\.+\s*\d?\.?(\d{3})", joined)
        if m:
            r["bonus"] = f3(m.group(1), m.group(2))
    r["date"] = parse_datetime("".join(b["t"] for b in boxes if b["x0"] > 1000 and b["y0"] < 200))
    # decimal part of the total, shown under the big integer: ".652点"
    for b in boxes:
        m = re.fullmatch(r"\.?(\d{3})点?", b["t"].strip())
        if m and 450 < b["x0"] < 1050 and b["y0"] < 500:
            r["total_decimal"] = m.group(1)
    # technique counts: the three "N回" boxes in the lower-left panel, top to bottom
    hits = sorted(((b["y0"], int(m.group(1))) for b in boxes if 150 < b["x0"] < 700 and b["y0"] > 520
                   for m in [re.search(r"(\d{1,2})\s*回", b["t"])] if m))
    rows = []
    for y, v in hits:  # one value per row (both scales see the same row)
        if not rows or y - rows[-1][0] > 30:
            rows.append((y, v))
    if len(rows) == 3:
        r["kobushi"], r["shakuri"], r["vibrato"] = [v for _, v in rows]
    # the breakdown + bonus must add up to the total, so we can compute it AND check it
    parts = [r.get(k) for k in ("pitch", "stability", "dynamics", "longtone", "technique", "bonus")]
    if None not in parts:
        r["total"] = round(sum(parts), 3)
        dec = r.get("total_decimal")
        r["check"] = "OK(小数部一致)" if dec and f"{r['total']:.3f}".endswith(dec) else ("小数部読めず" if not dec else "NG(不一致)")
    else:
        r["check"] = "内訳が欠けている"
    return r


def dam(boxes):
    r = {}
    one = [b for b in boxes if b["scale"] == 1.0]
    text = " ".join(b["t"] for b in one)
    r["date"] = parse_datetime(text)

    def near(label_re, val_re, dx=700, dy=40):
        for lb in one:
            if re.search(label_re, lb["t"]):
                m = re.search(label_re + r".*?" + val_re, lb["t"])
                if m:
                    return m
                cy = (lb["y0"] + lb["y1"]) / 2
                for b in sorted(one, key=lambda b: b["x0"]):
                    if b is lb or b["x0"] < lb["x0"] - 10 or b["x0"] > lb["x1"] + dx:
                        continue
                    if re.search(r"最高|前回|平均", b["t"]):  # another label's value
                        continue
                    if abs((b["y0"] + b["y1"]) / 2 - cy) < dy:
                        m = re.search(val_re, b["t"])
                        if m:
                            return m
        return None

    m = near(r"平均", r"(\d{2})[\.,]?(\d{3})点")
    if not m:  # label missed: an "xx.xxx点" glued to the date is the national average
        m = re.search(r"(\d{2})[\.,](\d{3})点\s*20\d\d/", text)
    if m:
        r["national_avg"] = f3(m.group(1), m.group(2))
    m = near(r"最高", r"(\d{2})[\.,]?(\d{3})")
    r["best"] = f3(m.group(1), m.group(2)) if m else None
    m = near(r"前回", r"(\d{2})[\.,]?(\d{3})", dx=300)
    r["prev"] = f3(m.group(1), m.group(2)) if m else None
    m = re.search(r"(\d{1,3})位", text); n = re.search(r"(\d{1,3})人中", text)
    if m and n:
        r["rank"], r["rank_of"] = int(m.group(1)), int(n.group(1))
    # bonus: the only "d.ddd点" number in the upper half
    for b in sorted(boxes, key=lambda b: -b["c"]):
        m = re.fullmatch(r"(\d)[\.,](\d{3})点?", b["t"].strip())
        if m and b["y0"] < 450:
            r["bonus"] = f3(m.group(1), m.group(2)); break
    # total: big decorative digits -> anchor on 全国平均, colour-binarise, recognise the clean line
    import dam_total
    val, why, _ = dam_total.read_total(IMG, boxes, "heart" if HEART else "ai")
    best = r.get("best")
    if val is not None and best is not None and val > best + 1e-9:
        # the total can never exceed the personal best shown on the same screen
        if f"{val:.3f}"[-3:] == f"{best:.3f}"[-3:]:
            r["total"], r["check"] = best, f"読取{val}が最高点超え→小数部一致の最高点{best}に補正"
        else:
            r["total"], r["check"] = None, f"読取{val}が最高点{best}と矛盾→VLMへ"
    elif val is not None:
        r["total"], r["check"] = val, "数字を二値化して読取" + ("（最高点と一致）" if best == val else "")
    else:
        r["total"], r["check"] = None, f"読めず({why})→VLMへ"
    return r


def title_artist(boxes, machine):
    """Song title / artist: the Chinese PP-OCR model is weak at kana, so these are best-effort."""
    one = [b for b in boxes if b["scale"] == 1.0 and b["y0"] < 200 and b["x1"] < 800]
    one = [b for b in one if not re.search(r"曲名|歌手名|音程|安定|/40|^\W*$", b["t"])]
    one.sort(key=lambda b: b["y0"])
    return [b["t"] for b in one[:2]]


if __name__ == "__main__":
    allb = json.load(open(sys.argv[1]))
    out = {}
    for k, boxes in allb.items():
        img = cv2.imread(f"warped/{k}.jpg")
        machine, mode = classify(boxes, img)
        IMG, HEART = img, "Heart" in mode
        r = joysound(boxes) if machine == "JOYSOUND" else dam(boxes)
        r = {"machine": machine, "mode": mode, **r, "title_artist_raw": title_artist(boxes, machine)}
        out[k] = r
    json.dump(out, open(sys.argv[2], "w"), ensure_ascii=False, indent=1)
