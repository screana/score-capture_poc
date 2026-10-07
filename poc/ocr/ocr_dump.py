"""Run RapidOCR on each straightened screen at two scales and save all text boxes."""
import sys, os, glob, json
import cv2
from rapidocr_onnxruntime import RapidOCR

eng = RapidOCR()
src, out = sys.argv[1], sys.argv[2]
allres = {}
for f in sorted([p for p in glob.glob(os.path.join(src, "*.jpg")) if not os.path.basename(p).startswith("dbg_")]):
    img = cv2.imread(f)
    boxes = []
    for s in (1.0, 0.5):
        im = img if s == 1.0 else cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        res, _ = eng(im)
        for b, t, c in res or []:
            xs = [p[0] / s for p in b]; ys = [p[1] / s for p in b]
            boxes.append({"t": t, "c": round(float(c), 3), "scale": s,
                          "x0": min(xs), "y0": min(ys), "x1": max(xs), "y1": max(ys)})
    allres[os.path.basename(f)[:-4]] = boxes
    print(os.path.basename(f), len(boxes), file=sys.stderr)
json.dump(allres, open(out, "w"), ensure_ascii=False, indent=1)
