"""Tell the karaoke machine / scoring mode from the screen image alone.

Features (on the straightened screen, or the whole image when no screen is found):
  - colour histogram in HSV (hue 18 x sat 3 x val 3), L1-normalised
  - colour layout: mean Lab colour of a 6x4 grid (where the dark/bright/pink/blue panels sit)
  - edge layout: edge density of a 6x4 grid (where text, charts and panels are)
Classifier: nearest neighbour by cosine similarity, each block weighted equally.
"""
import sys, os, glob, json
import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
POC = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(POC, "ocr"))
import screen  # noqa: E402

GW, GH = 6, 4


def rectify(img):
    r = screen.find_screen(img)
    if r is not None:
        q = r[1]
        area = cv2.contourArea(q.astype(np.float32)) / (img.shape[0] * img.shape[1])
        if 0.12 < area < 0.97:
            return screen.warp(img, q), True
    return cv2.resize(img, (1600, 900), interpolation=cv2.INTER_AREA), False


def features(scr):
    s = cv2.resize(scr, (192, 108), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(s, cv2.COLOR_BGR2HSV)
    h = cv2.calcHist([hsv], [0, 1, 2], None, [18, 3, 3], [0, 180, 0, 256, 0, 256]).ravel()
    h /= h.sum() + 1e-9
    lab = cv2.cvtColor(s, cv2.COLOR_BGR2LAB).astype(np.float32)
    grid = cv2.resize(lab, (GW, GH), interpolation=cv2.INTER_AREA).ravel() / 255.0
    g = cv2.cvtColor(s, cv2.COLOR_BGR2GRAY)
    e = cv2.Canny(g, 60, 160).astype(np.float32) / 255.0
    edge = cv2.resize(e, (GW, GH), interpolation=cv2.INTER_AREA).ravel()
    blocks = []
    for v in (np.sqrt(h), grid - grid.mean(), edge - edge.mean()):
        blocks.append(v / (np.linalg.norm(v) + 1e-9))
    return np.concatenate(blocks)


def knn(train_X, train_y, x, k=1):
    sims = train_X @ x   # sum of 3 block cosines (max 3)
    order = np.argsort(-sims)[:k]
    votes = {}
    for i in order:
        votes[train_y[i]] = votes.get(train_y[i], 0) + sims[i]
    best = max(votes, key=votes.get)
    return best, float(sims[order[0]] / 3)


if __name__ == "__main__":
    lab = json.load(open(os.path.join(HERE, "labels.json")))
    cls_of = {}
    for c, ids in lab.items():
        if c.startswith("_") or c.startswith("excluded"):
            continue
        for i in ids:
            cls_of[i] = c
    X, y, names = [], [], []
    for f in sorted(glob.glob(os.path.join(POC, "collect", "raw", "*.jpg"))):
        idx = f.split("__")[1][:4]
        if idx not in cls_of:
            continue
        img = cv2.imread(f)
        scr, found = rectify(img)
        X.append(features(scr)); y.append(cls_of[idx]); names.append(idx)
    X = np.array(X)

    # 1) leave-one-out on the web images
    ok = 0
    print("== Web画像で leave-one-out ==")
    for i in range(len(X)):
        m = np.ones(len(X), bool); m[i] = False
        p, s = knn(X[m], [y[j] for j in range(len(y)) if m[j]], X[i])
        ok += p == y[i]
        if p != y[i]:
            print(f"  ✗ {names[i]} 正解={y[i]} 予測={p} (類似度 {s:.2f})")
    print(f"  {ok}/{len(X)}")

    # 2) train on web images only, test on the user's own photos
    gt = json.load(open(os.path.join(POC, "ocr", "gt.json")))
    user = {k: ("joy_ai" if v["machine"] == "JOYSOUND" else "heart" if "Heart" in v["mode"] else "dam_ai") for k, v in gt.items()}
    # extra_tests.json: {"name": ["path/to/photo.jpg", "dam_ai"], ...}  (optional, outside the repo)
    ex_path = os.path.join(HERE, "extra_tests.json")
    extra = json.load(open(ex_path)) if os.path.exists(ex_path) else {}
    tests = [(k, os.path.join(POC, "ocr", "imgs", f"{k}.jpg"), c) for k, c in user.items()] + [(k, p, c) for k, (p, c) in extra.items()]
    print("== Web画像だけで学習 → 手元の写真で判定 ==")
    ok = 0; mach_ok = 0
    machine = lambda c: "JOYSOUND" if c.startswith("joy") else "DAM"
    for k, p, c in tests:
        img = cv2.imread(p)
        scr, found = rectify(img)
        pred, s = knn(X, y, features(scr))
        ok += pred == c; mach_ok += machine(pred) == machine(c)
        print(f"  {'✓' if pred == c else '✗'} {k:12s} 正解={c:8s} 予測={pred:10s} 類似度 {s:.2f} 画面検出={'あり' if found else 'なし'}")
    print(f"  モード {ok}/{len(tests)} / 機種 {mach_ok}/{len(tests)}")
