"""Find the TV screen in a photo and warp it to a front-facing 1600x900 image."""
import sys, os, glob
import cv2
import numpy as np

OUT_W, OUT_H = 1600, 900


def order_pts(p):
    p = p.reshape(4, 2).astype(np.float32)
    s = p.sum(1); d = np.diff(p, axis=1).ravel()
    return np.array([p[s.argmin()], p[d.argmin()], p[s.argmax()], p[d.argmax()]], np.float32)


def fit_quad(hull):
    peri = cv2.arcLength(hull, True)
    for eps in np.linspace(0.005, 0.08, 30):
        a = cv2.approxPolyDP(hull, eps * peri, True)
        if len(a) == 4:
            return a.reshape(4, 2).astype(np.float32)
    return cv2.boxPoints(cv2.minAreaRect(hull)).astype(np.float32)


def refine_quad(quad, contour):
    """Re-fit each side as a least-squares line through the contour points
    closest to it, then intersect neighbouring sides (fixes clipped corners)."""
    pts = contour.reshape(-1, 2).astype(np.float32)
    q = order_pts(quad)
    lines = []
    for i in range(4):
        a, b = q[i], q[(i + 1) % 4]
        ab = b - a; L = np.linalg.norm(ab) + 1e-6
        n = np.array([-ab[1], ab[0]]) / L
        t = ((pts - a) @ ab) / L ** 2
        dist = np.abs((pts - a) @ n)
        sel = pts[(t > 0.15) & (t < 0.85) & (dist < 0.03 * L)]
        if len(sel) < 10:
            return q
        vx, vy, x0, y0 = cv2.fitLine(sel, cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
        lines.append((np.array([x0, y0]), np.array([vx, vy])))
    out = []
    for i in range(4):
        (p1, d1), (p2, d2) = lines[i - 1], lines[i]
        A = np.array([d1, -d2]).T
        if abs(np.linalg.det(A)) < 1e-6:
            return q
        s_, _ = np.linalg.solve(A, p2 - p1)
        out.append(p1 + s_ * d1)
    return order_pts(np.array(out, np.float32))


def find_screen(img):
    """The screen is a bright region fenced in by the TV's black bezel.
    Make a 'dark' mask (bezel + dark walls), take the bright connected regions
    it separates, fill their holes, and pick the one that is large and looks
    most like a quadrilateral (area close to the fitted quad's area)."""
    h, w = img.shape[:2]
    scale = 800 / w
    small = cv2.resize(img, None, fx=scale, fy=scale)
    sh, sw = small.shape[:2]
    v = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)[..., 2]
    def candidates():
        # (a) bright regions separated by dark pixels
        for thr in (35, 50, 65, 80):
            bright = (v > thr).astype(np.uint8)
            bright = cv2.morphologyEx(bright, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
            n, lab, stats, _ = cv2.connectedComponentsWithStats(bright, 8)
            for i in range(1, n):
                if stats[i, cv2.CC_STAT_AREA] < 0.08 * sh * sw:
                    continue
                comp = (lab == i).astype(np.uint8) * 255
                cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                yield thr, max(cnts, key=cv2.contourArea)
        # (b) holes inside the dark bezel ring (works when walls are bright)
        for thr in (40, 55, 70, 90):
            dark = (v < thr).astype(np.uint8) * 255
            dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
            cnts, hier = cv2.findContours(dark, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
            if hier is None:
                continue
            for c, hh in zip(cnts, hier[0]):
                if hh[3] != -1 and cv2.contourArea(c) > 0.08 * sh * sw:
                    yield -thr, c

    best = None
    for thr, c in candidates():
        if True:
            hull = cv2.convexHull(c)
            quad = fit_quad(hull)
            qa = cv2.contourArea(quad)
            ca = cv2.contourArea(c)  # outer contour = holes filled
            if qa <= 0:
                continue
            area = ca / (sh * sw)
            rect = min(ca, qa) / max(ca, qa)  # 1.0 = perfect quadrilateral
            # screens are landscape and roughly 16:9 once warped; reject slivers
            o = refine_quad(quad, c)
            wd = (np.linalg.norm(o[1] - o[0]) + np.linalg.norm(o[2] - o[3])) / 2
            ht = (np.linalg.norm(o[3] - o[0]) + np.linalg.norm(o[2] - o[1])) / 2
            ar = wd / max(ht, 1)
            if not (1.1 < ar < 3.2) or area > 0.85:
                continue
            # quads hugging 3+ photo borders are almost always the wall, not the TV
            m = 4
            touch = sum([o[:, 0].min() < m, o[:, 1].min() < m,
                         o[:, 0].max() > sw - m, o[:, 1].max() > sh - m])
            if touch >= 3 or rect < 0.85:
                continue
            score = rect ** 4 * np.sqrt(area)
            if best is None or score > best[0]:
                best = (score, o / scale, thr, round(rect, 3), round(area, 3))
    return None if best is None else best


def warp(img, quad):
    dst = np.array([[0, 0], [OUT_W, 0], [OUT_W, OUT_H], [0, OUT_H]], np.float32)
    M = cv2.getPerspectiveTransform(quad, dst)
    return cv2.warpPerspective(img, M, (OUT_W, OUT_H), flags=cv2.INTER_CUBIC)


if __name__ == "__main__":
    src, out = sys.argv[1], sys.argv[2]
    os.makedirs(out, exist_ok=True)
    for f in sorted(glob.glob(os.path.join(src, "*.jpg"))):
        img = cv2.imread(f)
        r = find_screen(img)
        name = os.path.basename(f)
        if r is None:
            print(name, "NO SCREEN"); continue
        q = r[1]
        print(name, "thr", r[2], "rect", r[3], "area", r[4])
        dbg = img.copy()
        cv2.polylines(dbg, [q.astype(int)], True, (0, 255, 0), 8)
        cv2.imwrite(os.path.join(out, "dbg_" + name), cv2.resize(dbg, None, fx=0.3, fy=0.3))
        cv2.imwrite(os.path.join(out, name), warp(img, q))
        print(name, q.astype(int).tolist())
