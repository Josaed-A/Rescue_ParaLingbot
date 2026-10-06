"""Compare a reconstructed camera trajectory against a hand-drawn route sketch.

The sketches in captures/pruebas_reales/<sitio>/rutas_reales/ are freehand: a
coloured line on white, no scale and no north. So this does NOT measure metric
accuracy. What it does measure is the SHAPE of the walked route: the sketch
polyline and the top-down projection of the estimated camera path are both
resampled by arc length and aligned with a similarity transform (rotation,
uniform scale, translation, optionally a mirror, optionally reversed), and the
residual is reported as a percentage of the route length.

Top-down projection uses the cameras' own "down" axis (in OpenCV camera
convention +Y points down in the image, so R_c2w[:,1] is world-down for that
frame), averaged over the run — not PCA, which would happily flatten a drifting
path onto the wrong plane.

Metrics (all shape-only):
  error_pct        RMS distance between aligned paths / sketch length
  error_p90_pct    90th percentile of the same distance
  straightness     end-to-end distance / path length, for each path
  turn_total_deg   accumulated absolute heading change, for each path
  length_ratio     estimated length / sketch length after alignment (1.0 = same
                   proportions; the scale itself is fitted, so this is a check)

Usage:
  python3 scripts_context/compare_route.py --npz run.npz --sketch ruta.jpeg \
      --out_json route.json --out_png route.png [--label v5]
"""
import argparse
import json
import os

import cv2
import numpy as np


# ---------------------------------------------------------------- sketch side

def yellow_mask(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    m = cv2.inRange(hsv, np.array([15, 80, 120]), np.array([45, 255, 255]))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    if n > 1:  # keep the drawn route, drop specks
        big = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        m = np.uint8(lab == big) * 255
    return m


def thin(mask):
    """Zhang-Suen thinning to a 1-px skeleton."""
    img = (mask > 0).astype(np.uint8)
    changed = True
    while changed:
        changed = False
        for step in (0, 1):
            p = [np.roll(np.roll(img, dy, 0), dx, 1) for dy, dx in
                 [(-1, 0), (-1, -1), (0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1)]]
            # p[0]=N p[1]=NW p[2]=W p[3]=SW p[4]=S p[5]=SE p[6]=E p[7]=NE  (rolled copies)
            P2, P3, P4, P5, P6, P7, P8, P9 = p[0], p[7], p[6], p[5], p[4], p[3], p[2], p[1]
            B = P2 + P3 + P4 + P5 + P6 + P7 + P8 + P9
            seq = [P2, P3, P4, P5, P6, P7, P8, P9, P2]
            A = sum(((seq[i] == 0) & (seq[i + 1] == 1)).astype(np.uint8) for i in range(8))
            if step == 0:
                cond = (P2 * P4 * P6 == 0) & (P4 * P6 * P8 == 0)
            else:
                cond = (P2 * P4 * P8 == 0) & (P2 * P6 * P8 == 0)
            rm = (img == 1) & (B >= 2) & (B <= 6) & (A == 1) & cond
            if rm.any():
                img[rm] = 0
                changed = True
    return img


def longest_path(skel):
    """Longest path of the skeleton graph (ignores short spurs: arrowheads, branches)."""
    pts = np.argwhere(skel > 0)
    idx = {(int(y), int(x)): i for i, (y, x) in enumerate(pts)}
    nb = [[] for _ in pts]
    for (y, x), i in idx.items():
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy or dx:
                    j = idx.get((y + dy, x + dx))
                    if j is not None:
                        nb[i].append(j)

    def bfs(src):
        dist = {src: 0}
        prev = {src: None}
        q = [src]
        while q:
            cur = q.pop(0)
            for j in nb[cur]:
                if j not in dist:
                    dist[j] = dist[cur] + 1
                    prev[j] = cur
                    q.append(j)
        far = max(dist, key=dist.get)
        return far, prev

    a, _ = bfs(0)
    b, prev = bfs(a)
    path = []
    cur = b
    while cur is not None:
        path.append(pts[cur][::-1])  # (x, y)
        cur = prev[cur]
    return np.array(path, dtype=float)


def sketch_polyline(path, n=300, smooth=9):
    img = cv2.imread(path)
    if img is None:
        raise SystemExit(f"no se pudo leer {path}")
    poly = longest_path(thin(yellow_mask(img)))
    poly[:, 1] = img.shape[0] - poly[:, 1]          # image y grows down; flip to math axes
    if smooth > 2:
        k = np.ones(smooth) / smooth
        poly = np.stack([np.convolve(poly[:, i], k, mode="valid") for i in (0, 1)], 1)
    return resample(poly, n), img.shape


# ------------------------------------------------------------ trajectory side

def trajectory_topdown(npz):
    d = np.load(npz)
    real = d["is_real"]
    E = np.tile(np.eye(4), (len(real), 1, 1))
    E[:, :3, :4] = d["extrinsic"].astype(np.float64)
    c2w = np.linalg.inv(E)[real]                    # the viewer inverts it too
    centers = c2w[:, :3, 3]
    down = c2w[:, :3, 1].mean(0)                    # camera +Y is down
    down /= np.linalg.norm(down)
    e1 = np.array([1.0, 0, 0]) - down * down[0]
    if np.linalg.norm(e1) < 1e-6:
        e1 = np.array([0, 0, 1.0]) - down * down[2]
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(down, e1)
    flat = np.stack([centers @ e1, centers @ e2], 1)
    off_plane = float(np.std(centers @ down))
    return flat, off_plane, float(np.abs(centers @ down).ptp())


# ----------------------------------------------------------------- comparison

def resample(poly, n):
    seg = np.linalg.norm(np.diff(poly, axis=0), axis=1)
    s = np.concatenate([[0], np.cumsum(seg)])
    if s[-1] <= 0:
        raise SystemExit("trayectoria degenerada (longitud cero)")
    t = np.linspace(0, s[-1], n)
    return np.stack([np.interp(t, s, poly[:, i]) for i in (0, 1)], 1)


def umeyama(src, dst):
    """Similarity transform (scale, rotation, translation) mapping src onto dst."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    S = (dst - mu_d).T @ (src - mu_s) / len(src)
    U, D, Vt = np.linalg.svd(S)
    R = U @ np.diag([1, np.sign(np.linalg.det(U @ Vt))]) @ Vt
    var = ((src - mu_s) ** 2).sum() / len(src)
    c = float(np.trace(np.diag(D) @ np.diag([1, np.sign(np.linalg.det(U @ Vt))])) / var)
    return c, R, mu_d - c * R @ mu_s


def path_len(p):
    return float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum())


def turn_total(p):
    v = np.diff(p, axis=0)
    ang = np.unwrap(np.arctan2(v[:, 1], v[:, 0]))
    return float(np.degrees(np.abs(np.diff(ang)).sum()))


def straightness(p):
    return float(np.linalg.norm(p[-1] - p[0]) / max(path_len(p), 1e-9))


def best_alignment(est, ref):
    """Try mirror and reversal: the sketch has no north and may be drawn either way."""
    best = None
    for mirror in (False, True):
        e0 = est * np.array([-1, 1]) if mirror else est
        for rev in (False, True):
            e = e0[::-1] if rev else e0
            c, R, t = umeyama(e, ref)
            aligned = (c * (R @ e.T).T) + t
            err = np.linalg.norm(aligned - ref, axis=1)
            score = float(np.sqrt((err ** 2).mean()))
            if best is None or score < best["rmse"]:
                best = {"rmse": score, "err": err, "aligned": aligned,
                        "mirror": mirror, "reversed": rev, "scale": c}
    return best


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--npz", required=True)
    p.add_argument("--sketch", required=True)
    p.add_argument("--out_json", required=True)
    p.add_argument("--out_png", default=None)
    p.add_argument("--label", default="estimada")
    p.add_argument("--n", type=int, default=300)
    args = p.parse_args()

    ref, shape = sketch_polyline(args.sketch, args.n)
    est_raw, off_plane, off_span = trajectory_topdown(args.npz)
    est = resample(est_raw, args.n)
    b = best_alignment(est, ref)

    L = path_len(ref)
    res = {
        "npz": os.path.basename(args.npz),
        "sketch": os.path.basename(args.sketch),
        "error_pct": round(100 * b["rmse"] / L, 2),
        "error_p90_pct": round(100 * float(np.percentile(b["err"], 90)) / L, 2),
        "error_max_pct": round(100 * float(b["err"].max()) / L, 2),
        "straightness_est": round(straightness(est), 3),
        "straightness_sketch": round(straightness(ref), 3),
        "turn_total_deg_est": round(turn_total(resample(est, 60)), 1),
        "turn_total_deg_sketch": round(turn_total(resample(ref, 60)), 1),
        "length_ratio": round(path_len(b["aligned"]) / L, 3),
        "mirror": b["mirror"], "reversed": b["reversed"],
        "vertical_spread_rel": round(off_plane / max(path_len(est_raw), 1e-9), 4),
    }
    with open(args.out_json, "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps(res, indent=1))

    if args.out_png:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(13, 6))
        ax[0].plot(ref[:, 0], ref[:, 1], color="#d4a017", lw=4, label="ruta real (croquis)")
        a = b["aligned"]
        sc = ax[0].scatter(a[:, 0], a[:, 1], c=100 * b["err"] / L, cmap="viridis", s=12,
                           label=f"{args.label} (alineada)")
        ax[0].plot(a[0, 0], a[0, 1], "go", ms=10); ax[0].plot(a[-1, 0], a[-1, 1], "ro", ms=10)
        ax[0].set_aspect("equal"); ax[0].legend(); ax[0].set_axis_off()
        ax[0].set_title(f"error de forma {res['error_pct']}% del largo (p90 {res['error_p90_pct']}%)")
        plt.colorbar(sc, ax=ax[0], label="error local (% del largo)")
        ax[1].plot(100 * b["err"] / L)
        ax[1].set_xlabel("avance por la ruta (0-100%)"); ax[1].set_ylabel("error (% del largo)")
        ax[1].set_title("dónde se desvía")
        ax[1].grid(alpha=.3)
        fig.tight_layout(); fig.savefig(args.out_png, dpi=110); plt.close(fig)
        print("figura:", args.out_png)


if __name__ == "__main__":
    main()
