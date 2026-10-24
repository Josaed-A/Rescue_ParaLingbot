"""Self-consistency of a LingBot-Map reconstruction, without ground truth.

For pairs of REAL frames (a, b), the depth of a is lifted to 3D with a's
predicted pose and projected into b; the projected depth is compared with the
depth the model predicted for b. A geometrically coherent map (right poses,
right scale, right depth) agrees; drift, pose jumps or depth noise disagree.

Pairs:
  - consecutive real frames of the run
  - lag pairs: for each real frame, the real frame closest to +L video frames
    (same time offset for every run, so runs with different spacing compare
    fairly; L=15 is 0.5 s and L=30 is 1 s at 29.8 fps)

Metrics per pair set (medians over pairs):
  inlier  fraction of co-visible points whose depth agrees within 5 %
  relerr  median relative depth error of co-visible points
  photo   median absolute colour difference (0-255) at the projected points
  overlap fraction of a's confident points that land inside b
Plus trajectory smoothness (camera-centre jerk and rotation-step spikes) and
the model's own median depth confidence.

Usage:
  python3 src/mapas/evaluate_consistency.py name=path.npz [name=path.npz ...] --out report.json
"""
import argparse
import json

import numpy as np

VIDEO_FPS = 1962 / 65.77   # muestra_unisabana.mp4: 1962 frames, 29.83 fps
BASELINE_FPS = 10.0         # prueba_1 was extracted with ffmpeg -vf fps=10


def load(path):
    d = np.load(path)
    out = {k: d[k] for k in d.files}
    ds = int(out["ds"])
    K = out["intrinsic"].astype(np.float64).copy()
    K[:, :2, :] /= ds
    out["K"] = K
    src = out["source_index"].astype(np.int64)
    if (src == np.arange(len(src))).all() and out["is_real"].all():
        # plain run (prueba_1): map extraction index to video frame index
        src = np.round(np.arange(len(src)) * VIDEO_FPS / BASELINE_FPS).astype(np.int64)
    out["src"] = src
    # The viewer (lingbot_map/utils/geometry.py::depth_to_world_coords_points)
    # treats this array as cam-from-world and inverts it: use the same
    # camera-to-world the map is drawn with.
    E = np.tile(np.eye(4), (len(src), 1, 1))
    E[:, :3, :4] = out["extrinsic"].astype(np.float64)
    out["c2w"] = np.linalg.inv(E)[:, :3, :4]
    return out


def pair_metrics(run, a, b, stride=3, rel_tol=0.05):
    D = run["depth"]
    C = run["depth_conf"]
    I = run["images"]
    E = run["c2w"]
    K = run["K"]
    h, w = D.shape[1:]
    vs, us = np.mgrid[0:h:stride, 0:w:stride]
    za = D[a][vs, us].astype(np.float64)
    ca = C[a][vs, us].astype(np.float64)
    sel = (za > 0) & (ca >= np.median(C[a]))
    u, v, z = us[sel], vs[sel], za[sel]
    Ka, Kb = K[a], K[b]
    x = (u - Ka[0, 2]) / Ka[0, 0] * z
    y = (v - Ka[1, 2]) / Ka[1, 1] * z
    pa = np.stack([x, y, z], 1)
    pw = pa @ E[a, :3, :3].T + E[a, :3, 3]
    pb = (pw - E[b, :3, 3]) @ E[b, :3, :3]
    zb = pb[:, 2]
    front = zb > 1e-6
    ub = Kb[0, 0] * pb[:, 0] / np.where(front, zb, 1) + Kb[0, 2]
    vb = Kb[1, 1] * pb[:, 1] / np.where(front, zb, 1) + Kb[1, 2]
    ui, vi = np.round(ub).astype(int), np.round(vb).astype(int)
    inside = front & (ui >= 0) & (ui < w) & (vi >= 0) & (vi < h)
    n_sel = int(sel.sum())
    if inside.sum() < 50:
        return None
    ui, vi, zb_in = ui[inside], vi[inside], zb[inside]
    db = D[b][vi, ui].astype(np.float64)
    good = db > 0
    rel = np.abs(zb_in[good] - db[good]) / db[good]
    ia = I[a][v[inside][good], u[inside][good]].astype(np.float64)
    ib = I[b][vi[good], ui[good]].astype(np.float64)
    return {
        "inlier": float((rel < rel_tol).mean()),
        "relerr": float(np.median(rel)),
        "photo": float(np.median(np.abs(ia - ib).mean(1))),
        "overlap": float(inside.sum() / max(n_sel, 1)),
    }


def summarize(rows):
    rows = [r for r in rows if r is not None]
    if not rows:
        return {"pairs": 0}
    out = {"pairs": len(rows)}
    for k in rows[0]:
        vals = np.array([r[k] for r in rows])
        out[k] = round(float(np.median(vals)), 4)
        if k == "inlier":
            out["inlier_p10"] = round(float(np.percentile(vals, 10)), 4)
    return out


def trajectory(run, idx):
    E = run["c2w"][idx]
    c = E[:, :3, 3]
    steps = np.diff(c, axis=0)
    step_len = np.linalg.norm(steps, axis=1)
    jerk = np.linalg.norm(np.diff(steps, axis=0), axis=1)
    R = E[:, :3, :3]
    cosang = np.clip((np.einsum("nij,nij->n", R[1:], R[:-1]) - 1) / 2, -1, 1)
    ang = np.degrees(np.arccos(cosang))
    med = np.median(step_len) + 1e-12
    return {
        "path_length_rel": round(float(step_len.sum() / med), 1),
        "jerk_rel_median": round(float(np.median(jerk) / med), 3),
        "jerk_rel_p95": round(float(np.percentile(jerk, 95) / med), 3),
        "rot_step_deg_median": round(float(np.median(ang)), 3),
        "rot_step_deg_p99": round(float(np.percentile(ang, 99)), 3),
        "rot_step_deg_max": round(float(ang.max()), 3),
    }


def evaluate(run, lags=(15, 30)):
    real = np.flatnonzero(run["is_real"])
    src = run["src"][real]
    res = {
        "frames_total": int(len(run["is_real"])),
        "frames_real": int(len(real)),
        "depth_conf_median_real": round(float(np.median(run["depth_conf"][real].astype(np.float32))), 3),
        "consecutive": summarize([pair_metrics(run, a, b) for a, b in zip(real[:-1], real[1:])]),
    }
    for L in lags:
        rows = []
        for i, a in enumerate(real):
            j = int(np.argmin(np.abs(src - (src[i] + L))))
            if j <= i or abs(src[j] - src[i] - L) > L * 0.25:
                continue
            rows.append(pair_metrics(run, a, real[j]))
        res[f"lag{L}"] = summarize(rows)
    res["trajectory"] = trajectory(run, real)
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument("runs", nargs="+", help="nombre=ruta.npz")
    p.add_argument("--out", required=True)
    args = p.parse_args()
    report = {}
    for spec in args.runs:
        name, path = spec.split("=", 1)
        report[name] = evaluate(load(path))
        print(name, json.dumps(report[name], indent=1), flush=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=1)

    keys = [("consecutive", "inlier"), ("lag15", "inlier"), ("lag30", "inlier"),
            ("lag30", "relerr"), ("lag30", "photo")]
    print("\n" + "run".ljust(16) + "".join(f"{s}.{k}".rjust(16) for s, k in keys) + "   jerk_p95  rot_p99")
    for name, r in report.items():
        cells = "".join(str(r[s].get(k, "-")).rjust(16) for s, k in keys)
        t = r["trajectory"]
        print(name.ljust(16) + cells + f"{t['jerk_rel_p95']:>10}{t['rot_step_deg_p99']:>9}")


if __name__ == "__main__":
    main()
