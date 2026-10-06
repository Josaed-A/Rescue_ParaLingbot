"""Dense point cloud from a run's predictions, with overlapping views merged.

The model is not sampling a sparse grid of interest points: its depth head
outputs a depth value PER PIXEL (518x518 = 268k points per frame, upsampled by
the DPT head from an internal 37x37 patch grid). What thins the cloud is the
visualization: PointCloudViewer's `downsample_factor` (10 by default) keeps one
point in ten, and --save_predictions stores half resolution on top of that.

This script goes the other way: it takes every valid pixel of every real frame,
unprojects it with the predicted pose, and then merges the result on a voxel
grid. Merging matters because consecutive frames see the same surface many
times: without it the cloud is mostly duplicated points (heavy to load, no extra
detail); with it, the detail is kept and the redundancy is not.

Usage:
  python3 src/mapas/export_dense_cloud.py run.npz --out_ply dense.ply \
      [--voxel_rel 0.0015] [--conf_percentile 40] [--frame_stride 1] [--out_glb dense.glb]
"""
import argparse
import json
import os

import numpy as np


def unproject(d, ds_idx=1):
    """Yield (points, colors, conf) in world coordinates for each real frame."""
    depth = d["depth"]
    conf = d["depth_conf"]
    img = d["images"]
    real = np.flatnonzero(d["is_real"])
    ds = int(d["ds"])
    K = d["intrinsic"].astype(np.float64).copy()
    K[:, :2, :] /= ds
    E = np.tile(np.eye(4), (len(d["is_real"]), 1, 1))
    E[:, :3, :4] = d["extrinsic"].astype(np.float64)
    c2w = np.linalg.inv(E)                       # same convention the viewer draws
    h, w = depth.shape[1:]
    v, u = np.mgrid[0:h, 0:w]
    for i in real[::ds_idx]:
        z = depth[i].astype(np.float64)
        c = conf[i].astype(np.float32)
        m = np.isfinite(z) & (z > 0)
        yield (i, u[m], v[m], z[m], c[m], img[i][m], K[i], c2w[i])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("npz")
    p.add_argument("--out_ply", required=True)
    p.add_argument("--out_glb", default=None)
    p.add_argument("--voxel_rel", type=float, default=0.0015,
                   help="tamaño de vóxel como fracción de la diagonal de la escena")
    p.add_argument("--conf_percentile", type=float, default=40,
                   help="descarta el X%% de píxeles menos confiables (0 = usar todos)")
    p.add_argument("--frame_stride", type=int, default=1, help="usar 1 de cada N frames reales")
    p.add_argument("--chunk_frames", type=int, default=30,
                   help="frames acumulados antes de fusionar en vóxeles (controla la memoria)")
    p.add_argument("--max_points_per_frame", type=int, default=0,
                   help="0 = todos los píxeles válidos del frame")
    p.add_argument("--float32", action="store_true",
                   help="guardar las coordenadas en float32 en vez del double de open3d: "
                        "mismo número de puntos, archivo a la mitad y carga más rápida en el visor")
    args = p.parse_args()

    d = np.load(args.npz)
    confs = d["depth_conf"][d["is_real"]].astype(np.float32)
    thr = float(np.percentile(confs, args.conf_percentile)) if args.conf_percentile > 0 else -np.inf
    print(f"umbral de confianza: {thr:.3f} (percentil {args.conf_percentile})", flush=True)

    def frame_points(i, u, v, z, c, rgb, K, c2w):
        keep = c >= thr
        u, v, z, rgb = u[keep], v[keep], z[keep], rgb[keep]
        n = len(z)
        if args.max_points_per_frame and n > args.max_points_per_frame:
            sel = np.random.default_rng(i).choice(n, args.max_points_per_frame, replace=False)
            u, v, z, rgb = u[sel], v[sel], z[sel], rgb[sel]
        if len(z) == 0:      # every pixel of this frame fell below the confidence threshold
            return n, None, None
        x = (u - K[0, 2]) / K[0, 0] * z
        y = (v - K[1, 2]) / K[1, 1] * z
        w = np.stack([x, y, z], 1) @ c2w[:3, :3].T + c2w[:3, 3]
        return n, w.astype(np.float32), rgb

    # Pass 1: the scene extent fixes the voxel size before anything is accumulated
    # (same diagonal as before: min/max over every kept point).
    lo = np.full(3, np.inf)
    hi = np.full(3, -np.inf)
    raw = 0
    for args_ in unproject(d, args.frame_stride):
        n, w, _ = frame_points(*args_)
        raw += n
        if w is not None:
            lo = np.minimum(lo, w.min(0))
            hi = np.maximum(hi, w.max(0))
    diag = float(np.linalg.norm(hi - lo))
    voxel = args.voxel_rel * diag
    print(f"puntos crudos: {raw:,} de {int(d['is_real'].sum())} frames reales", flush=True)

    def merge(pts, cols):
        """Promedio de posición y color por vóxel (lo mismo que voxel_down_sample de open3d),
        en numpy y con datos compactos: float32 + uint8, en vez de float64 en open3d."""
        ijk = np.floor((pts - lo) / voxel).astype(np.int64)
        key = (ijk[:, 0] << 42) | (ijk[:, 1] << 21) | ijk[:, 2]
        _, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
        del key, ijk
        out_p = np.empty((len(cnt), 3), np.float32)
        out_c = np.empty((len(cnt), 3), np.uint8)
        for k in range(3):
            out_p[:, k] = np.bincount(inv, weights=pts[:, k], minlength=len(cnt)) / cnt
            out_c[:, k] = np.round(np.bincount(inv, weights=cols[:, k], minlength=len(cnt)) / cnt)
        return out_p, out_c

    # Pass 2: accumulate in chunks, each merged onto the (fixed) voxel grid
    chunks_p, chunks_c, buf_p, buf_c = [], [], [], []

    def flush():
        if buf_p:
            mp, mc = merge(np.concatenate(buf_p), np.concatenate(buf_c))
            chunks_p.append(mp)
            chunks_c.append(mc)
            buf_p.clear()
            buf_c.clear()

    for args_ in unproject(d, args.frame_stride):
        _, w, rgb = frame_points(*args_)
        if w is None:
            continue
        buf_p.append(w)
        buf_c.append(rgb)
        if len(buf_p) >= args.chunk_frames:
            flush()
    flush()
    pts, cols = merge(np.concatenate(chunks_p), np.concatenate(chunks_c))
    del chunks_p, chunks_c
    n_vox = len(pts)
    print(f"tras fusionar en vóxeles de {voxel:.4f} (={args.voxel_rel} de la diagonal): {n_vox:,} puntos", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.out_ply)) or ".", exist_ok=True)
    if args.float32:
        dt = np.dtype([("xyz", "<f4", 3), ("rgb", "u1", 3)])
        buf = np.empty(len(pts), dtype=dt)
        buf["xyz"], buf["rgb"] = pts, cols
        with open(args.out_ply, "wb") as f:
            f.write((f"ply\nformat binary_little_endian 1.0\nelement vertex {len(pts)}\n"
                     "property float x\nproperty float y\nproperty float z\n"
                     "property uchar red\nproperty uchar green\nproperty uchar blue\n"
                     "end_header\n").encode("ascii"))
            f.write(buf.tobytes())
    else:
        import open3d as o3d
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
        pc.colors = o3d.utility.Vector3dVector(cols.astype(np.float64) / 255.0)
        o3d.io.write_point_cloud(args.out_ply, pc)
    print("PLY:", args.out_ply, f"{os.path.getsize(args.out_ply)/2**20:.0f} MB", flush=True)

    if args.out_glb:
        import trimesh
        cloud = trimesh.PointCloud(pts, cols)
        trimesh.Scene([cloud]).export(args.out_glb)
        print("GLB:", args.out_glb, f"{os.path.getsize(args.out_glb)/2**20:.0f} MB", flush=True)

    with open(os.path.splitext(args.out_ply)[0] + "_info.json", "w") as f:
        json.dump({"npz": os.path.basename(args.npz), "frames_reales": int(d["is_real"].sum()),
                   "puntos_crudos": int(raw), "puntos_finales": n_vox,
                   "voxel_rel": args.voxel_rel, "conf_percentile": args.conf_percentile,
                   "frame_stride": args.frame_stride, "resolucion_guardada": int(d["ds"])}, f, indent=1)


if __name__ == "__main__":
    main()
