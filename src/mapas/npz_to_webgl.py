"""Convert a --save_predictions .npz into static assets for the WebGL viewer
(scripts_context/webgl_viewer/): a point cloud (PLY, binary little-endian) and
a camera-path JSON (position + look direction per real frame), so the browser
can draw the trajectory and frustums without needing viser or re-running the
model.

Two cloud modes:
  --mode raw     every real frame's valid pixels, concatenated (this is what
                 view_npz.py/viser shows: per-frame overlap included, so it is
                 the honest "what the model produced" view -- heavier).
  --mode dense   frames unprojected the same way, then merged on a voxel grid
                 (same algorithm as export_dense_cloud.py, reimplemented here
                 so this script has no cross-dependency) -- lighter, no
                 duplicated surface points.

Camera path convention matches the rest of the repo: the viewer inverts the
stored extrinsic (treats it as camera-from-world), so c2w = inv(extrinsic).

Usage:
  python3 scripts_context/npz_to_webgl.py run.npz --out_dir captures/.../webgl \
      --name final_m1 --mode dense --voxel_rel 0.0004 --conf_percentile 25
"""
import argparse
import json
import os
import struct

import numpy as np


def unproject_frames(d, conf_percentile, frame_stride, max_points_per_frame, rng):
    depth = d["depth"]
    conf = d["depth_conf"]
    img = d["images"]
    real = np.flatnonzero(d["is_real"])
    ds = int(d["ds"])
    K = d["intrinsic"].astype(np.float64).copy()
    K[:, :2, :] /= ds
    E = np.tile(np.eye(4), (len(d["is_real"]), 1, 1))
    E[:, :3, :4] = d["extrinsic"].astype(np.float64)
    c2w = np.linalg.inv(E)  # same convention the viewer / renderer use
    h, w = depth.shape[1:]
    v, u = np.mgrid[0:h, 0:w]

    confs = conf[real].astype(np.float32)
    thr = float(np.percentile(confs, conf_percentile)) if conf_percentile > 0 else -np.inf

    cams = []
    for i in real[::frame_stride]:
        cams.append({"index": int(i), "c2w": c2w[i].tolist(), "intrinsic": K[i].tolist()})

    def gen():
        for i in real[::frame_stride]:
            z = depth[i].astype(np.float64)
            c = conf[i].astype(np.float32)
            m = np.isfinite(z) & (z > 0) & (c >= thr)
            uu, vv, zz, cc = u[m], v[m], z[m], c[m]
            rgb = img[i][m]
            if max_points_per_frame and len(zz) > max_points_per_frame:
                sel = rng.choice(len(zz), max_points_per_frame, replace=False)
                uu, vv, zz, rgb = uu[sel], vv[sel], zz[sel], rgb[sel]
            if len(zz) == 0:
                continue
            x = (uu - K[i, 0, 2]) / K[i, 0, 0] * zz
            y = (vv - K[i, 1, 2]) / K[i, 1, 1] * zz
            cam_xyz = np.stack([x, y, zz], 1)
            world = cam_xyz @ c2w[i][:3, :3].T + c2w[i][:3, 3]
            yield world.astype(np.float32), rgb.astype(np.uint8)

    return gen, cams, thr


def write_ply_binary(path, points, colors):
    n = len(points)
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n"
    ).encode("ascii")
    dtype = np.dtype([("xyz", "<f4", 3), ("rgb", "u1", 3)])
    buf = np.empty(n, dtype=dtype)
    buf["xyz"] = points
    buf["rgb"] = colors
    with open(path, "wb") as f:
        f.write(header)
        f.write(buf.tobytes())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("npz")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--name", required=True, help="prefijo de los archivos de salida")
    p.add_argument("--mode", choices=["raw", "dense"], default="dense")
    p.add_argument("--conf_percentile", type=float, default=25.0)
    p.add_argument("--frame_stride", type=int, default=1)
    p.add_argument("--max_points_per_frame", type=int, default=0,
                   help="raw: 0 = todos los píxeles válidos (pesado); dense lo ignora salvo antes de fusionar")
    p.add_argument("--voxel_rel", type=float, default=0.0004,
                   help="solo --mode dense: tamaño de vóxel relativo a la diagonal de la escena")
    p.add_argument("--chunk_frames", type=int, default=60)
    args = p.parse_args()

    d = np.load(args.npz)
    rng = np.random.default_rng(0)
    gen, cams, thr = unproject_frames(d, args.conf_percentile, args.frame_stride,
                                       args.max_points_per_frame, rng)
    print(f"umbral de confianza: {thr:.3f} (percentil {args.conf_percentile})", flush=True)

    os.makedirs(args.out_dir, exist_ok=True)
    ply_path = os.path.join(args.out_dir, f"{args.name}_{args.mode}.ply")
    cams_path = os.path.join(args.out_dir, f"{args.name}_cameras.json")

    raw_count = 0
    if args.mode == "raw":
        pts_all, col_all = [], []
        for pts, col in gen():
            raw_count += len(pts)
            pts_all.append(pts)
            col_all.append(col)
        points = np.concatenate(pts_all) if pts_all else np.zeros((0, 3), np.float32)
        colors = np.concatenate(col_all) if col_all else np.zeros((0, 3), np.uint8)
        write_ply_binary(ply_path, points, colors)
        n_final = len(points)
    else:
        import open3d as o3d

        def to_o3d(pts, col):
            pc = o3d.geometry.PointCloud()
            pc.points = o3d.utility.Vector3dVector(np.asarray(pts, dtype=np.float64))
            pc.colors = o3d.utility.Vector3dVector(np.asarray(col, dtype=np.float64) / 255.0)
            return pc

        chunks, pts_all, col_all, extent = [], [], [], None

        def flush(voxel):
            if not pts_all:
                return
            pc = to_o3d(np.concatenate(pts_all), np.concatenate(col_all))
            chunks.append(pc.voxel_down_sample(voxel) if voxel else pc)
            pts_all.clear()
            col_all.clear()

        for pts, col in gen():
            raw_count += len(pts)
            pts_all.append(pts)
            col_all.append(col)
            lo, hi = pts.min(0), pts.max(0)
            extent = (np.minimum(extent[0], lo), np.maximum(extent[1], hi)) if extent else (lo, hi)
            if len(pts_all) >= args.chunk_frames:
                diag = float(np.linalg.norm(extent[1] - extent[0]))
                flush(args.voxel_rel * diag)
        diag = float(np.linalg.norm(extent[1] - extent[0])) if extent is not None else 1.0
        voxel = args.voxel_rel * diag
        flush(voxel)
        pc = chunks[0]
        for c_ in chunks[1:]:
            pc += c_
        pc = pc.voxel_down_sample(voxel)
        write_ply_binary(ply_path, np.asarray(pc.points, dtype=np.float32),
                          (np.asarray(pc.colors) * 255).astype(np.uint8))
        n_final = len(pc.points)

    with open(cams_path, "w") as f:
        json.dump({"npz": os.path.basename(args.npz), "n_cameras": len(cams), "cameras": cams}, f)

    with open(os.path.join(args.out_dir, f"{args.name}_{args.mode}_info.json"), "w") as f:
        json.dump({"npz": os.path.basename(args.npz), "mode": args.mode,
                   "frames_reales": int(d["is_real"].sum()), "puntos_crudos": int(raw_count),
                   "puntos_finales": int(n_final), "conf_percentile": args.conf_percentile,
                   "voxel_rel": args.voxel_rel if args.mode == "dense" else None,
                   "frame_stride": args.frame_stride}, f, indent=1)

    print(f"puntos crudos: {raw_count:,} -> {n_final:,} en {os.path.basename(ply_path)} "
          f"({os.path.getsize(ply_path)/2**20:.0f} MB)", flush=True)
    print(f"cámaras: {len(cams)} -> {os.path.basename(cams_path)}", flush=True)


if __name__ == "__main__":
    main()
