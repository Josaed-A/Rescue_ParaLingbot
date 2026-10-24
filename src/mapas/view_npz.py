"""Serve the viser map from a saved --save_predictions .npz, without re-running the model.

Only real frames are shown (synthetic context frames were already excluded from
the map). The .npz is half resolution, so the cloud has ~4x fewer points than a
live run: lighter in the browser, same geometry.

Usage:
  python3 src/mapas/view_npz.py <run.npz> [--port 8080] [--glb_out map.glb]
"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("npz")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--conf_threshold", type=float, default=1.5)
    p.add_argument("--downsample_factor", type=int, default=3)
    p.add_argument("--glb_out", default=None)
    args = p.parse_args()

    d = np.load(args.npz)
    real = d["is_real"]
    ds = int(d["ds"])
    K = d["intrinsic"][real].astype(np.float32).copy()
    K[:, :2, :] /= ds
    pred = {
        "images": d["images"][real].transpose(0, 3, 1, 2).astype(np.float32) / 255.0,
        "depth": d["depth"][real].astype(np.float32)[..., None],
        "depth_conf": d["depth_conf"][real].astype(np.float32),
        "extrinsic": d["extrinsic"][real].astype(np.float32),
        "intrinsic": K,
    }
    print(f"{int(real.sum())} frames reales de {len(real)}", flush=True)

    from lingbot_map.vis import PointCloudViewer
    viewer = PointCloudViewer(pred_dict=pred, port=args.port, vis_threshold=args.conf_threshold,
                              downsample_factor=args.downsample_factor, point_size=0.00001)
    if args.glb_out:
        viewer.glb_output_path.value = args.glb_out
        viewer._export_glb()
        print(viewer.glb_status.value, flush=True)
    print(f"Visor en http://localhost:{args.port}", flush=True)
    viewer.run(background_mode=True)
    while True:
        time.sleep(3600)


if __name__ == "__main__":
    main()
