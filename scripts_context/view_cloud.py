"""Serve a merged dense point cloud (.ply from export_dense_cloud.py) in the browser.

view_npz.py shows the map the way the model produces it: one point cloud per
frame, with the overlap between frames left in. This one shows the merged cloud
(one point per occupied voxel), which is what you want to look at when the
question is "how much real detail is there", not "what did each frame see".

Usage:
  python3 scripts_context/view_cloud.py dense.ply [--port 8080] [--point_size 0.004]
"""
import argparse
import time

import numpy as np
import open3d as o3d
import viser


def main():
    p = argparse.ArgumentParser()
    p.add_argument("ply")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--point_size", type=float, default=None)
    p.add_argument("--max_points", type=int, default=12_000_000,
                   help="submuestreo aleatorio si la nube supera esto (el navegador se ahoga)")
    args = p.parse_args()

    pc = o3d.io.read_point_cloud(args.ply)
    pts = np.asarray(pc.points, dtype=np.float32)
    col = (np.asarray(pc.colors) * 255).astype(np.uint8)
    if len(pts) > args.max_points:
        idx = np.random.choice(len(pts), args.max_points, replace=False)
        pts, col = pts[idx], col[idx]
        print(f"submuestreado a {args.max_points:,} puntos para el navegador", flush=True)
    size = args.point_size or float(np.linalg.norm(pts.max(0) - pts.min(0))) * 0.0004
    print(f"{len(pts):,} puntos, tamaño de punto {size:.5f}", flush=True)

    server = viser.ViserServer(port=args.port)
    server.scene.add_point_cloud("/nube", points=pts, colors=col, point_size=size)
    gui = server.gui.add_slider("tamaño del punto", min=size / 4, max=size * 6, step=size / 20,
                                initial_value=size)

    @gui.on_update
    def _(_):
        server.scene.add_point_cloud("/nube", points=pts, colors=col, point_size=gui.value)

    print(f"Visor en http://localhost:{args.port}", flush=True)
    while True:
        time.sleep(3600)


if __name__ == "__main__":
    main()
