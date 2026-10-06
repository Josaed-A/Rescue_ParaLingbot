"""Convert a --save_predictions .npz into the form demo_render/batch_demo.py renders.

batch_demo.py --load_predictions reads images / depth / depth_conf / extrinsic /
intrinsic stacked along frames (demo_render/rgbd_render/data/loader.py), which is
what process_and_view.py already saves — except that our file also carries the
synthetic context frames and a few extra arrays. This drops the synthetic frames
(they must not appear in the rendered walkthrough) and keeps only those keys, so
the renderer never loads the model: the upstream loader puts the checkpoint on
the GPU and then moves the model there too, which needs ~9 GB and does not fit
on this 8 GB card.

Usage:
  python3 scripts_context/npz_for_render.py run.npz --out render_input.npz
"""
import argparse

import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("npz")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    d = np.load(args.npz)
    real = d["is_real"]
    out = {
        "images": d["images"][real],
        "depth": d["depth"][real].astype(np.float32),
        "depth_conf": d["depth_conf"][real].astype(np.float32),
        "extrinsic": d["extrinsic"][real].astype(np.float32),
        "intrinsic": d["intrinsic"][real].astype(np.float32),
    }
    np.savez(args.out, **out)
    print(f"{int(real.sum())} frames reales de {len(real)} -> {args.out}")
    for k, v in out.items():
        print(f"  {k}: {v.shape} {v.dtype}")


if __name__ == "__main__":
    main()
