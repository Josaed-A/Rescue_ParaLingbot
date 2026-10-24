"""Lightweight Context Analyzer: per-frame sharpness and inter-frame motion.

Runs over a pool of candidate frames (e.g. every frame of a video) and writes a
JSON with, per frame, a blur score and, per consecutive pair, how much the image
moved (RAFT optical flow). curate_and_synthesize.py uses it to decide which
frames are worth sending to LingBot-Map, which are redundant or too blurry, and
where the gap between kept frames is too large and needs synthetic in-betweens.

Motion is reported in pixels at the candidate resolution (540 px wide for the
unisabana run), measured on a half-resolution copy and scaled back.

Usage:
  python3 src/mapas/analyze_frames.py --frames_dir <dir> --out <analysis.json>
"""
import argparse
import glob
import json
import os
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.models.optical_flow import Raft_Small_Weights, raft_small


def sharpness(gray):
    """Variance of the Laplacian: drops sharply on motion-blurred frames."""
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def to_raft(batch_rgb, size):
    t = torch.from_numpy(np.stack(batch_rgb)).permute(0, 3, 1, 2).float() / 255.0
    t = F.interpolate(t, size=size, mode="bilinear", align_corners=False)
    return t * 2.0 - 1.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--frames_dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--flow_updates", type=int, default=12)
    args = p.parse_args()

    paths = sorted(glob.glob(os.path.join(args.frames_dir, "*.png")))
    if len(paths) < 2:
        raise SystemExit(f"no hay frames en {args.frames_dir}")
    first = cv2.imread(paths[0])
    H, W = first.shape[:2]
    # RAFT needs sides divisible by 8; half resolution is enough to rank motion
    fh, fw = (H // 2) // 8 * 8, (W // 2) // 8 * 8
    scale_back = W / fw

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = raft_small(weights=Raft_Small_Weights.DEFAULT).to(device).eval()

    t0 = time.time()
    sharp = []
    rgb = []
    for pth in paths:
        img = cv2.imread(pth)
        sharp.append(sharpness(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)))
        rgb.append(cv2.cvtColor(cv2.resize(img, (fw, fh), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB))
    print(f"{len(paths)} frames leídos, nitidez calculada en {time.time() - t0:.1f}s", flush=True)

    motion_med, motion_p90, motion_mean_vec = [], [], []
    t0 = time.time()
    with torch.no_grad():
        for s in range(0, len(paths) - 1, args.batch):
            e = min(s + args.batch, len(paths) - 1)
            a = to_raft(rgb[s:e], (fh, fw)).to(device)
            b = to_raft(rgb[s + 1:e + 1], (fh, fw)).to(device)
            flow = model(a, b, num_flow_updates=args.flow_updates)[-1] * scale_back
            mag = flow.norm(dim=1).flatten(1)
            motion_med += mag.median(dim=1).values.tolist()
            motion_p90 += mag.quantile(0.9, dim=1).tolist()
            motion_mean_vec += flow.mean(dim=(2, 3)).tolist()
    print(f"flujo de {len(paths) - 1} pares en {time.time() - t0:.1f}s", flush=True)

    out = {
        "frames_dir": os.path.abspath(args.frames_dir),
        "resolution": [W, H],
        "files": [os.path.basename(x) for x in paths],
        "sharpness": sharp,
        "pairs": {"motion_med": motion_med, "motion_p90": motion_p90, "mean_flow": motion_mean_vec},
    }
    with open(args.out, "w") as f:
        json.dump(out, f)
    s = np.array(sharp)
    m = np.array(motion_med)
    print(f"nitidez: mediana {np.median(s):.1f}, p10 {np.percentile(s, 10):.1f}, p90 {np.percentile(s, 90):.1f}")
    print(f"movimiento (px/par): mediana {np.median(m):.2f}, p90 {np.percentile(m, 90):.2f}, max {m.max():.2f}, total {m.sum():.0f}")
    print(f"escrito {args.out}")


if __name__ == "__main__":
    main()
