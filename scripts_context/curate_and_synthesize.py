"""Adaptive frame budget + visual buffer (context-to-image) for low-quality video.

Given the candidate pool (every frame of the video) and analyze_frames.py's JSON:

1. Spacing by motion, not by time: a frame is kept roughly every --step_px pixels
   of accumulated optical flow, so slow stretches are not over-sampled
   (redundant frames skipped) and fast ones are not under-sampled.
2. Best-of-window: inside each spacing window the sharpest frame is kept, so
   motion-blurred frames from the walking gait are skipped.
3. Visual buffer: when two consecutive kept frames are still far apart (fast
   turns, or a stretch where every frame is blurry), synthetic in-between
   frames are generated from the two sharp neighbours with bidirectional RAFT
   flow. They give LingBot-Map a smooth temporal context for pose; the mapping
   step (process_and_view.py --manifest) keeps them out of the point cloud so no
   invented geometry reaches the map.

Writes <out_dir>/frames/NNNNNN.png (real frames are hard links to candidates),
<out_dir>/manifest.json and <out_dir>/synthetic_examples.png.

Usage:
  python3 scripts_context/curate_and_synthesize.py --analysis <analysis.json> --out_dir <prueba_dir>
"""
import argparse
import json
import math
import os
import shutil

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import percentile_filter
from torchvision.models.optical_flow import Raft_Large_Weights, raft_large


# ----------------------------------------------------------------------------
# Selection
# ----------------------------------------------------------------------------

def select_frames(sharp, motion, step, blur_rel, window, max_stretch=2.0):
    """Return kept candidate indices and, per kept frame, the gap (px) to the previous one."""
    sharp = np.asarray(sharp)
    ref = percentile_filter(sharp, 75, size=window, mode="nearest")
    rel = sharp / np.maximum(ref, 1e-6)
    cum = np.concatenate([[0.0], np.cumsum(motion)])
    n = len(sharp)
    ok = rel >= blur_rel

    start = int(np.argmax(ok[: max(1, window)]))  # first sharp frame near the beginning
    kept = [start]
    while True:
        last = kept[-1]
        base = cum[last]
        lo = np.searchsorted(cum, base + 0.6 * step, side="left")
        hi = np.searchsorted(cum, base + 1.4 * step, side="right")
        lo = max(lo, last + 1)
        if lo >= n:
            break
        target = base + step
        score = lambda k: rel[k] - 0.5 * abs(cum[k] - target) / step
        cand = [j for j in range(lo, min(hi, n)) if ok[j]]
        if not cand:
            # nothing sharp in the window: look a bit further, but never more
            # than max_stretch * step away (skipping a whole blurry stretch makes
            # gaps too large to bridge); failing that, keep the least blurry
            far = np.searchsorted(cum, base + max_stretch * step, side="right")
            wide = list(range(lo, max(min(far, n), lo + 1)))
            cand = [j for j in wide if ok[j]] or [max(wide, key=lambda k: rel[k])]
        j = max(cand, key=score)
        kept.append(j)
    gaps = [0.0] + [float(cum[b] - cum[a]) for a, b in zip(kept[:-1], kept[1:])]
    return kept, gaps, rel


# ----------------------------------------------------------------------------
# Synthesis (bidirectional flow interpolation)
# ----------------------------------------------------------------------------

def backwarp(img, flow):
    """Sample img at (x + flow). img [1,C,H,W], flow [1,2,H,W]. Returns image and in-bounds mask."""
    _, _, H, W = img.shape
    gy, gx = torch.meshgrid(torch.arange(H, device=img.device), torch.arange(W, device=img.device), indexing="ij")
    x = gx + flow[:, 0]
    y = gy + flow[:, 1]
    grid = torch.stack([2 * x / (W - 1) - 1, 2 * y / (H - 1) - 1], dim=-1)
    out = F.grid_sample(img, grid, mode="bilinear", padding_mode="border", align_corners=True)
    valid = ((x >= 0) & (x <= W - 1) & (y >= 0) & (y <= H - 1)).float()[:, None]
    return out, valid


class Interpolator:
    def __init__(self, device, flow_updates=20):
        self.device = device
        self.flow_updates = flow_updates
        self.model = raft_large(weights=Raft_Large_Weights.DEFAULT).to(device).eval()

    def _prep(self, bgr):
        t = torch.from_numpy(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float()[None] / 255.0
        return t.to(self.device)

    @torch.no_grad()
    def __call__(self, bgr0, bgr1, ts):
        i0, i1 = self._prep(bgr0), self._prep(bgr1)
        H, W = i0.shape[-2:]
        ph, pw = (-H) % 8, (-W) % 8
        pad = lambda t: F.pad(t, (0, pw, 0, ph), mode="replicate")
        a, b = pad(i0) * 2 - 1, pad(i1) * 2 - 1
        f01 = self.model(a, b, num_flow_updates=self.flow_updates)[-1][..., :H, :W]
        f10 = self.model(b, a, num_flow_updates=self.flow_updates)[-1][..., :H, :W]
        # forward-backward inconsistency marks pixels occluded in the other frame
        e0 = (f01 + backwarp(f10, f01)[0]).norm(dim=1, keepdim=True)
        e1 = (f10 + backwarp(f01, f10)[0]).norm(dim=1, keepdim=True)
        outs = []
        for t in ts:
            ft0 = -(1 - t) * t * f01 + t * t * f10
            ft1 = (1 - t) ** 2 * f01 - t * (1 - t) * f10
            w0i, v0 = backwarp(i0, ft0)
            w1i, v1 = backwarp(i1, ft1)
            o0 = backwarp(e0, ft0)[0]
            o1 = backwarp(e1, ft1)[0]
            w0 = (1 - t) * v0 * torch.exp(-o0 / 2.0)
            w1 = t * v1 * torch.exp(-o1 / 2.0)
            den = w0 + w1
            blend = (w0 * w0i + w1 * w1i) / den.clamp_min(1e-4)
            fallback = (1 - t) * w0i + t * w1i
            out = torch.where(den > 1e-3, blend, fallback)
            img = (out[0].permute(1, 2, 0).clamp(0, 1) * 255).byte().cpu().numpy()
            outs.append(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        return outs, float(f01.norm(dim=1).median())


# ----------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--analysis", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--step_px", type=float, default=36.0,
                   help="movimiento acumulado (px) entre frames conservados")
    p.add_argument("--blur_rel", type=float, default=0.5,
                   help="nitidez mínima relativa al p75 local para considerar un frame nítido")
    p.add_argument("--window", type=int, default=45, help="ventana (frames) del p75 local de nitidez")
    p.add_argument("--synth_factor", type=float, default=1.5,
                   help="sintetizar cuando el salto entre conservados supera step_px * este factor")
    p.add_argument("--max_synth", type=int, default=6, help="máximo de frames sintéticos por salto")
    p.add_argument("--max_synth_gap", type=float, default=240.0,
                   help="saltos mayores (px) no se sintetizan: la interpolación se desarma")
    p.add_argument("--no_synth", action="store_true", help="solo curar (variante de control)")
    args = p.parse_args()

    with open(args.analysis) as f:
        an = json.load(f)
    src_dir = an["frames_dir"]
    files = an["files"]
    kept, gaps, rel = select_frames(an["sharpness"], an["pairs"]["motion_med"],
                                    args.step_px, args.blur_rel, args.window)

    frames_dir = os.path.join(args.out_dir, "frames")
    if os.path.isdir(frames_dir):
        shutil.rmtree(frames_dir)
    os.makedirs(frames_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    interp = None if args.no_synth else Interpolator(device)

    manifest = []
    examples = []

    def emit(img_path=None, img=None, **meta):
        name = f"{len(manifest):06d}.png"
        dst = os.path.join(frames_dir, name)
        if img_path is not None:
            os.link(img_path, dst)
        else:
            cv2.imwrite(dst, img)
        manifest.append({"file": name, **meta})

    n_synth = 0
    for k, (ci, gap) in enumerate(zip(kept, gaps)):
        if interp is not None and k > 0 and args.synth_factor * args.step_px < gap <= args.max_synth_gap:
            n = min(args.max_synth, math.ceil(gap / args.step_px) - 1)
            prev = kept[k - 1]
            a = cv2.imread(os.path.join(src_dir, files[prev]))
            b = cv2.imread(os.path.join(src_dir, files[ci]))
            ts = [(i + 1) / (n + 1) for i in range(n)]
            imgs, _ = interp(a, b, ts)
            for t, im in zip(ts, imgs):
                emit(img=im, kind="synthetic", between=[prev, ci], t=round(t, 3), gap_px=round(gap, 1))
            n_synth += n
            if len(examples) < 4 and n >= 2:
                examples.append((a, imgs[len(imgs) // 2], b, prev, ci, gap))
        emit(img_path=os.path.join(src_dir, files[ci]), kind="real", source_index=ci,
             sharpness_rel=round(float(rel[ci]), 3), gap_px=round(gap, 1))

    real_gaps = np.array(gaps[1:])
    summary = {
        "candidates": len(files),
        "kept_real": len(kept),
        "synthetic": n_synth,
        "total": len(manifest),
        "step_px": args.step_px,
        "blur_rel": args.blur_rel,
        "synth_factor": args.synth_factor,
        "gap_px_median": float(np.median(real_gaps)),
        "gap_px_p90": float(np.percentile(real_gaps, 90)),
        "gap_px_max": float(real_gaps.max()),
        "kept_blurry_fraction": float((rel[kept] < args.blur_rel).mean()),
        "gaps_too_large_unfilled": int((real_gaps > args.max_synth_gap).sum()),
    }
    with open(os.path.join(args.out_dir, "manifest.json"), "w") as f:
        json.dump({"summary": summary, "frames": manifest}, f, indent=1)
    print(json.dumps(summary, indent=1))

    if examples:
        rows = []
        for a, m, b, pa, pb, gap in examples:
            row = np.concatenate([a, m, b], axis=1)
            cv2.putText(row, f"real {pa}  |  sintetico  |  real {pb}   (salto {gap:.0f}px)", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 255), 2)
            rows.append(cv2.resize(row, (row.shape[1] // 2, row.shape[0] // 2)))
        cv2.imwrite(os.path.join(args.out_dir, "synthetic_examples.png"), np.concatenate(rows, axis=0))


if __name__ == "__main__":
    main()
