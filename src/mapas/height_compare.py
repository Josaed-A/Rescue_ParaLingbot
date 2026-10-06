#!/usr/bin/env python3
"""Compara varias corridas del mismo video: planta (vista desde arriba) y perfil de altura.

    python src/mapas/height_compare.py --out comp.png nombre=a.npz nombre2=b.npz ...
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from height_profile import load, vertical  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--fps", type=float, default=10.0)
    ap.add_argument("runs", nargs="+", help="etiqueta=archivo.npz")
    a = ap.parse_args()
    S = 420
    cols = []
    for spec in a.runs:
        name, path = spec.split("=", 1)
        d, real, c2w = load(path)
        c2w = c2w[real]
        up, _ = vertical(c2w)
        C = c2w[:, :3, 3]
        h = C @ up
        h -= h[:30].mean()
        e1 = np.cross(up, [0, 0, 1.0]); e1 /= np.linalg.norm(e1)
        e2 = np.cross(up, e1)
        xy = np.stack([C @ e1, C @ e2], 1)
        xy -= xy.mean(0)
        k = (S - 40) / (np.abs(xy).max() * 2 + 1e-9)
        top = np.full((S, S, 3), 255, np.uint8)
        n = len(xy)
        for i in range(n - 1):
            c = cv2.applyColorMap(np.uint8([[255 * i / n]]), cv2.COLORMAP_VIRIDIS)[0, 0].tolist()
            p0 = tuple(int(v) for v in xy[i] * k + S / 2); p1 = tuple(int(v) for v in xy[i + 1] * k + S / 2)
            cv2.line(top, p0, p1, c, 2)
        cv2.circle(top, tuple(int(v) for v in xy[0] * k + S / 2), 6, (0, 0, 220), -1)
        prof = np.full((200, S, 3), 255, np.uint8)
        lo, hi = h.min(), h.max()
        t = np.arange(n)
        pts = np.stack([t / (n - 1) * (S - 20) + 10, 190 - (h - lo) / (hi - lo + 1e-9) * 170], 1).astype(np.int32)
        cv2.polylines(prof, [pts], False, (40, 90, 200), 2)
        y0 = int(190 - (0 - lo) / (hi - lo + 1e-9) * 170)
        cv2.line(prof, (10, y0), (S - 10, y0), (180, 180, 180), 1)
        for x in range(0, int(n / a.fps) + 1, 10):
            px = int(x * a.fps / (n - 1) * (S - 20) + 10)
            cv2.putText(prof, f"{x}s", (px, 198), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)
        col = np.vstack([top, prof])
        cv2.putText(col, name, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        cv2.putText(col, "planta (rojo = inicio)", (10, S - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (90, 90, 90), 1)
        cv2.putText(col, "altura vs tiempo", (10, S + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (90, 90, 90), 1)
        cols.append(np.pad(col, ((0, 0), (0, 6), (0, 0)), constant_values=200))
    cv2.imwrite(a.out, np.hstack(cols))


if __name__ == "__main__":
    main()
