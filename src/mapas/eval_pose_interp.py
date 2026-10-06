#!/usr/bin/env python3
"""¿Cuánto se equivoca la interpolación de poses según el hueco entre muestras? (etapa 5)

Sobre trayectorias reales densas: para cada muestra i se buscan dos vecinas j < i < k cuya
separación sea ~g segundos, se interpola en t_i (centro lineal + SLERP, como PoseBuffer) y se
compara con la pose medida en i. El resultado fija con datos la tolerancia `max_gap` del buffer.

Error de posición normalizado por el desplazamiento típico en 1 s de esa misma trayectoria
(así es comparable entre fuentes con escalas distintas y con el RPE a 1 s de la etapa 4).

  python3 src/mapas/eval_pose_interp.py --video V.mp4 --out OUT.json \\
      stella=stella_run:DIR ref=windowed:NPZ [--gaps 0.05,0.1,...]
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "vivo"))
sys.path.insert(0, HERE)

import compare_tracking as ct  # noqa: E402
from pose_buffer import interpolate_pose  # noqa: E402
from traj_align import rot_angle_deg  # noqa: E402


def disp_1s(t, P):
    d = []
    for i in range(len(t)):
        j = int(np.searchsorted(t, t[i] + 1.0))
        if j < len(t) and abs(t[j] - t[i] - 1.0) < 0.15:
            d.append(np.linalg.norm(P[j] - P[i]))
    return float(np.median(d)) if d else float("nan")


def evaluate(src, gaps, tol_rel=0.25, seg=None):
    t, T = src["t"], src["T"]
    segs = src["segments"] if src["segments"] is not None else np.zeros(len(t), int)
    D = disp_1s(t, T[:, :3, 3])
    rows = []
    for g in gaps:
        ep, er = [], []
        for i in range(1, len(t) - 1):
            # vecinas: la combinación (j, k) con j < i < k y t_k - t_j más cercana a g
            best = None
            for j in range(i - 1, -1, -1):
                if t[i] - t[j] > g:
                    break
                k = int(np.searchsorted(t, t[j] + g))
                for kk in (k - 1, k):
                    if i < kk < len(t) and segs[j] == segs[i] == segs[kk]:
                        err = abs(t[kk] - t[j] - g)
                        if best is None or err < best[0]:
                            best = (err, j, kk)
            if best is None or best[0] > tol_rel * g:
                continue
            _, j, k = best
            a = (t[i] - t[j]) / (t[k] - t[j])
            Ti = interpolate_pose(T[j], T[k], a)
            ep.append(np.linalg.norm(Ti[:3, 3] - T[i, :3, 3]) / D)
            er.append(float(rot_angle_deg(Ti[:3, :3].T @ T[i, :3, :3])))
        ep, er = np.array(ep), np.array(er)
        rows.append({"hueco_s": g, "n": int(len(ep)),
                     "pos_pct_1s_mediana": round(100 * float(np.median(ep)), 2) if len(ep) else None,
                     "pos_pct_1s_p90": round(100 * float(np.percentile(ep, 90)), 2) if len(ep) else None,
                     "rot_deg_mediana": round(float(np.median(er)), 3) if len(er) else None,
                     "rot_deg_p90": round(float(np.percentile(er, 90)), 3) if len(er) else None})
    dt = np.diff(t)
    return {"fuente": src["name"], "muestras": int(len(t)), "paso_mediano_s": round(float(np.median(dt)), 4),
            "desplazamiento_1s": D, "por_hueco": rows}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sources", nargs="+")
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cache", default=None)
    ap.add_argument("--gaps", default="0.067,0.1,0.15,0.2,0.25,0.35,0.5,0.75,1.0")
    a = ap.parse_args()
    clock = ct.VideoClock(a.video)
    gaps = [float(x) for x in a.gaps.split(",")]
    res = []
    for spec in a.sources:
        name, rest = spec.split("=", 1)
        src = ct.load(name, rest, clock, None, a.cache or os.path.join(os.path.dirname(a.out), "cache"))
        r = evaluate(src, gaps)
        res.append(r)
        print(f"== {name}: {r['muestras']} muestras, paso mediano {r['paso_mediano_s']} s", flush=True)
        for row in r["por_hueco"]:
            print(f"   hueco {row['hueco_s']:5.3f} s  n={row['n']:4d}  pos {row['pos_pct_1s_mediana']}% "
                  f"(p90 {row['pos_pct_1s_p90']}%) del desplaz. en 1 s   rot {row['rot_deg_mediana']}° "
                  f"(p90 {row['rot_deg_p90']}°)", flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
