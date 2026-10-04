"""Espesor aparente de las paredes de una reconstrucción (nube, malla o splat).

Para cada tramo de pared de <out>_estructura.json (geo_filter.py) toma los puntos de la
geometría que caen dentro del rectángulo del tramo y a menos de --window (x prof. mediana)
de su plano, y mide la dispersión de su distancia con signo al plano: p90 - p10. Una pared
bien reconstruida es una lámina (espesor chico); una pared doble son dos láminas (espesor
grande y "fracción fuera" alta: puntos a más de --outer de la lámina principal).

Aviso: con ajuste a planos (depth_snap) la profundidad se lleva justo a estos planos, así
que en esas variantes la métrica es en parte circular. La variante solo con máscaras no lo es.
"""
import argparse
import json
import os

import numpy as np


def read_points(path, min_opacity=0.5, n_surface=2_000_000):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npz":                                       # nube de trabajo de geo_filter --dump_points
        return np.load(path)["P"]
    if ext == ".glb" or path.endswith("_malla.ply"):
        import trimesh
        m = trimesh.load(path, force="mesh")
        pts, _ = trimesh.sample.sample_surface(m, n_surface, seed=0)
        return np.asarray(pts)
    with open(path, "rb") as f:
        head = b""
        while not head.endswith(b"end_header\n"):
            head += f.readline()
        names, n = [], 0
        for line in head.decode().splitlines():
            if line.startswith("element vertex"):
                n = int(line.split()[-1])
            elif line.startswith("property"):
                names.append((line.split()[1], line.split()[2]))
        dt = np.dtype([(nm, {"float": "<f4", "double": "<f8", "uchar": "u1"}[t]) for t, nm in names])
        data = np.fromfile(f, dtype=dt, count=n)
    pts = np.stack([data["x"], data["y"], data["z"]], 1).astype(np.float64)
    if "opacity" in data.dtype.names:                       # splat: solo gaussianas opacas
        pts = pts[1 / (1 + np.exp(-data["opacity"])) > min_opacity]
    return pts


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("structure")
    ap.add_argument("geometry", nargs="+", help="nombre=archivo (.ply nube, _splat.ply, _malla.ply o .glb)")
    ap.add_argument("--window", type=float, default=0.15, help="semiancho de la banda alrededor del plano")
    ap.add_argument("--outer", type=float, default=0.05, help="'fuera de la lámina' (x prof. mediana)")
    ap.add_argument("--out_json", default=None)
    a = ap.parse_args()
    st = json.load(open(a.structure))
    up = np.array(st["up"])
    med = st["median_depth"]
    res = {}
    for g in a.geometry:
        name, path = g.split("=", 1) if "=" in g else (os.path.basename(g), g)
        P = read_points(path)
        h = P @ up
        th, fr, w = [], [], []
        for s in st["walls"]:
            n = np.array(s["n"])
            t = np.cross(up, n)
            dist = P @ n - s["d"]
            u = P @ t
            m = (np.abs(dist) < a.window * med) & (u > s["u0"]) & (u < s["u1"]) & (h > s["h0"]) & (h < s["h1"])
            if m.sum() < 200:
                continue
            dd = dist[m]
            core = np.median(dd)
            th.append((np.percentile(dd, 90) - np.percentile(dd, 10)) / med)
            fr.append(float((np.abs(dd - core) > a.outer * med).mean()))
            w.append(m.sum())
        w = np.array(w, float)
        res[name] = {"walls": len(th),
                     "thickness_rel_median": round(float(np.median(th)), 4) if th else None,
                     "thickness_rel_weighted": round(float((np.array(th) * w).sum() / w.sum()), 4) if th else None,
                     "outside_frac_weighted": round(float((np.array(fr) * w).sum() / w.sum()), 4) if th else None,
                     "points": int(len(P))}
        print(name, res[name], flush=True)
    if a.out_json:
        json.dump(res, open(a.out_json, "w"), indent=1)


if __name__ == "__main__":
    main()
