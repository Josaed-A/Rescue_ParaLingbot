#!/usr/bin/env python3
"""Interfaz para evaluar GenPC (completado de geometría) DESPUÉS de la estabilización. No integra GenPC.

GenPC (u otro completado) no forma parte del pipeline: esta interfaz fija qué recibe, qué tiene que devolver y
cómo se valida, para poder probarlo sin mezclar geometría inferida con la observada.

  exportar   la geometría observada, registrada y filtrada (o3d_estable.py) como entrada:
             <out>/entrada.ply (puntos, color, normales), <out>/entrada.json (procedencia, cámaras, versión de
             poses, unidades). Cada punto es "observado".
  validar    lo que devuelva el completado (<out>/salida.ply). Todo punto que no esté en la entrada es
             "inferido" y se contrasta con el espacio libre observado: si una cámara vio MÁS LEJOS a través de
             él (el rayo pasa por ese lugar hasta una superficie detrás), el punto contradice una observación
             (una puerta cerrada, un pasillo tapado, una pared inventada). Se mide además cuánto completa
             (vóxeles nuevos fuera del espacio libre) y si se movió alguna observación (no debe).

Nada de esto toca la geometría observada ni sustituye observaciones: un punto inferido nunca se usa como
geometría física confirmada sin pasar esta validación (y, aun así, queda marcado como inferido).

    python3 src/mapas/genpc_interfaz.py exportar eval/mapa.npz --estable exports/estable/mapa --out exports/genpc
    python3 src/mapas/genpc_interfaz.py validar eval/mapa.npz --estable exports/estable/mapa --out exports/genpc
"""
import argparse
import json
import os

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree


def free_space_violations(Q, depth, K, c2w, frames, margin_rel=0.05, stride=2):
    """Para cada punto Q (mundo), si alguna cámara lo ve DELANTE de lo que observó en ese píxel (más cerca que su
    profundidad, con un margen relativo): ese lugar se vio libre. Devuelve la máscara de puntos en espacio libre."""
    bad = np.zeros(len(Q), bool)
    H, W = depth.shape[1:3]
    for i in frames:
        w2c = np.linalg.inv(c2w[i])
        pc = Q @ w2c[:3, :3].T + w2c[:3, 3]
        z = pc[:, 2]
        ok = z > 1e-6
        u = np.full(len(Q), -1)
        v = np.full(len(Q), -1)
        u[ok] = np.round(K[i][0, 0] * pc[ok, 0] / z[ok] + K[i][0, 2]).astype(int)
        v[ok] = np.round(K[i][1, 1] * pc[ok, 1] / z[ok] + K[i][1, 2]).astype(int)
        inside = ok & (u >= 0) & (u < W) & (v >= 0) & (v < H)
        if not inside.any():
            continue
        d = depth[i][v[inside], u[inside]].astype(np.float64)
        seen = (d > 0) & (z[inside] < d * (1 - margin_rel))      # la cámara vio una superficie más lejos
        idx = np.flatnonzero(inside)[seen]
        bad[idx] = True
    return bad


def validate(observed, completed, depth, K, c2w, frames, voxel, margin_rel=0.05):
    """observed, completed: puntos (N x 3). Clasifica los puntos del completado en observados e inferidos y mide."""
    tree = cKDTree(observed)
    d, _ = tree.query(completed, k=1)
    inferred = d > 0.5 * voxel
    q = completed[inferred]
    viol = free_space_violations(q, depth, K, c2w, frames, margin_rel) if len(q) else np.zeros(0, bool)
    # Observaciones que el completado movió o borró: puntos observados sin un punto del completado al lado.
    back, _ = cKDTree(completed).query(observed, k=1)
    key = lambda P: np.unique(np.floor(P / voxel).astype(np.int64), axis=0)
    new_vox = len(key(q[~viol])) if len(q) else 0
    return {"puntos_completado": int(len(completed)), "observados": int((~inferred).sum()),
            "inferidos": int(inferred.sum()), "inferidos_en_espacio_libre": int(viol.sum()),
            "frac_inferidos_que_contradicen": float(viol.mean()) if len(viol) else 0.0,
            "observaciones_perdidas": int((back > 0.5 * voxel).sum()),
            "voxeles_nuevos_validos": int(new_vox), "voxeles_observados": int(len(key(observed))),
            "aceptable": bool((viol.mean() if len(viol) else 0.0) < 0.01 and (back > 0.5 * voxel).mean() < 0.001),
            "mascara_inferidos": inferred, "mascara_contradicen": viol}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["exportar", "validar"])
    ap.add_argument("npz")
    ap.add_argument("--estable", required=True, help="prefijo de o3d_estable.py (<pref>_estable.npz, ...)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    z = np.load(a.estable + "_estable.npz")
    meta = json.loads(str(z["meta"]))
    if a.cmd == "exportar":
        pc = o3d.io.read_point_cloud(a.estable + "_nube_estable.ply")
        o3d.io.write_point_cloud(os.path.join(a.out, "entrada.ply"), pc)
        json.dump({"procedencia": "observada, registrada y filtrada (o3d_estable.py)", "puntos": len(pc.points),
                   "version_poses": meta["version_poses"], "sesion": meta["sesion"], "unidades": meta["unidades"],
                   "marco": meta["marco"], "camaras_c2w": z["c2w"].tolist(),
                   "contrato": "la salida (salida.ply) debe contener todos los puntos de entrada sin moverlos; los "
                               "puntos nuevos son inferidos y se validan contra el espacio libre observado"},
                  open(os.path.join(a.out, "entrada.json"), "w"))
        print(f"entrada para el completado: {len(pc.points)} puntos observados -> {a.out}")
        return
    d = np.load(a.npz)
    real = z["real"]
    obs = np.asarray(o3d.io.read_point_cloud(os.path.join(a.out, "entrada.ply")).points)
    comp = np.asarray(o3d.io.read_point_cloud(os.path.join(a.out, "salida.ply")).points)
    sample = d["depth"][real[::10]].astype(np.float32)
    voxel = float(np.median(sample[sample > 0])) / 150
    depth = d["depth"]
    c2w = np.zeros((len(d["is_real"]), 4, 4))
    c2w[real] = z["c2w"]
    res = validate(obs, comp, depth, d["intrinsic"].astype(np.float64), c2w, real[::4], voxel)
    res.pop("mascara_inferidos")
    res.pop("mascara_contradicen")
    json.dump(res, open(os.path.join(a.out, "validacion.json"), "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
