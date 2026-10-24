#!/usr/bin/env python3
"""Comparación de la geometría de una prueba antes y después de la estabilización con Open3D.

  A  nube original (poses de LingBot, sin filtrar)           <out>_nube_original.ply  (o3d_estable.py)
  B  nube estabilizada (grafo de poses + filtrado)           <out>_nube_estable.ply
  C  mallas TSDF de línea base: sin filtro y con el filtro   <name>_base_h8, <name>_filtrada_h8 (tsdf_mesh.py)
  D  malla TSDF con la estabilización                        <name>_estable_h8 (tsdf_mesh.py --stable)

Todas las mallas se construyen con 1 de cada 8 frames apartados (los mismos) y se miden en esos frames: color
(PSNR, SSIM) y coincidencia de profundidad (inlier5), la medida que no sale de los frames que construyen. El
espesor de pared (wall_thickness.py) usa la estructura del filtro geométrico, calculada con las poses de LingBot:
vale para D solo si la estabilización conservó las poses (se indica). Sin referencia externa, todo es
consistencia interna o reproyección, no error absoluto.

    python3 src/mapas/comparar_geometria.py <prueba> --name final_m2
      -> exports/variantes/estable/comparacion_<name>.json y .png
"""
import argparse
import json
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def load(path):
    return json.load(open(path)) if os.path.isfile(path) else None


def mesh_row(info):
    if info is None:
        return None
    t = info.get("test", {})
    to = info.get("test_original_poses")
    return {"triangulos": info["triangles"], "segundos": info.get("seconds"),
            "inlier5_frames_usados": info["pass1"]["inlier5_mean"], "error_rel_mediano": info["pass1"]["rel_err_median"],
            "apartados_psnr": t.get("psnr"), "apartados_ssim": t.get("ssim"), "apartados_inlier5": t.get("inlier5"),
            "apartados_con_poses_originales": to}


def slice_image(paths, up, height, band, size=520):
    """Corte horizontal de cada malla (puntos de su superficie a `height` ± `band` sobre la vertical)."""
    import cv2
    import trimesh

    pts = []
    for p in paths:
        m = trimesh.load(p, force="mesh")
        P, _ = trimesh.sample.sample_surface(m, 600_000, seed=0)
        P = np.asarray(P)
        h = P @ up
        pts.append(P[np.abs(h - height) < band])
    a = np.cross(up, [1.0, 0, 0] if abs(up[0]) < 0.9 else [0, 1.0, 0])
    a /= np.linalg.norm(a)
    b = np.cross(up, a)
    allp = np.concatenate([np.c_[q @ a, q @ b] for q in pts if len(q)]) if any(len(q) for q in pts) else np.zeros((1, 2))
    lo, hi = np.percentile(allp, 0.5, 0), np.percentile(allp, 99.5, 0)
    s = (size - 40) / max((hi - lo).max(), 1e-9)
    tiles = []
    for q, p in zip(pts, paths):
        img = np.full((size, size, 3), 255, np.uint8)
        if len(q):
            uv = ((np.c_[q @ a, q @ b] - lo) * s + 20).astype(int)
            ok = (uv >= 0).all(1) & (uv < size).all(1)
            img[size - 1 - uv[ok, 1], uv[ok, 0]] = (40, 40, 40)
        cv2.putText(img, os.path.basename(p)[:48], (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 200), 1, cv2.LINE_AA)
        tiles.append(img)
    return np.hstack(tiles)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("prueba")
    ap.add_argument("--name", required=True)
    a = ap.parse_args()
    P = os.path.abspath(a.prueba)
    var = os.path.join(P, "exports", "variantes", "estable")
    rep = load(os.path.join(var, f"{a.name}_h8_estable.json"))
    if rep is None:
        sys.exit(f"falta {a.name}_h8_estable.json en {var}")
    out = {"prueba": os.path.basename(P), "nombre": a.name, "sesion": rep["sesion"],
           "grafo_poses": {k: rep["grafo_poses"][k] for k in ("decision", "nodos", "aristas", "conteo_aristas")},
           "correccion_por_frame": rep["correccion_por_frame"], "validacion": {
               "frames_repetidos": len(rep["validacion"]["frames_repetidos"]),
               "saltos": len(rep["validacion"]["saltos"]["saltos_traslacion"]),
               "giros_mayores_20deg": rep["validacion"]["saltos"]["giros_mayores_20deg"],
               "profundidad": rep["validacion"]["profundidad"]},
           "filtrado": rep["filtrado"], "tiempos_estabilizacion_s": rep["tiempos_s"]}
    for k in ("original", "estable"):
        r = rep["diagnostico"][k]["resumen"]
        out[f"nube_{k}"] = {"superficies_paralelas": r["por_clase"], "pares": r["pares_por_clase"],
                            "duplicadas_separacion_rel_mediana": r["duplicadas_separacion_rel_mediana"],
                            "espesor_local_rel": r["espesor_local_rel"], "estados_estructura": r["estados_estructura"]}
    for k in ("base", "filtrada", "estable"):
        out[f"malla_{k}"] = mesh_row(load(os.path.join(var, f"{a.name}_{k}_h8_info.json")))
    # Espesor de pared contra la estructura del filtro (poses de LingBot).
    st = os.path.join(P, "exports", "estructura", f"{a.name}_estructura.json")
    poses_kept = rep["correccion_por_frame"]["camara_desplazamiento_max"] < 1e-9
    geoms = [f"A_nube_original={var}/{a.name}_h8_nube_original.ply", f"B_nube_estable={var}/{a.name}_h8_nube_estable.ply"]
    for k in ("base", "filtrada", "estable"):
        g = os.path.join(var, f"{a.name}_{k}_h8.glb")
        if os.path.isfile(g):
            geoms.append(f"malla_{k}={g}")
    if os.path.isfile(st):
        tj = os.path.join(var, f"comparacion_{a.name}_espesor.json")
        subprocess.run([sys.executable, os.path.join(HERE, "wall_thickness.py"), st, *geoms, "--out_json", tj],
                       check=True, capture_output=True)
        out["espesor_pared"] = load(tj)
        out["espesor_pared_valido_para_estable"] = bool(poses_kept)
        S = load(st)
        up = np.array(S["up"])
        d = np.load(os.path.join(P, "eval", f"{a.name}.npz"))
        real = np.flatnonzero(d["is_real"])
        E = np.tile(np.eye(4), (len(real), 1, 1))
        E[:, :3, :4] = d["extrinsic"][real]
        cam_h = float(np.median(np.linalg.inv(E)[:, :3, 3] @ up))
        floors = [f["height"] for f in S.get("floors", []) if f["height"] < cam_h]
        h = (max(floors) + cam_h) / 2 if floors else cam_h - 0.3 * S["median_depth"]
        meshes = [os.path.join(var, f"{a.name}_{k}_h8.glb") for k in ("filtrada", "estable")]
        if all(os.path.isfile(m) for m in meshes):
            import cv2
            png = os.path.join(var, f"comparacion_{a.name}.png")
            cv2.imwrite(png, slice_image(meshes, up, h, 0.02 * S["median_depth"]))
            out["corte"] = {"png": png, "altura": h, "nota": "corte horizontal a media altura entre el piso y la cámara"}
    path = os.path.join(var, f"comparacion_{a.name}.json")
    json.dump(out, open(path, "w"), indent=1, ensure_ascii=False)
    print(json.dumps(out, indent=1, ensure_ascii=False)[:6000])


if __name__ == "__main__":
    main()
