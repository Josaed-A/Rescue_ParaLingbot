#!/usr/bin/env python3
"""Registra la geometría de LingBot con una pose de referencia y mide el mapa resultante
(integración Stella-VSLAM, etapas 9, 10, 11, 13 y 15).

LingBot no se toca: su profundidad, confianza e intrínsecos por frame se toman del .npz de la sesión
y se registran en el mundo con la pose que se elija (externa al modelo, etapa 10):

  --reference basic           pose BASIC (pose_basic): el baseline
  --reference hybrid|stella   pose de referencia del selector grabada en la sesión (pose_ref_*)
  --reference NPZ:CLAVE       poses de otro archivo con `stamps` (p. ej. simulate_hybrid.py o la
                              corrección por keyframes), asociadas por stamp a los frames de la sesión

La geometría se acumula en un MapAccumulator (puntos en coordenadas de cámara + pose por frame, etapa
13) y se fusiona por vóxel. Opcional: --sky quita el cielo con skyseg.onnx (etapa 15: filtro
auxiliar, no reemplaza a LingBot).

Métricas (etapa 11), todas sin ground truth salvo el croquis:
  consistencia multivista   proyectar la profundidad del frame a en el frame b con las poses de
                            referencia y comparar con la profundidad que LingBot predijo en b
                            (fracción a <5%). Si la pose (o su escala) no es coherente con la
                            geometría, baja. Pares consecutivos y a ~1 s.
  duplicación               puntos crudos por vóxel y fracción de vóxeles vistos por >=2 frames
  continuidad               jerk y saltos de rotación de la trayectoria de referencia
  deriva                    |fin - inicio| / largo (útil si el recorrido vuelve al inicio) y error
                            de forma contra el croquis (--sketch)
  escala (etapa 9)          escala Sim(3) de la referencia frente a BASIC y su variación en ventanas

Salidas en --out: <nombre>_mapa.ply (fusionado), <nombre>_registrado.npz (la sesión con `extrinsic`
= referencia, `extrinsic_basic` = BASIC, para build_maps.py: TSDF y splat), <nombre>_metricas.json.

  python3 src/mapas/register_map.py SESION.npz --reference hybrid --out DIR --name hybrid
"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, "src/vivo"))
sys.path.insert(0, HERE)

from map_accumulator import MapAccumulator, backproject, write_ply  # noqa: E402
from traj_align import path_length, umeyama, windowed_scale  # noqa: E402


def load_reference(d, ref):
    stamps = d["stamps"]
    if ref == "basic":
        return d["pose_basic"].astype(np.float64)
    if ref in ("hybrid", "stella"):
        T = d[f"pose_ref_{ref}"].astype(np.float64)
        if np.isnan(T).all():
            raise SystemExit(f"la sesión no tiene pose_ref_{ref} (se grabó sin el puente ROS2)")
        return T
    path, key = ref.rsplit(":", 1)
    e = np.load(path)
    lut = {round(float(t), 4): i for i, t in enumerate(e["stamps"])}
    T = np.full((len(stamps), 4, 4), np.nan)
    for i, t in enumerate(stamps):
        j = lut.get(round(float(t), 4))
        if j is not None:
            T[i] = e[key][j]
    return T


def sky_masks(images, model_path):
    import onnxruntime
    sys.path.insert(0, REPO)
    from lingbot_map.vis.sky_segmentation import segment_sky_from_array
    sess = onnxruntime.InferenceSession(model_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    H, W = images.shape[1:3]
    return np.stack([segment_sky_from_array(im, sess, H, W) > 0.5 for im in images])


def consistency(depth, conf, images, K, c2w, lag_s, stamps, stride=3):
    """Pares (consecutivos y a ~lag_s): mediana de la fracción de inliers <5% (evaluate_consistency)."""
    from evaluate_consistency import pair_metrics
    run = {"depth": depth, "depth_conf": conf, "images": images, "K": K, "c2w": c2w[:, :3, :4]}
    ok = np.flatnonzero(np.isfinite(c2w[:, 0, 0]))
    res = {}
    rows = [pair_metrics(run, a, b, stride=stride) for a, b in zip(ok[:-1], ok[1:])]
    rows = [r for r in rows if r]
    res["consecutivos"] = {"pares": len(rows), "inlier": round(float(np.median([r["inlier"] for r in rows])), 4) if rows else None}
    rows = []
    for i in ok:
        j = ok[np.argmin(np.abs(stamps[ok] - (stamps[i] + lag_s)))]
        if j > i and abs(stamps[j] - stamps[i] - lag_s) < 0.35 * lag_s:
            r = pair_metrics(run, i, j, stride=stride)
            if r:
                rows.append(r)
    res[f"a_{lag_s:g}s"] = {"pares": len(rows), "inlier": round(float(np.median([r["inlier"] for r in rows])), 4) if rows else None,
                           "relerr": round(float(np.median([r["relerr"] for r in rows])), 4) if rows else None}
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("session")
    ap.add_argument("--reference", default="basic")
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", default=None)
    ap.add_argument("--conf_percentile", type=float, default=30.0)
    ap.add_argument("--points_per_frame", type=int, default=20000)
    ap.add_argument("--voxel_rel", type=float, default=0.01, help="vóxel = este factor × profundidad mediana")
    ap.add_argument("--sky", action="store_true", help="quitar cielo con skyseg.onnx (etapa 15)")
    ap.add_argument("--sky_model", default=os.path.join(REPO, "skyseg.onnx"))
    ap.add_argument("--sketch", default=None, help="croquis de la ruta (compare_route.py)")
    ap.add_argument("--lag", type=float, default=1.0)
    a = ap.parse_args()
    name = a.name or a.reference.replace(":", "_").replace("/", "_")
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    d = np.load(a.session)
    stamps = d["stamps"]
    T = load_reference(d, a.reference)
    ok = np.isfinite(T[:, 0, 0])
    depth = d["depth"].astype(np.float32)
    conf = d["depth_conf"].astype(np.float32)
    images = d["images"]
    ds = int(d["ds"])
    K = d["intrinsic"].astype(np.float64).copy()
    K[:, :2, :] /= ds
    thr = float(np.percentile(conf, a.conf_percentile))
    med = float(np.median(depth[depth > 0]))
    voxel = a.voxel_rel * med

    masks = None
    sky_info = None
    if a.sky:
        masks = sky_masks(images, a.sky_model)
        sky_info = {"fraccion_pixeles_cielo": round(float(1 - masks.mean()), 4),
                    "frames_con_cielo_>5%": int(((1 - masks.reshape(len(masks), -1).mean(1)) > 0.05).sum())}

    acc = MapAccumulator(voxel=voxel, max_points_per_frame=a.points_per_frame)
    rng = np.random.default_rng(0)
    removed_sky = 0
    for i in np.flatnonzero(ok):
        P, (v, u) = backproject(depth[i], K[i], conf[i], thr, None if masks is None else masks[i], None, rng)
        if masks is not None:
            P_all, _ = backproject(depth[i], K[i], conf[i], thr)
            removed_sky += len(P_all) - len(P)
        acc.add_frame(int(i), P, images[i][v, u], T[i], float(stamps[i]))
    Pf, Cf, nfr, cnt = acc.fused()
    write_ply(os.path.join(a.out, f"{name}_mapa.ply"), Pf, Cf)

    # npz registrado para build_maps.py (TSDF / splat) y el visor
    keys = {k: d[k] for k in d.files}
    E = np.linalg.inv(np.where(np.isfinite(T), T, d["pose_basic"]))[:, :3, :4].astype(np.float32)
    keys["extrinsic_basic"] = d["extrinsic"]
    keys["extrinsic"] = E
    keys["referencia_registro"] = a.reference
    if masks is not None:
        keys["depth"] = np.where(masks, d["depth"], 0).astype(d["depth"].dtype)
    np.savez(os.path.join(a.out, f"{name}_registrado.npz"), **keys)

    # métricas
    Pc = T[ok, :3, 3]
    L = path_length(Pc)
    from evaluate_consistency import trajectory as traj_metrics
    run_t = {"c2w": T[:, :3, :4]}
    m = {"sesion": a.session, "referencia": a.reference, "frames": int(len(stamps)), "frames_con_pose": int(ok.sum()),
         "mapa": acc.stats(voxel), "consistencia_multivista": consistency(depth, conf, images, K, T, a.lag, stamps),
         "continuidad": traj_metrics(run_t, np.flatnonzero(ok)),
         "deriva": {"largo": round(L, 4), "cierre_fin_inicio_pct": round(100 * float(np.linalg.norm(Pc[-1] - Pc[0])) / max(L, 1e-9), 3)},
         "cielo": sky_info, "puntos_quitados_por_cielo": removed_sky if a.sky else None}
    if a.reference != "basic":
        Pb = d["pose_basic"][ok, :3, 3].astype(np.float64)
        s, _, _ = umeyama(Pc, Pb)
        cs, ss = windowed_scale(stamps[ok], Pb, Pc, win=5.0, step=1.0)
        m["escala_vs_basic"] = {"sim3_global": round(float(s), 4),
                                "ventanas_5s_cv": round(float(np.std(ss) / np.mean(ss)), 4) if len(ss) else None}
    if a.sketch:
        import subprocess
        tmp = os.path.join(a.out, f"{name}_para_croquis.npz")
        np.savez(tmp, extrinsic=np.linalg.inv(T[ok])[:, :3, :4].astype(np.float32), is_real=np.ones(int(ok.sum()), bool))
        js = os.path.join(a.out, f"{name}_croquis.json")
        subprocess.run([sys.executable, os.path.join(HERE, "compare_route.py"), "--npz", tmp, "--sketch", a.sketch,
                        "--out_json", js, "--out_png", js.replace(".json", ".png"), "--label", name], capture_output=True)
        os.remove(tmp)
        if os.path.isfile(js):
            c = json.load(open(js))
            m["croquis"] = {k: c[k] for k in ("error_pct", "error_p90_pct", "length_ratio", "straightness_est")}
    m["segundos"] = round(time.time() - t0, 1)
    json.dump(m, open(os.path.join(a.out, f"{name}_metricas.json"), "w"), indent=1, ensure_ascii=False)
    print(json.dumps({k: m[k] for k in ("referencia", "mapa", "consistencia_multivista", "deriva")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
