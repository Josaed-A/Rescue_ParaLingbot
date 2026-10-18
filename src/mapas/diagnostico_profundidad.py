#!/usr/bin/env python3
"""Diagnóstico: ¿el error espacial entre visitas viene de la profundidad, de las poses o de los intrínsecos?

Cada frame se compara con una superficie de referencia hecha con OTROS momentos del recorrido (fuera de su bloque
temporal y de un margen: sus vecinos comparten sus errores). Sobre esa referencia se ajustan, por frame y punto a
plano, modelos de corrección de complejidad creciente:

  A  nada (la profundidad y la pose originales)
  C  escala de la profundidad:              X = c + s (X0 - c)
  D  escala y desfase de la profundidad:    X = c + (s + t/z) (X0 - c)        (z: profundidad del punto)
  E  escala que varía con la fila:          X = c + (1 + a + b y/z) (X0 - c)   (solo si A-D no explican el error:
                                                                              el piso se ve rasante, abajo en la imagen)
  F  focal (intrínsecos):                   X = X0 + (k - 1) R (x, y, 0)      (x, y: coordenadas laterales de cámara)
  P  pose rígida (6 grados de libertad):    X = X0 + w x (X0 - c) + v

Se ajusta en la mitad de los píxeles (bloques de la imagen en damero) y se mide en la otra mitad, por grupo
(piso, pared, escalera, puerta, otro); y también ajustando sin el piso para medir sobre el piso (y al revés), para
que ninguna conclusión dependa solo del suelo. Un modelo explica el error si baja el residuo en los píxeles no
usados, en todos los grupos, y sus parámetros son estables (error estándar chico, poca dispersión entre bloques).

Cada frame queda como una observación (frame de origen, versión de las poses, profundidad original, píxeles por
grupo, correspondencias, parámetros con su error estándar y validación) en <out>.json: el grafo de observaciones
sin otro optimizador. No modifica nada: solo lee el .npz (y opcionalmente las poses de una estabilización).

    python3 src/mapas/diagnostico_profundidad.py <prueba>/eval/<name>.npz --filter <..>_filtro.npz --out <prefijo>
        [--poses <..>_estable.npz]     # medir con las poses de una estabilización (p. ej. la del piso alineado)
"""
import argparse
import dataclasses
import json
import os
import sys
import time
from dataclasses import dataclass

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import o3d_estable as OE  # noqa: E402

MODELS = ["A", "C", "D", "E", "F", "P"]
GROUP_REPORT = ["piso", "pared", "escalera", "puerta", "otro"]


@dataclass
class Params:
    stride: int = 4
    conf_percentile: float = 35.0
    depth_max_percentile: float = 97.0
    blocks: int = 8                     # bloques temporales: la referencia de un frame excluye su bloque...
    margin_frames: int = 8              # ... y estos frames a cada lado
    ref_voxel_rel: float = 1 / 150      # vóxel de la referencia, x prof. mediana
    min_ref_frames: int = 2             # vóxeles de la referencia vistos por al menos estos frames
    gates: tuple = (0.15, 0.08, 0.05)   # correspondencias: distancia máxima / profundidad, por iteración
    test_gate: float = 0.15             # medición: residuos truncados a esto x profundidad
    split_px: int = 32                  # damero de bloques de la imagen (píxeles) para ajustar / medir
    min_corr: int = 300                 # correspondencias mínimas para ajustar un frame
    every: int = 1                      # medir uno de cada `every` frames
    # aceptación de una corrección (--corregir)
    max_colinealidad: float = 0.995
    max_se: float = 0.02
    min_mejora: float = 0.95            # el residuo de prueba de su grupo tiene que bajar al menos un 5 %
    max_escala: float = 0.2             # plausibilidad: escala +-20 %, desfase <= 0.2 x prof. mediana, giro <= 5°


def basis_for(name, X, c, n, z, R, xl, yl):
    """Columnas del modelo lineal e = e0 + J p (residuo punto a plano)."""
    d = X - c
    if name == "C":
        return (n * d).sum(1)[:, None]
    if name == "D":
        nd = (n * d).sum(1)
        return np.c_[nd, nd / z]
    if name == "E":
        nd = (n * d).sum(1)
        return np.c_[nd, nd * yl / z]
    if name == "F":
        lat = xl[:, None] * R[:, 0][None] + yl[:, None] * R[:, 1][None]
        return (n * lat).sum(1)[:, None]
    if name == "P":
        return np.c_[np.cross(d, n), n]
    raise ValueError(name)


def apply_model(name, p, X, c, z, R, xl, yl):
    d = X - c
    if name == "A":
        return X
    if name == "C":
        return c + (1 + p[0]) * d
    if name == "D":
        return c + (1 + p[0] + p[1] / z)[:, None] * d
    if name == "E":
        return c + (1 + p[0] + p[1] * yl / z)[:, None] * d
    if name == "F":
        return X + p[0] * (xl[:, None] * R[:, 0][None] + yl[:, None] * R[:, 1][None])
    if name == "P":
        w, v = p[:3], p[3:]
        return X + np.cross(w[None], d) + v[None]
    raise ValueError(name)


def fit(name, X, c, z, R, xl, yl, ref, gates, min_corr):
    """Ajuste robusto (correspondencias más cercanas con umbral decreciente, mínimos cuadrados recortados)."""
    tree, RP, RN = ref
    npar = {"A": 0, "C": 1, "D": 2, "E": 2, "F": 1, "P": 6}[name]
    p = np.zeros(npar)
    info = {"n_corr": 0, "se": None, "estable": False}
    if npar == 0:
        return p, info
    for g in gates:
        Xc = apply_model(name, p, X, c, z, R, xl, yl)
        dist, idx = tree.query(Xc, distance_upper_bound=g * float(np.percentile(z, 99)))
        ok = np.isfinite(dist) & (dist < g * z)
        if ok.sum() < min_corr:
            if info["se"] is None:                 # ni la primera iteración tuvo correspondencias: inconcluso
                info["n_corr"] = int(ok.sum())
                return np.zeros(npar), info
            break                                  # se conserva el último ajuste con correspondencias suficientes
        Y, n = RP[idx[ok]], RN[idx[ok]]
        e0 = (n * (Xc[ok] - Y)).sum(1)
        # linealizado alrededor de la solución actual: e(p + dp) = e0 + J dp
        J = basis_for(name, Xc[ok], c, n, z[ok], R, xl[ok], yl[ok])
        keep = np.ones(len(e0), bool)
        for _ in range(3):
            dp, *_ = np.linalg.lstsq(J[keep], -e0[keep], rcond=None)
            res = e0 + J @ dp
            mad = 1.4826 * np.median(np.abs(res[keep])) + 1e-12
            keep = np.abs(res) <= 3 * mad
        p = p + dp if name != "P" else compose_pose(p, dp)
        sigma2 = float(np.mean(res[keep] ** 2))
        JtJ = J[keep].T @ J[keep]
        try:
            info["se"] = np.sqrt(np.diag(np.linalg.inv(JtJ)) * sigma2).tolist()
        except np.linalg.LinAlgError:
            info["se"] = None
        info["n_corr"] = int(keep.sum())
        if J.shape[1] >= 2:
            Jk = J[keep] / (np.linalg.norm(J[keep], axis=0) + 1e-12)
            info["colinealidad"] = float(np.abs(Jk[:, 0] @ Jk[:, 1]))     # 1 = escala y desfase indistinguibles
    info["estable"] = info["se"] is not None and np.all(np.isfinite(info["se"]))
    return p, info


def compose_pose(p, dp):
    # incrementos chicos: suma (aproximación de primer orden, suficiente para correcciones de pocos grados)
    return p + dp


def residuals(Xm, z, ref, cap):
    tree, RP, RN = ref
    dist, idx = tree.query(Xm)
    e = np.abs((RN[idx] * (Xm - RP[idx])).sum(1)) / z
    return np.minimum(e, cap)


def summarize(e, groups, gnames):
    out = {"todos": {"n": int(len(e)), "mediana": float(np.median(e)) if len(e) else None}}
    for gi, gn in enumerate(gnames):
        m = groups == gi
        if m.sum() >= 50:
            out[gn] = {"n": int(m.sum()), "mediana": float(np.median(e[m]))}
    m = groups != gnames.index("piso")
    if m.sum() >= 50:
        out["no_piso"] = {"n": int(m.sum()), "mediana": float(np.median(e[m]))}
    return out


def build_reference(P, F, posF, excl_lo, excl_hi, voxel, min_frames):
    m = (posF < excl_lo) | (posF > excl_hi)
    Q, f = P[m].astype(np.float64), F[m]
    ijk = np.floor(Q / voxel).astype(np.int64) + (1 << 20)
    key = (ijk[:, 0] << 42) | (ijk[:, 1] << 21) | ijk[:, 2]
    uk, inv = np.unique(key, return_inverse=True)
    inv = inv.ravel()
    cnt = np.bincount(inv, minlength=len(uk))
    cen = np.stack([np.bincount(inv, Q[:, k], len(uk)) for k in range(3)], 1) / cnt[:, None]
    # frames distintos por vóxel
    pair = np.unique(inv.astype(np.int64) * (int(f.max()) + 1) + f)
    nfr = np.bincount(pair // (int(f.max()) + 1), minlength=len(uk))
    cen = cen[nfr >= min_frames]
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(cen))
    pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=3 * voxel, max_nn=30))
    return cKDTree(cen), cen, np.asarray(pc.normals)


def run(npz, out, p, filter_path=None, poses_path=None, log=print):
    t0 = time.time()
    d = OE.load_arrays(npz)
    real = np.flatnonzero(d["is_real"])
    E = np.tile(np.eye(4), (len(real), 1, 1))
    E[:, :3, :4] = d["extrinsic"][real]
    c2w = np.linalg.inv(E)
    pose_src = "originales del .npz (BASIC)"
    if poses_path:
        z = np.load(poses_path)
        if not np.array_equal(z["real"], real) or not np.allclose(z["c2w_original"], c2w.astype(np.float32), atol=1e-4):
            raise SystemExit(f"{poses_path} es de otra reconstrucción")
        c2w = z["c2w"].astype(np.float64)
        pose_src = poses_path
    c2w_of = {int(i): c2w[k] for k, i in enumerate(real)}
    sample = d["depth"][real[::10]].astype(np.float32)
    med = float(np.median(sample[sample > 0]))
    dmax = float(np.percentile(sample[sample > 0], p.depth_max_percentile))
    cthr = float(np.percentile(d["depth_conf"][real[::10]].astype(np.float32), p.conf_percentile))
    removed = groups_px = None
    if filter_path:
        flt = np.load(filter_path)
        lut = np.zeros(256, bool)
        lut[OE.REMOVE_DEPTH] = True
        glut = np.zeros(256, np.int64)
        for g, ids_ in enumerate(OE.GROUPS.values(), 1):
            glut[ids_] = g
        removed = {int(i): lut[flt["labels"][k]] for k, i in enumerate(real)}
        groups_px = {int(i): glut[flt["labels"][k]] for k, i in enumerate(real)}
    ids = [int(i) for i in real]
    lp = OE.Params(stride=p.stride, conf_percentile=p.conf_percentile, depth_max_percentile=p.depth_max_percentile)
    P, F, YX, _, G, Z = OE.lift(d, ids, c2w_of, lp, cthr, dmax, removed, groups_px)
    if G is None:
        G = np.zeros(len(P), np.int64)
    rep_idx = {n: i for i, n in enumerate(GROUP_REPORT)}
    Gr = np.array([rep_idx.get(name, rep_idx["otro"]) for name in OE.GROUP_NAMES])[G]
    posF = np.searchsorted(real, F)
    sl = OE.frame_slices(F)
    voxel = med * p.ref_voxel_rel
    log(f"{os.path.basename(npz)}: {len(real)} frames, {len(P) / 1e6:.2f} M puntos, prof. mediana {med:.4f}, "
        f"poses: {pose_src}", flush=True)
    n = len(real)
    bsize = int(np.ceil(n / p.blocks))
    frames_out = []
    agg = {m: {"ajuste_todo": [], "ajuste_sin_piso": [], "ajuste_solo_piso": []} for m in MODELS}
    for b in range(p.blocks):
        lo, hi = b * bsize, min(n, (b + 1) * bsize) - 1
        if lo > hi:
            continue
        ref = build_reference(P, F, posF, lo - p.margin_frames, hi + p.margin_frames, voxel, p.min_ref_frames)
        for k in range(lo, hi + 1, p.every):
            i = int(real[k])
            if i not in sl:
                continue
            a_, b_ = sl[i]
            X0 = P[a_:b_].astype(np.float64)
            z = Z[a_:b_].astype(np.float64)
            yx = YX[a_:b_].astype(np.int64) * p.stride
            gr = Gr[a_:b_]
            T = c2w_of[i]
            c, R = T[:3, 3], T[:3, :3]
            Kf = d["intrinsic"][i].astype(np.float64)
            xl = (yx[:, 1] + p.stride // 2 - Kf[0, 2]) / Kf[0, 0] * z
            yl = (yx[:, 0] + p.stride // 2 - Kf[1, 2]) / Kf[1, 1] * z
            train = ((yx[:, 0] // p.split_px + yx[:, 1] // p.split_px) % 2) == 0
            # Píxeles de prueba: los que, con la profundidad y la pose originales, tienen referencia cerca (mismo
            # conjunto para todos los modelos; sin solapamiento con otros momentos no hay nada que medir).
            dist0, _ = ref[0].query(X0, distance_upper_bound=p.test_gate * float(np.max(z)))
            test = ~train & (dist0 < p.test_gate * z)
            obs = {"frame": i, "source_index": int(d["source_index"][i]) if "source_index" in d else i,
                   "focal": float(Kf[0, 0]), "puntos": int(len(X0)),
                   "puntos_por_grupo": {gn: int((gr == gi).sum()) for gi, gn in enumerate(GROUP_REPORT)},
                   "referencia": {"bloque": b, "excluye_frames": [int(max(0, lo - p.margin_frames)),
                                                                  int(min(n - 1, hi + p.margin_frames))]},
                   "pixeles_prueba": int(test.sum()), "modelos": {}}
            floor = gr == rep_idx["piso"]
            if test.sum() >= 100:
                dist, idx = ref[0].query(X0[test])
                es = (ref[2][idx] * (X0[test] - ref[1][idx])).sum(1) / z[test]
                ok = np.abs(es) < p.test_gate
                zt, rows, gt = z[test][ok], yx[test][ok, 0], gr[test][ok]
                es = es[ok]
                shape = {}
                for key, v in (("distancia", zt), ("fila", rows)):
                    q = np.quantile(v, [1 / 3, 2 / 3]) if len(v) >= 30 else None
                    if q is not None:
                        b_ = np.digitize(v, q)
                        shape[key] = [float(np.median(es[b_ == k])) if (b_ == k).sum() >= 10 else None for k in range(3)]
                shape["piso"] = float(np.median(es[gt == rep_idx["piso"]])) if (gt == rep_idx["piso"]).sum() >= 30 else None
                shape["no_piso"] = float(np.median(es[gt != rep_idx["piso"]])) if (gt != rep_idx["piso"]).sum() >= 30 else None
                obs["forma_error_A"] = shape
            for mode, tr in (("ajuste_todo", train), ("ajuste_sin_piso", train & ~floor),
                             ("ajuste_solo_piso", train & floor)):
                for m in MODELS:
                    if m == "A" and mode != "ajuste_todo":
                        continue
                    sel = tr
                    if sel.sum() < p.min_corr:
                        continue
                    par, info = fit(m, X0[sel], c, z[sel], R, xl[sel], yl[sel], ref, p.gates, p.min_corr)
                    Xm = apply_model(m, par, X0[test], c, z[test], R, xl[test], yl[test])
                    e = residuals(Xm, z[test], ref, p.test_gate)
                    s = summarize(e, gr[test], GROUP_REPORT)
                    obs["modelos"].setdefault(m, {})[mode] = {"parametros": par.tolist(), **info, "prueba": s}
                    agg[m][mode].append(s)
            frames_out.append(obs)
        log(f"bloque {b + 1}/{p.blocks}: frames {lo}-{hi}, referencia {len(ref[1]) / 1e6:.2f} M vóxeles "
            f"({time.time() - t0:.0f} s)", flush=True)

    # resumen: mediana sobre frames de la mediana por frame del residuo relativo en los píxeles de prueba
    def med_over(lst, g):
        v = [s[g]["mediana"] for s in lst if g in s and s[g]["mediana"] is not None]
        return (float(np.median(v)), len(v)) if v else (None, 0)

    summary = {}
    for m in MODELS:
        summary[m] = {}
        for mode in ("ajuste_todo", "ajuste_sin_piso", "ajuste_solo_piso"):
            if agg[m][mode]:
                summary[m][mode] = {g: med_over(agg[m][mode], g) for g in ["todos"] + GROUP_REPORT}
    params_stats = {}
    for m in ("C", "D", "F", "P"):
        vals = [o["modelos"][m]["ajuste_todo"]["parametros"] for o in frames_out
                if m in o["modelos"] and "ajuste_todo" in o["modelos"][m] and o["modelos"][m]["ajuste_todo"]["n_corr"]]
        if vals:
            V = np.array(vals)
            params_stats[m] = {"frames": len(V), "mediana": np.median(V, 0).tolist(),
                               "p10": np.percentile(V, 10, 0).tolist(), "p90": np.percentile(V, 90, 0).tolist()}
    # ¿la focal del frame explica su parámetro de escala / focal?
    corr_focal = {}
    for m in ("C", "F"):
        pts = [(o["focal"], o["modelos"][m]["ajuste_todo"]["parametros"][0]) for o in frames_out
               if m in o["modelos"] and "ajuste_todo" in o["modelos"][m] and o["modelos"][m]["ajuste_todo"]["n_corr"]]
        if len(pts) > 10:
            a = np.array(pts)
            corr_focal[m] = float(np.corrcoef(a[:, 0], a[:, 1])[0, 1])
    report = {"npz": os.path.abspath(npz), "poses": pose_src, "filtro": filter_path, "parametros": dataclasses.asdict(p),
              "version_poses_medidas": OE.sha1(c2w.astype(np.float32)),
              "version_poses_originales": OE.sha1(np.linalg.inv(E).astype(np.float32)),
              "profundidad": "depth del .npz, sin modificar (observación = frame del .npz con su índice)",
              "profundidad_mediana": med, "frames": len(real), "frames_medidos": len(frames_out),
              "metrica": "residuo punto a plano / profundidad, en píxeles NO usados para ajustar, truncado a "
                         f"{p.test_gate}; mediana por frame y luego mediana entre frames",
              "resumen": summary, "parametros_estimados": params_stats, "correlacion_con_focal": corr_focal,
              "focal": {"mediana": float(np.median(d["intrinsic"][real][:, 0, 0])),
                        "cv": float(np.std(d["intrinsic"][real][:, 0, 0]) / np.mean(d["intrinsic"][real][:, 0, 0]))},
              "tiempo_s": round(time.time() - t0, 1), "observaciones": frames_out}
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out + ".json", "w") as fh:
        json.dump(report, fh, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    return report


def plausible(model, par, med, p):
    """Correcciones físicamente plausibles (si no, el ajuste "mejora" deformando: en la windowed del desnivel, el
    ajuste D del piso llegó a escalas negativas que aplanan el piso a una profundidad constante)."""
    s = p.max_escala
    if model in ("C", "F"):
        return abs(par[0]) <= s
    if model == "D":
        return abs(par[0]) <= s and abs(par[1]) <= s * med
    if model == "E":
        return abs(par[0]) <= s and abs(par[1]) * 0.4 <= s       # |y/z| < 0.4 en estas cámaras
    if model == "P":
        return np.linalg.norm(par[:3]) <= np.radians(5) and np.linalg.norm(par[3:]) <= s * med
    return True


def accept(obs, mode, group, p, med, models=("E", "C")):
    """El primer modelo de `models` cuyo ajuste es identificable, plausible, y mejora los píxeles de prueba de su
    grupo. Devuelve (modelo, parámetros, motivo) o (None, None, motivos)."""
    a = obs["modelos"].get("A", {}).get("ajuste_todo", {}).get("prueba", {}).get(group)
    if a is None or not a.get("mediana"):
        return None, None, "sin píxeles de prueba de ese grupo (sin otro momento con que compararlo)"
    why = []
    for m in models:
        fm = obs["modelos"].get(m, {}).get(mode)
        if fm is None or fm["n_corr"] < p.min_corr or not fm["estable"]:
            why.append(f"{m}: correspondencias insuficientes")
            continue
        if fm.get("colinealidad", 0) > p.max_colinealidad:
            why.append(f"{m}: parámetros indistinguibles ({fm['colinealidad']:.4f})")
            continue
        if max(fm["se"]) > p.max_se:
            why.append(f"{m}: incierto ({max(fm['se']):.4f})")
            continue
        if not plausible(m, fm["parametros"], med, p):
            why.append(f"{m}: implausible {np.round(fm['parametros'], 3).tolist()}")
            continue
        b = fm.get("prueba", {}).get(group)
        if b is None or b["mediana"] > p.min_mejora * a["mediana"]:
            why.append(f"{m}: no mejora los píxeles de prueba")
            continue
        return m, list(fm["parametros"]), f"{m}: {b['mediana'] / a['mediana']:.2f} del residuo en píxeles de prueba"
    return None, None, "; ".join(why)


def write_corrected(report, filter_path, out_npz, p):
    """Profundidad corregida por frame fuera del piso, con el primer modelo aceptado entre E (escala que varía con la
    fila) y C (escala), ajustado SIN el piso contra otros momentos del recorrido. El piso no se corrige por
    profundidad: ningún modelo plausible lo explica (DIAGNOSTICO_PROFUNDIDAD.md); para el piso está la alineación
    del piso (o3d_estable --alinear_piso). Nuevo .npz con todo lo demás igual; el original no se toca. Frames sin
    corrección aceptada quedan como estaban."""
    z = np.load(report["npz"])
    real = np.flatnonzero(z["is_real"])
    flt = np.load(filter_path)
    floor_ids = np.array(OE.GROUPS["piso"])
    depth = z["depth"].astype(np.float32).copy()
    med = report["profundidad_mediana"]
    H, W = depth.shape[1:3]
    rows = np.arange(H, dtype=np.float64)[:, None]
    table = {}
    for o in report["observaciones"]:
        i = o["frame"]
        k = int(np.searchsorted(real, i))
        m, par, why = accept(o, "ajuste_sin_piso", "no_piso", p, med)
        if par is not None:
            K = z["intrinsic"][i].astype(np.float64)
            factor = 1 + par[0] + (par[1] * (rows - K[1, 2]) / K[1, 1] if m == "E" else 0.0)
            factor = np.broadcast_to(factor, (H, W))
            rest = ~np.isin(flt["labels"][k], floor_ids)
            depth[i][rest] = (factor[rest] * depth[i][rest]).astype(np.float32)
        table[i] = {"modelo": m, "parametros": par, "motivo": why}
    meta = {"modelo": "fuera del piso: E (z' = (1 + a + b (v - cy) / fy) z) o, si E no se acepta, C (z' = (1 + s) z); "
                      "ajustados sin el piso contra otros momentos del recorrido; el piso no se corrige",
            "npz_original": report["npz"], "version_poses": report["version_poses_medidas"],
            "criterios": {"min_corr": p.min_corr, "max_colinealidad": p.max_colinealidad, "max_se": p.max_se,
                          "min_mejora": p.min_mejora, "max_escala": p.max_escala},
            "frames_corregidos": {mm: sum(v["modelo"] == mm for v in table.values()) for mm in ("E", "C")},
            "frames": len(real)}
    arrays = {k: z[k] for k in z.files if k != "depth"}
    arrays["depth"] = depth.astype(z["depth"].dtype)
    arrays["correccion_profundidad"] = np.array(json.dumps({"meta": meta, "por_frame": {str(k): v for k, v in table.items()}}))
    np.savez(out_npz, **arrays)
    return meta


def print_summary(r):
    print(f"{r['frames_medidos']} frames medidos de {r['frames']} | focal cv {r['focal']['cv']:.1%} | {r['tiempo_s']} s")
    print("residuo relativo (mediana) en píxeles de prueba:")
    hdr = ["todos"] + GROUP_REPORT
    print(f"  {'modelo/ajuste':24s}" + "".join(f"{g:>10s}" for g in hdr))
    for m, modes in r["resumen"].items():
        for mode, row in modes.items():
            print(f"  {m + ' ' + mode:24s}" + "".join(
                f"{(row[g][0] if row[g][0] is not None else float('nan')):10.4f}" for g in hdr))
    for m, s in r["parametros_estimados"].items():
        print(f"  parámetros {m}: mediana {np.round(s['mediana'], 4).tolist()} p10 {np.round(s['p10'], 4).tolist()} "
              f"p90 {np.round(s['p90'], 4).tolist()}")
    print("  correlación con la focal del frame:", {k: round(v, 3) for k, v in r["correlacion_con_focal"].items()})


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("npz")
    ap.add_argument("--out", required=True)
    ap.add_argument("--filter", default=None)
    ap.add_argument("--poses", default=None, help="<..>_estable.npz: medir con esas poses (misma reconstrucción)")
    ap.add_argument("--corregir", default=None, help="escribir aquí un .npz con la profundidad corregida por frame "
                                                    "fuera del piso (E o C); requiere --filter y --every 1")
    for f in dataclasses.fields(Params):
        if not isinstance(f.default, tuple):
            ap.add_argument("--" + f.name, type=type(f.default), default=f.default)
    a = ap.parse_args()
    p = Params(**{f.name: getattr(a, f.name) for f in dataclasses.fields(Params) if not isinstance(f.default, tuple)})
    r = run(a.npz, a.out, p, a.filter, a.poses)
    print_summary(r)
    if a.corregir:
        if not a.filter or p.every != 1 or a.poses:
            ap.error("--corregir requiere --filter, --every 1 y las poses originales")
        print("corrección:", write_corrected(r, a.filter, a.corregir, p))


if __name__ == "__main__":
    main()
