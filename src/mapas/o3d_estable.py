#!/usr/bin/env python3
"""Estabilización geométrica con Open3D ANTES de la integración TSDF (grafo de poses + filtrado + estructura).

Lee las predicciones de una prueba (eval/<name>.npz: profundidad, confianza, imagen, intrínsecos y pose de
cada frame real) y escribe, sin tocarlas, archivos aparte con poses corregidas y máscaras por píxel que
tsdf_mesh.py --stable integra sobre la profundidad ORIGINAL de cada frame (no se arma una secuencia RGB-D falsa
a partir de una nube).

  1. Validación: profundidad y confianza no finitas o fuera de rango, intrínsecos, poses (rotación ortonormal,
     det +1, finitas), saltos de pose, frames repetidos (static hold), compatibilidad con el filtro geométrico.
     Las unidades son las del modelo: no hay escala métrica.
  2. Nubes por frame, trazables: cada punto guarda su frame y su píxel (grilla de --stride).
  3. Grafo de poses (grafo_poses.py): fragmentos solapados, odometría y cierres verificados por imagen y
     geometría, optimización robusta de Open3D, corrección aceptada solo si es plausible y mejora la
     consistencia. Las poses originales se conservan; las nuevas tienen versión (huella).
  4. Filtrado de la nube fusionada por vóxel (el del TSDF): Statistical y Radius Outlier Removal, configurables.
     Lo descartado vuelve a sus píxeles como máscara. Personas, animales y cielo (semántica del filtro
     geométrico, si se pasa --filter) se quitan por semántica, que no depende de las poses.
  5. Grafo estructural (grafo_estructura.py), solo diagnóstico: planos dominantes como entidades, superficies
     paralelas clasificadas (posible duplicado, paredes paralelas reales, dos caras, ambigua), adyacencias,
     espesor local. Original contra estabilizada.

    python3 src/mapas/o3d_estable.py eval/mapa.npz --out exports/estable/mapa [--filter <..._filtro.npz>]
      -> mapa_estable.npz      poses originales y estabilizadas, máscaras por píxel, metadato
         mapa_estable.json     informe (validación, filtrado, diagnóstico, tiempos)
         mapa_grafo.json       grafo de poses (nodos, aristas con procedencia y estado)
         mapa_estructura_grafo.json   grafo estructural
         mapa_dependencias.json       qué artefactos dependen de qué versión de poses
         mapa_nube_original.ply, mapa_nube_estable.ply (con normales), mapa_duplicadas.ply
"""
import argparse
import dataclasses
import hashlib
import json
import os
import sys
import time
from dataclasses import dataclass

import numpy as np
import open3d as o3d

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import grafo_estructura as GE  # noqa: E402
import grafo_poses as GP  # noqa: E402

PIPELINE_VERSION = "o3d-estable-1"

# Grupos ADE20K (0-based, como geo_filter.py) para medir qué se quita de cada estructura.
GROUPS = {"pared": [0, 22, 27, 100, 144], "puerta": [14, 58], "ventana": [8, 63, 18], "piso": [3, 28, 6, 11, 52],
          "escalera": [53, 59, 121, 38, 95], "techo": [5], "columna": [42]}
GROUP_NAMES = ["otro"] + list(GROUPS)
assert GROUP_NAMES == GE.GROUP_NAMES
REMOVE_DEPTH = [12, 126, 2]          # persona, animal, cielo (las mismas de geo_filter.py)


@dataclass
class Params:
    stride: int = 4                     # un píxel de cada stride x stride
    conf_percentile: float = 35.0       # como tsdf_mesh.py
    depth_max_percentile: float = 97.0  # como tsdf_mesh.py
    holdout_every: int = 0              # frames apartados (no participan del registro ni del filtrado)
    # grafo de poses
    register: bool = True
    frag_len: int = 8                   # frames por fragmento
    frag_step: int = 4                  # avance entre fragmentos (< frag_len: comparten frames)
    kf_trans_rel: float = 0.15          # keyframe: movimiento, x prof. mediana
    kf_rot_deg: float = 15.0
    kf_max_gap: int = 10
    reg_voxel_rel: float = 1 / 60       # vóxel del registro, x prof. mediana
    icp_dist_mult: tuple = (4.0, 2.0, 1.0)        # odometría, x vóxel del registro
    loop_icp_mult: tuple = (12.0, 6.0, 3.0, 1.5)  # cierres: arrancan más lejos (la deriva los separa)
    eval_dist_mult: float = 1.5
    loop_radius_rel: float = 0.6        # cámaras a menos de esto (x prof. mediana): candidato a cierre
    loop_search_vox: float = 8.0
    loop_overlap: float = 0.2
    loops_per_fragment: int = 3
    min_fitness: float = 0.3
    min_normal_spread: float = 0.05
    min_visual_inliers: int = 30
    loop_residual_vox: float = 3.0
    prior_weight: float = 0.05          # peso de la pose relativa original de LingBot frente a la odometría ICP
    loop_preference: float = 2.0         # como el registro de escena del sistema de reconstrucción de Open3D
    tol_trans_vox: float = 0.5          # correcciones por debajo de esto son despreciables
    tol_rot_deg: float = 1.0
    max_trans_rel: float = 0.15         # límite de la corrección de un fragmento, x prof. mediana
    max_loop_trans_rel: float = 0.3     # corrección de un cierre siempre aceptable (x prof. mediana)...
    drift_rate: float = 0.08            # ... o hasta esta fracción de lo recorrido entre los dos fragmentos
    max_loop_rot_deg: float = 15.0
    max_rot_deg: float = 5.0
    odo_residual_vox: float = 3.0       # la solución no puede romper la odometría más que esto (x vóxel)
    odo_residual_deg: float = 1.5
    dof4: bool = False                  # la corrección del grafo solo en giro sobre la vertical y desplazamiento
                                        # horizontal: inclinación y altura quedan como las fijan la gravedad y el piso
    icp_tukey_rel: float = 0.5          # ICP punto a plano robusto: núcleo de Tukey de 0.5 x la distancia de
                                        # correspondencia (0 = sin núcleo, como antes del 2026-10-17)
    dof4_rigidez: float = 100.0         # dof4: peso de inclinación y altura en las aristas de LingBot, x una odometría
    registro: str = "verificado"       # verificado: cierres verificados por imagen y geometría (solve);
                                        # multiway: el tutorial de Open3D (solve_multiway), solo con la poda
    multiway_overlap: float = 0.3       # multiway: pares que comparten esta fracción de celdas del ICP grueso
    multiway_prune: float = 0.25        # multiway: edge_prune_threshold de la optimización global
    # permanencia del piso: cada frame se ajusta en altura e inclinación al piso combinado de todos los frames
    alinear_piso: bool = False
    floor_depth_rel: float = 2.0        # solo piso a menos de esto x prof. mediana (lejos la profundidad es ruidosa)
    floor_min_pts: int = 200            # puntos de piso de un frame para ajustarlo
    floor_min_frames: int = 3           # celdas del piso combinado: vistas por al menos estos frames
    floor_iters: int = 3
    floor_max_tilt_deg: float = 3.0     # corrección máxima (total)
    floor_max_dz_rel: float = 0.3       # x prof. mediana (total)
    floor_smooth: int = 5               # mediana móvil de las correcciones, en frames
    # filtrado
    use_filter_mask: bool = True        # con poses conservadas, sumar la consistencia multivista del filtro (no su
                                        # ajuste a planos): es de las mismas poses; si el grafo las corrige, no vale
    filt_voxel_rel: float = 1 / 150     # = vóxel por defecto de tsdf_mesh.py
    sor: bool = True
    sor_nb: int = 20
    sor_std: float = 2.0
    ror: bool = True
    ror_radius_vox: float = 3.0
    ror_min: int = 4
    # diagnóstico
    normals_radius_vox: float = 3.0
    thick_cell_vox: float = 5.0
    planes_max: int = 40
    plane_dist_vox: float = 1.5
    plane_min_frac: float = 0.003
    dup_max_rel: float = 0.3
    dup_angle_deg: float = 10.0
    dup_min_overlap: float = 0.2
    floor_cell_rel: float = 0.03        # permanencia del piso: celda en planta, x prof. mediana
    floor_visit_frames: int = 16        # permanencia del piso: frames consecutivos que cuentan como una visita
    seed: int = 0


rot_deg = GP.rot_deg
sha1 = GP.pose_hash


def pcd(P, colors=None, normals=None):
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.asarray(P, np.float64)))
    if colors is not None:
        pc.colors = o3d.utility.Vector3dVector(np.asarray(colors, np.float64) / 255.0)
    if normals is not None:
        pc.normals = o3d.utility.Vector3dVector(np.asarray(normals, np.float64))
    return pc


def estimate_up(c2w):
    """Vertical como en geo_filter.py: el eje X de las cámaras es casi siempre horizontal."""
    x = c2w[:, :3, 0]
    _, vecs = np.linalg.eigh(x.T @ x)
    up = vecs[:, 0]
    if up @ c2w[:, :3, 1].mean(0) > 0:
        up = -up
    return up / np.linalg.norm(up)


def load_arrays(npz):
    """Arreglos del .npz cargados una sola vez (cada d[clave] de un NpzFile vuelve a leer y descomprimir)."""
    z = np.load(npz)
    keys = ["depth", "depth_conf", "images", "intrinsic", "extrinsic", "is_real"] + \
        [k for k in ("estatico", "pose_source", "stella_status", "frame_ids", "source_index", "ds") if k in z.files]
    D = {k: z[k] for k in keys}
    D["path"] = os.path.abspath(npz)
    return D


# --- 1. validación ----------------------------------------------------------------------------------------

def validate(d, real, c2w, flt=None):
    """Chequeos por frame. Devuelve (informe, frames inválidos que no deben usarse)."""
    rep, bad = {}, set()
    R = c2w[:, :3, :3]
    orth = np.abs(np.einsum("nji,njk->nik", R, R) - np.eye(3)).max((1, 2))
    det = np.linalg.det(R)
    finite = np.isfinite(c2w).all((1, 2))
    inval = np.flatnonzero(~finite | (orth > 1e-3) | (np.abs(det - 1) > 1e-3))
    bad |= {int(real[i]) for i in inval}
    rep["poses"] = {"invalidas": [int(real[i]) for i in inval], "ortonormalidad_max": float(orth.max()),
                    "det_min": float(det.min()), "det_max": float(det.max())}
    step = np.linalg.norm(np.diff(c2w[:, :3, 3], axis=0), axis=1)
    rstep = np.array([rot_deg(R[i].T @ R[i + 1]) for i in range(len(R) - 1)])
    med_step = float(np.median(step)) if len(step) else 0.0
    mad = float(np.median(np.abs(step - med_step))) if len(step) else 0.0
    jumps = np.flatnonzero(step > med_step + 8 * max(mad, 1e-9) + 1e-6)
    rep["saltos"] = {"paso_mediano": med_step, "saltos_traslacion": [int(real[i + 1]) for i in jumps][:50],
                     "giros_mayores_20deg": int((rstep > 20).sum()),
                     "paso_max": float(step.max()) if len(step) else 0.0,
                     "giro_max_deg": float(rstep.max()) if len(rstep) else 0.0}
    K = d["intrinsic"][real].astype(np.float64)
    H, W = d["depth"].shape[1:3]
    kbad = np.flatnonzero(~np.isfinite(K).all((1, 2)) | (K[:, 0, 0] <= 0) | (K[:, 1, 1] <= 0) |
                          (K[:, 0, 2] < 0) | (K[:, 0, 2] > W) | (K[:, 1, 2] < 0) | (K[:, 1, 2] > H))
    bad |= {int(real[i]) for i in kbad}
    rep["intrinsecos"] = {"invalidos": [int(real[i]) for i in kbad], "fx_mediana": float(np.median(K[:, 0, 0]))}
    sig = [hashlib.sha1(np.ascontiguousarray(d["images"][i][::8, ::8]).tobytes()).hexdigest() for i in real]
    rep["frames_repetidos"] = [int(real[i]) for i in range(1, len(real))
                               if sig[i] == sig[i - 1] and np.abs(c2w[i] - c2w[i - 1]).max() < 1e-7]
    if "estatico" in d:
        rep["frames_estaticos_del_vivo"] = int(np.asarray(d["estatico"])[real].sum())
    if flt is not None:
        same_real = np.array_equal(np.asarray(flt["real"]), real)
        rep["filtro"] = {"mismos_frames": bool(same_real)}
        if same_real:
            fix = flt["c2w_fix"].astype(np.float64)
            dt = np.linalg.norm(fix[:, :3, 3] - c2w[:, :3, 3], axis=1)
            dr = np.array([rot_deg(fix[i, :3, :3].T @ R[i]) for i in range(len(R))])
            rep["filtro"].update({"realineado": bool(flt["aligned"]), "dif_max": float(dt.max()),
                                  "dif_max_deg": float(dr.max())})
    return rep, bad


def depth_stats(d, real, cthr, dmax):
    nonfin = nonpos = far = low = tot = 0
    for i in real:
        z = d["depth"][i].astype(np.float32)
        c = d["depth_conf"][i].astype(np.float32)
        tot += z.size
        nonfin += int((~np.isfinite(z)).sum() + (~np.isfinite(c)).sum())
        nonpos += int((z <= 0).sum())
        far += int((z > dmax).sum())
        low += int((c < cthr).sum())
    return {"pixeles": tot, "no_finitos": nonfin, "no_positivos": nonpos, "mas_lejos_que_el_maximo": far,
            "confianza_baja": low}


# --- 2. nubes por frame -----------------------------------------------------------------------------------

def lift(d, ids, c2w_of, p, cthr, dmax, removed=None, groups_px=None):
    """Puntos de cada frame (mundo) con su frame, su celda de píxel (grilla de stride), su color y su grupo.
    Quedan agrupados por frame, en el orden de `ids`."""
    s = p.stride
    H, W = d["depth"].shape[1:3]
    ys, xs = np.mgrid[s // 2:H:s, s // 2:W:s]
    P, F, YX, C, Gs, Z = [], [], [], [], [], []
    for i in ids:
        z = d["depth"][i][ys, xs].astype(np.float32)
        c = d["depth_conf"][i][ys, xs].astype(np.float32)
        ok = np.isfinite(z) & (z > 0) & (z < dmax) & (c >= cthr)
        if removed is not None:
            ok &= ~removed[i][ys, xs]
        K = d["intrinsic"][i].astype(np.float64)
        zz = z[ok].astype(np.float64)
        pc = np.stack([(xs[ok] - K[0, 2]) / K[0, 0] * zz, (ys[ok] - K[1, 2]) / K[1, 1] * zz, zz], 1)
        T = c2w_of[i]
        P.append((pc @ T[:3, :3].T + T[:3, 3]).astype(np.float32))
        F.append(np.full(ok.sum(), i, np.int32))
        YX.append(np.stack([ys[ok] // s, xs[ok] // s], 1).astype(np.int16))
        C.append(d["images"][i][ys, xs][ok])
        Z.append(z[ok])
        if groups_px is not None:
            Gs.append(groups_px[i][ys, xs][ok])
    return (np.concatenate(P), np.concatenate(F), np.concatenate(YX), np.concatenate(C),
            np.concatenate(Gs) if groups_px is not None else None, np.concatenate(Z))


def frame_slices(F):
    """Tramos [a, b) de cada frame en un arreglo de puntos agrupado por frame."""
    u, start = np.unique(F, return_index=True)
    order = np.argsort(start)
    u, start = u[order], start[order]
    end = np.r_[start[1:], len(F)]
    return {int(i): (int(a), int(b)) for i, a, b in zip(u, start, end)}


def fuse(P, F, voxel, groups=None, colors=None, Z=None):
    """Vóxeles: centro, puntos, frames distintos, primer frame, frame medio, color medio y grupo mayoritario.
    `inv` lleva cada punto a su vóxel."""
    origin = P.min(0)
    ijk = np.floor((P - origin) / voxel).astype(np.int64)
    key = (ijk[:, 0] << 42) | (ijk[:, 1] << 21) | ijk[:, 2]
    uk, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
    inv = inv.ravel()
    M = len(uk)
    cen = np.stack([np.bincount(inv, P[:, k].astype(np.float64), M) for k in range(3)], 1) / cnt[:, None]
    fmin = np.full(M, np.iinfo(np.int32).max, np.int64)
    np.minimum.at(fmin, inv, F)
    fmean = np.bincount(inv, F.astype(np.float64), M) / cnt
    nf = int(F.max()) + 1
    nfr = np.bincount(np.unique(inv.astype(np.int64) * nf + F) // nf, minlength=M)
    out = {"centers": cen, "count": cnt, "inv": inv, "fmin": fmin, "fmean": fmean, "nframes": nfr, "voxel": voxel}
    if Z is not None:
        out["zmean"] = np.bincount(inv, Z.astype(np.float64), M) / cnt      # profundidad media de observación
    if colors is not None:
        out["color"] = np.stack([np.bincount(inv, colors[:, k].astype(np.float64), M) for k in range(3)], 1) / cnt[:, None]
    if groups is not None:
        ng = len(GROUP_NAMES)
        votes = np.bincount(inv.astype(np.int64) * ng + groups, minlength=M * ng).reshape(M, ng)
        out["group"] = votes.argmax(1)
    return out


PER_VOXEL = ("centers", "count", "fmin", "fmean", "nframes", "color", "group", "zmean")


def subset(V, keep_vox, keep_pts):
    """Vóxeles conservados, con `inv` reindexado para los puntos conservados."""
    W = {k: (v[keep_vox] if k in PER_VOXEL else v) for k, v in V.items()}
    remap = -np.ones(len(keep_vox), np.int64)
    remap[keep_vox] = np.arange(keep_vox.sum())
    W["inv"] = remap[V["inv"][keep_pts]]
    return W


# --- 4. filtrado ------------------------------------------------------------------------------------------

def filter_voxels(V, p, fx):
    """SOR y ROR sobre los centros de vóxel, relativos a la densidad esperada de cada vóxel: una superficie lejana
    se muestrea más rala (separación ~ profundidad x stride / fx), y un umbral único la tomaría por atípica.
    Devuelve la máscara de vóxeles conservados y cuántos quitó cada filtro."""
    from scipy.spatial import cKDTree

    X = V["centers"]
    keep = np.ones(len(X), bool)
    spacing = np.maximum(V["voxel"], V["zmean"] * p.stride / fx) if "zmean" in V else np.full(len(X), V["voxel"])
    stats = {"voxeles": int(len(keep)), "separacion_esperada_mediana": float(np.median(spacing)),
             "criterio": "relativo a la separación esperada de cada vóxel (profundidad x stride / fx)"}
    tree = cKDTree(X)
    if p.sor:
        dist, _ = tree.query(X, k=p.sor_nb + 1, workers=-1)
        u = dist[:, 1:].mean(1) / spacing
        m = u <= u.mean() + p.sor_std * u.std()
        stats["sor_quitados"] = int((keep & ~m).sum())
        keep &= m
    if p.ror:
        cnt = tree.query_ball_point(X, r=p.ror_radius_vox * spacing, return_length=True, workers=-1) - 1
        m = cnt >= p.ror_min
        stats["ror_quitados"] = int((keep & ~m).sum())
        keep &= m
    stats["conservados"] = int(keep.sum())
    return keep, stats


# --- 5. diagnóstico ---------------------------------------------------------------------------------------

def local_thickness(P, cell, min_pts=30):
    """Espesor local: en celdas de `cell`, dispersión de los puntos a lo largo de su normal (p10-p90 de una
    normal = 2.56 desvíos) en las celdas planas. Una superficie bien registrada es una lámina."""
    if len(P) == 0:
        return {"celdas": 0}
    ijk = np.floor((P - P.min(0)) / cell).astype(np.int64)
    key = (ijk[:, 0] << 42) | (ijk[:, 1] << 21) | ijk[:, 2]
    _, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
    inv = inv.ravel()
    M = len(cnt)
    X = P.astype(np.float64)
    mu = np.stack([np.bincount(inv, X[:, k], M) for k in range(3)], 1) / cnt[:, None]
    S = np.empty((M, 3, 3))
    for a in range(3):
        for b in range(a, 3):
            S[:, a, b] = S[:, b, a] = np.bincount(inv, X[:, a] * X[:, b], M) / cnt - mu[:, a] * mu[:, b]
    big = cnt >= min_pts
    ev = np.linalg.eigvalsh(S[big])
    planar = ev[:, 1] > 10 * np.maximum(ev[:, 0], 1e-18)
    th = 2.563 * np.sqrt(np.maximum(ev[planar, 0], 0))
    if len(th) == 0:
        return {"celdas": int(big.sum()), "celdas_planas": 0}
    return {"celdas": int(big.sum()), "celdas_planas": int(planar.sum()),
            "espesor_mediano": float(np.median(th)), "espesor_p90": float(np.percentile(th, 90)),
            "frac_gruesas": float((th > 0.4 * cell).mean())}


def _runs(*keys):
    """Inicio y fin de los tramos de claves iguales en arreglos ya ordenados."""
    change = np.zeros(len(keys[0]) - 1, bool)
    for k in keys:
        change |= np.diff(k) != 0
    brk = np.flatnonzero(change) + 1
    return np.r_[0, brk], np.r_[brk, len(keys[0])]


def plan_basis(up):
    """Vertical y dos ejes horizontales (e1, e2, u: dextrógiro)."""
    u = up / np.linalg.norm(up)
    e1 = np.cross(u, [1.0, 0, 0] if abs(u[0]) < 0.9 else [0, 1.0, 0])
    e1 /= np.linalg.norm(e1)
    return u, e1, np.cross(u, e1)


def align_floor(P, F, Z, G_pt, up, real, med, p):
    """Permanencia del piso: el registro multivía restringido al piso. Se combina el piso de todos los frames (en
    cada celda en planta, la mediana de las alturas medianas de los frames que la ven) y cada frame se ajusta a
    ese piso combinado en altura e inclinación: los 3 grados de libertad que el piso fija. Lo que no fija
    (desplazamiento en planta, giro alrededor de la vertical) no se toca. Mínimos cuadrados recortados (piso de
    otro nivel o mal etiquetado), correcciones acotadas y suavizadas en el tiempo, iterado. Los frames sin piso
    toman la corrección interpolada de sus vecinos en el recorrido.
    Devuelve (corrección 4x4 por frame de `real` o None, estadísticas)."""
    if G_pt is None:
        return None, {"nota": "sin la semántica del filtro no se distingue el piso"}
    m = (G_pt == GROUP_NAMES.index("piso")) & (Z <= p.floor_depth_rel * med)
    if m.sum() < p.floor_min_pts:
        return None, {"nota": "sin piso suficiente", "puntos_piso": int(m.sum())}
    X = P[m].astype(np.float64)
    f = F[m]
    u, e1, e2 = plan_basis(up)
    cell = p.floor_cell_rel * med
    pos_real = {int(i): k for k, i in enumerate(real)}
    max_t = np.tan(np.radians(p.floor_max_tilt_deg)) / p.floor_iters
    max_dz = p.floor_max_dz_rel * med / p.floor_iters
    of = np.argsort(f, kind="stable")
    stf, enf = _runs(f[of])
    total, hist, dz_tot = {}, [], {}
    for it in range(p.floor_iters):
        h, x, y = X @ u, X @ e1, X @ e2
        ij = np.floor(np.stack([x, y], 1) / cell).astype(np.int64) + (1 << 30)
        ck = (ij[:, 0] << 31) | ij[:, 1]
        o = np.lexsort((h, f, ck))
        st, en = _runs(ck[o], f[o])
        hm, cc = h[o][(st + en - 1) // 2], ck[o][st]              # altura mediana de cada frame en cada celda
        o2 = np.lexsort((hm, cc))
        st2, en2 = _runs(cc[o2])
        ok = (en2 - st2) >= p.floor_min_frames
        cells, H = cc[o2][st2][ok], hm[o2][(st2 + en2 - 1) // 2][ok]
        if not len(cells):
            break
        j = np.minimum(np.searchsorted(cells, ck), len(cells) - 1)
        inside = cells[j] == ck
        r = H[j] - h
        params = {}
        for a_, b_ in zip(stf, enf):
            idx = of[a_:b_]
            idx = idx[inside[idx]]
            if len(idx) < p.floor_min_pts:
                continue
            xc, yc, hc = x[idx].mean(), y[idx].mean(), h[idx].mean()
            A = np.c_[np.ones(len(idx)), x[idx] - xc, y[idx] - yc]
            b = r[idx]
            keep = np.ones(len(idx), bool)
            for _ in range(3):
                sol = np.linalg.lstsq(A[keep], b[keep], rcond=None)[0]
                res = b - A @ sol
                mad = 1.4826 * np.median(np.abs(res[keep] - np.median(res[keep]))) + 1e-12
                keep = np.abs(res - np.median(res[keep])) <= 3 * mad
                if keep.sum() < p.floor_min_pts // 2:
                    break
            params[int(f[idx[0]])] = (sol, xc, yc, hc)
        if not params:
            break
        fr = sorted(params, key=lambda i: pos_real[i])
        S = np.array([params[i][0] for i in fr])
        if len(fr) >= p.floor_smooth:
            from scipy.ndimage import median_filter
            S = median_filter(S, size=(p.floor_smooth, 1), mode="nearest")
        S[:, 0] = np.clip(S[:, 0], -max_dz, max_dz)
        S[:, 1:] = np.clip(S[:, 1:], -max_t, max_t)
        step, dzs = {}, {}
        for i, (dz, a, b) in zip(fr, S):
            dzs[i] = float(dz)
            _, xc, yc, hc = params[i]
            R = GP.Rotation.from_rotvec(e1 * np.arctan(b) - e2 * np.arctan(a)).as_matrix()
            c = xc * e1 + yc * e2 + hc * u
            T = np.eye(4)
            T[:3, :3], T[:3, 3] = R, c - R @ c + dz * u
            step[i] = T
        for a_, b_ in zip(stf, enf):
            i = int(f[of[a_]])
            if i in step:
                idx = of[a_:b_]
                X[idx] = X[idx] @ step[i][:3, :3].T + step[i][:3, 3]
                total[i] = step[i] @ total.get(i, np.eye(4))
                dz_tot[i] = dz_tot.get(i, 0.0) + dzs[i]
        hist.append({"iteracion": it + 1, "frames_ajustados": len(step), "celdas_piso_combinado": int(len(cells)),
                     "residuo_mediano_antes": float(np.median(np.abs(r[inside])))})
    if not total:
        return None, {"nota": "ningún frame con piso suficiente", "iteraciones": hist}
    have = sorted(total, key=lambda i: pos_real[i])
    tq = np.array([pos_real[i] for i in have], float)
    q = np.clip(np.arange(len(real), dtype=float), tq[0], tq[-1])
    if len(have) > 1:
        Rs = GP.Slerp(tq, GP.Rotation.from_matrix(np.array([total[i][:3, :3] for i in have])))(q).as_matrix()
        ts = np.stack([np.interp(q, tq, [total[i][k, 3] for i in have]) for k in range(3)], 1)
    else:
        Rs, ts = np.repeat(total[have[0]][None, :3, :3], len(real), 0), np.repeat(total[have[0]][None, :3, 3], len(real), 0)
    corr = {}
    for k, i in enumerate(real.tolist()):
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = Rs[k], ts[k]
        corr[int(i)] = total.get(int(i), T)
    dz = [abs(v) for v in dz_tot.values()]                     # cambio de altura en el centro del piso del frame
    return corr, {"frames_con_piso": len(total), "frames_interpolados": int(len(real) - len(total)),
                  "puntos_piso": int(m.sum()), "iteraciones": hist, "dz_max": max(dz), "dz_max_rel": max(dz) / med,
                  "inclinacion_max_deg": float(max(rot_deg(T[:3, :3]) for T in total.values()))}


def floor_permanence(P, pos, G_pt, up, cell, visit_frames, min_pts=10):
    """Permanencia espacial del piso: un mismo lugar del piso visto en momentos distintos debe quedar a la misma
    altura. En planta, celdas de `cell`; en cada una, la altura mediana del piso por visita (bloques de
    `visit_frames` frames consecutivos). Mide:
    - la diferencia entre visitas: lo que una deriva separa en capas;
    - el espesor del piso dentro de cada celda: p10-p90 de la altura, todas las visitas juntas.
    Solo los puntos del grupo 'piso' del filtro (las escaleras no cuentan)."""
    if G_pt is None:
        return {"celdas": 0, "nota": "sin la semántica del filtro no se distingue el piso"}
    m = G_pt == GROUP_NAMES.index("piso")
    if m.sum() < 100:
        return {"celdas": 0, "puntos_piso": int(m.sum())}
    X = P[m].astype(np.float64)
    u, e1, e2 = plan_basis(up)
    h = X @ u
    ij = np.floor(np.stack([X @ e1, X @ e2], 1) / cell).astype(np.int64) + (1 << 30)
    ck = (ij[:, 0] << 31) | ij[:, 1]
    vis = pos[m] // visit_frames
    out = {"puntos_piso": int(m.sum()), "celda": cell, "visita_frames": visit_frames}
    # espesor del piso por celda
    o = np.lexsort((h, ck))
    st, en = _runs(ck[o])
    n = en - st
    big = n >= 2 * min_pts
    hs = h[o]
    th = hs[st + ((n - 1) * 0.9).astype(int)] - hs[st + ((n - 1) * 0.1).astype(int)]
    th = th[big]
    out["celdas"] = int(big.sum())
    if len(th):
        out["espesor_p10_p90"] = {"mediana": float(np.median(th)), "p90": float(np.percentile(th, 90))}
    # diferencia entre visitas
    o = np.lexsort((h, vis, ck))
    st, en = _runs(ck[o], vis[o])
    ok = (en - st) >= min_pts
    med = h[o][(st + en - 1) // 2][ok]
    cell_v = ck[o][st][ok]
    if len(cell_v):
        st2, en2 = _runs(cell_v)
        rng = np.maximum.reduceat(med, st2) - np.minimum.reduceat(med, st2)
        multi = (en2 - st2) >= 2
        out["celdas_varias_visitas"] = int(multi.sum())
        if multi.any():
            r = rng[multi]
            out["diferencia_entre_visitas"] = {"mediana": float(np.median(r)), "p90": float(np.percentile(r, 90)),
                                               "max": float(r.max())}
            out["frac_celdas_separadas"] = float((r > 0.5 * cell).mean())
            out["tolerancia"] = 0.5 * cell
    return out


def orient_to_cameras(centers, normals, cam_of_voxel):
    flip = np.einsum("ij,ij->i", normals, cam_of_voxel - centers) < 0
    normals[flip] *= -1
    return normals


def structure_diagnosis(V, F, cams, up, med, p, n_frames_total, session, pose_version, fragments):
    sub = None
    if len(V["centers"]) > 1_500_000:
        sub = np.random.default_rng(p.seed).choice(len(V["centers"]), 1_500_000, replace=False)
    planes = GE.dominant_planes(V, cams, up, p, sub)
    frames = [GE.plane_frames(V, F, pl, n_frames_total) for pl in planes]
    pairs = GE.classify_pairs(V, F, planes, med, p, n_frames_total, frames)
    stacks = GE.stacks_of(pairs, len(planes))
    graph = GE.build(V, F, planes, pairs, med, up, session, pose_version, fragments, n_frames_total)
    counts, pcounts = {}, {}
    for st in stacks:
        counts[st["clase"]] = counts.get(st["clase"], 0) + 1
    for q in pairs:
        pcounts[q["clase"]] = pcounts.get(q["clase"], 0) + 1
    dup = [q for q in pairs if q["clase"] == "duplicada"]
    summary = {"planos": len(planes), "pares": len(pairs), "pares_por_clase": pcounts, "superficies": len(stacks),
               "por_clase": counts, "pilas": stacks,
               "duplicadas_separacion_rel_mediana": float(np.median([q["separacion_rel"] for q in dup])) if dup else None}
    return planes, pairs, summary, graph


# --- principal --------------------------------------------------------------------------------------------

def run(npz, out, p, filter_path=None, log=print):
    t0 = time.time()
    times = {}
    d = load_arrays(npz) if isinstance(npz, str) else npz
    npz = d["path"]
    real = np.flatnonzero(d["is_real"])
    sample = d["depth"][real[::10]].astype(np.float32)
    med = float(np.median(sample[sample > 0]))
    dmax = float(np.percentile(sample[sample > 0], p.depth_max_percentile))
    cthr = float(np.percentile(d["depth_conf"][real[::10]].astype(np.float32), p.conf_percentile))
    E = np.tile(np.eye(4), (len(real), 1, 1))
    E[:, :3, :4] = d["extrinsic"][real]
    c2w = np.linalg.inv(E)
    c2w_of = {int(i): c2w[k] for k, i in enumerate(real)}
    pos_of = {int(i): k for k, i in enumerate(real)}
    session = GP.session_id(npz, d["extrinsic"][real])
    fids = GP.frame_ids_of(d, real)
    pose_in = sha1(c2w.astype(np.float32))
    flt = np.load(filter_path) if filter_path else None
    report = {"version": PIPELINE_VERSION, "npz": npz, "filtro": filter_path, "parametros": dataclasses.asdict(p),
              "sesion": {"id": session, "frames_reales": int(len(real)),
                         "metodo": "windowed" if npz.endswith("_windowed.npz") else
                         ("streaming (vivo)" if "pose_source" in d else "offline"),
                         "poses": "BASIC (extrinsic del .npz)", "version_poses_entrada": pose_in,
                         "marco": "mundo de LingBot-Map, cámaras OpenCV (x der, y abajo, z adelante), c2w = inv(extrinsic)",
                         "unidades": "unidades del modelo (sin escala métrica: monocular, no calibrada)",
                         "profundidad_mediana": med, "profundidad_max_usada": dmax, "confianza_min": cthr}}
    val, bad = validate(d, real, c2w, flt)
    val["profundidad"] = depth_stats(d, real, cthr, dmax)
    report["validacion"] = val
    if flt is not None and not val["filtro"]["mismos_frames"]:
        raise SystemExit(f"{filter_path} es de otra versión de {npz}: no tiene los mismos frames")
    if flt is not None and not bool(flt["aligned"]) and val["filtro"].get("dif_max", 0) > 0.01 * med:
        raise SystemExit(f"{filter_path} se calculó con otras poses (difieren {val['filtro']['dif_max']:.4f})")
    times["validacion"] = time.time() - t0
    log(f"{session}: {len(real)} frames | prof. mediana {med:.4f} | inválidos {len(bad)} | repetidos "
        f"{len(val['frames_repetidos'])} | saltos {len(val['saltos']['saltos_traslacion'])}", flush=True)

    removed = groups_px = None
    if flt is not None:
        lut = np.zeros(256, bool)
        lut[REMOVE_DEPTH] = True
        glut = np.zeros(256, np.int64)
        for g, ids_ in enumerate(GROUPS.values(), 1):
            glut[ids_] = g
        labels = flt["labels"]
        removed = {int(i): lut[labels[k]] for k, i in enumerate(real)}
        groups_px = {int(i): glut[labels[k]] for k, i in enumerate(real)}
    test = set(real[::p.holdout_every].tolist()) if p.holdout_every > 0 else set()
    ids = [int(i) for i in real if int(i) not in bad and int(i) not in test]

    # 2. nubes por frame (poses originales)
    t1 = time.time()
    P0, F, YX, C, G_pt, Z = lift(d, ids, c2w_of, p, cthr, dmax, removed, groups_px)
    sl = frame_slices(F)
    times["nubes"] = time.time() - t1
    log(f"nubes por frame: {len(P0) / 1e6:.2f} M puntos de {len(ids)} frames (stride {p.stride})", flush=True)

    # 3. grafo de poses
    t1 = time.time()
    reg_voxel = med * p.reg_voxel_rel
    corr = {int(i): np.eye(4) for i in real}
    graph = GP.PoseGraph(session, med, reg_voxel, p)
    graph.up = estimate_up(np.array([c2w_of[i] for i in ids])) if p.dof4 and len(ids) else None
    if p.register and len(ids) >= 2 * p.frag_len:
        lost = np.asarray(d["stella_status"])[ids] == 2 if "stella_status" in d else None
        kf, _ = GP.select_keyframes(np.array([c2w_of[i] for i in ids]), [d["images"][i] for i in ids], med,
                                    p.kf_trans_rel, p.kf_rot_deg, p.kf_max_gap, lost=lost)
        kf_real = np.zeros(len(real), bool)
        kf_real[[pos_of[i] for i in ids]] = kf
        frags = GP.make_fragments(ids, p.frag_len, p.frag_step)
        graph.add_fragments(frags, fids, pos_of, c2w_of, P0, sl, kf_real)
    if p.register and len(ids) >= 2 * p.frag_len and p.registro == "multiway":
        G = graph.solve_multiway(p.multiway_overlap, p.multiway_prune, log)
        pf = GP.per_frame_correction(graph.nodes, G, ids)
        idpos_arr = np.array([pos_of[i] for i in ids], float)
        for i in real.tolist():
            if i in pf:
                corr[i] = pf[i]
            else:
                j = int(np.clip(np.searchsorted(idpos_arr, pos_of[i]), 0, len(ids) - 1))
                corr[i] = pf[ids[j]]
        graph.stats["keyframes"] = int(kf.sum())
    elif p.register and len(ids) >= 2 * p.frag_len:
        near = graph.measure_odometry()
        def lift_px(i, uv):
            """Píxeles (x, y) de un keyframe -> puntos del mundo con su profundidad (y si son válidos)."""
            x = np.clip(np.round(uv[:, 0]).astype(int), 0, d["depth"].shape[2] - 1)
            y = np.clip(np.round(uv[:, 1]).astype(int), 0, d["depth"].shape[1] - 1)
            z = d["depth"][i][y, x].astype(np.float64)
            # Sin el corte de profundidad máxima del TSDF: lo que dos visitas ven en común suele estar lejos.
            ok = np.isfinite(z) & (z > 0) & (d["depth_conf"][i][y, x].astype(np.float32) >= cthr)
            K = d["intrinsic"][i].astype(np.float64)
            pc = np.stack([(x - K[0, 2]) / K[0, 0] * z, (y - K[1, 2]) / K[1, 1] * z, z], 1)
            T = c2w_of[i]
            return pc @ T[:3, :3].T + T[:3, 3], ok, z

        graph.search_loops({i: d["images"][i] for i in ids}, near, lift_px)
        G = graph.solve()
        pf = GP.per_frame_correction(graph.nodes, G, ids)
        idpos_arr = np.array([pos_of[i] for i in ids], float)
        for i in real.tolist():
            if i in pf:
                corr[i] = pf[i]
            else:                         # apartados o inválidos: la corrección de su posición en el recorrido
                j = int(np.clip(np.searchsorted(idpos_arr, pos_of[i]), 0, len(ids) - 1))
                corr[i] = pf[ids[j]]
        graph.stats["keyframes"] = int(kf.sum())
    else:
        graph.decision = "registro desactivado o demasiado pocos frames: se conservan las poses"
    times["grafo_poses"] = time.time() - t1
    gj = graph.to_json()
    log(f"grafo de poses: {len(graph.nodes)} nodos, {len(graph.edges)} aristas {gj['conteo_aristas']} -> "
        f"{graph.decision} ({times['grafo_poses']:.0f} s)", flush=True)

    def corrected(corr):
        """Los puntos por frame con la corrección de cada frame."""
        if all(np.allclose(T, np.eye(4)) for T in corr.values()):
            return P0
        out_ = np.empty_like(P0)
        for i, (a_, b_) in sl.items():
            T = corr[i]
            out_[a_:b_] = (P0[a_:b_].astype(np.float64) @ T[:3, :3].T + T[:3, 3]).astype(np.float32)
        return out_

    P1 = corrected(corr)
    floor_report = None
    if p.alinear_piso:                        # permanencia del piso: cada frame al piso combinado
        t1 = time.time()
        up_ = estimate_up(c2w)
        posF = np.searchsorted(real, F)
        floor_report = {"tras_grafo": floor_permanence(P1, posF, G_pt, up_, p.floor_cell_rel * med,
                                                       p.floor_visit_frames)}
        fc, floor_report["alineacion"] = align_floor(P1, F, Z, G_pt, up_, real, med, p)
        if fc is not None:
            corr = {int(i): fc[int(i)] @ corr[int(i)] for i in real}
            P1 = corrected(corr)
        times["piso"] = time.time() - t1
        al = floor_report["alineacion"]
        log(f"alineación del piso: {al.get('frames_con_piso', 0)} frames con piso, corrección máx. "
            f"{al.get('dz_max_rel', 0):.3f} x prof. mediana y {al.get('inclinacion_max_deg', 0):.2f}° "
            f"({times['piso']:.0f} s)", flush=True)
    c2w_new = np.array([corr[int(i)] @ c2w_of[int(i)] for i in real])
    pose_out = sha1(c2w_new.astype(np.float32))
    moved = [(float(np.linalg.norm(c2w_new[k, :3, 3] - c2w[k, :3, 3])), rot_deg(corr[int(i)][:3, :3]))
             for k, i in enumerate(real)]
    report["grafo_poses"] = {"decision": graph.decision, "nodos": len(graph.nodes), "aristas": len(graph.edges),
                             "conteo_aristas": gj["conteo_aristas"], "estadisticas": graph.stats,
                             "version_poses_salida": pose_out}
    report["correccion_por_frame"] = {"camara_desplazamiento_max": max(x for x, _ in moved),
                                      "camara_desplazamiento_mediano": float(np.median([x for x, _ in moved])),
                                      "giro_max_deg": max(y for _, y in moved)}

    # 4. filtrado
    t1 = time.time()
    fvox = med * p.filt_voxel_rel
    V1 = fuse(P1, F, fvox, G_pt, C, Z)
    vkeep, fstats = filter_voxels(V1, p, float(np.median(d["intrinsic"][real][:, 0, 0])))
    pkeep = vkeep[V1["inv"]]
    fstats.update({"puntos": int(len(pkeep)), "puntos_quitados": int((~pkeep).sum()),
                   "frac_quitada": float((~pkeep).mean())})
    if G_pt is not None:
        fstats["frac_quitada_por_grupo"] = {GROUP_NAMES[g]: float((~pkeep[G_pt == g]).mean())
                                            for g in range(len(GROUP_NAMES)) if (G_pt == g).any()}
        fstats["semantica_quitada_px"] = int(sum(v.sum() for v in removed.values()))
    times["filtrado"] = time.time() - t1
    report["filtrado"] = fstats
    log(f"filtrado: {fstats['frac_quitada']:.2%} de los puntos ({times['filtrado']:.0f} s)", flush=True)

    s = p.stride
    H, W = d["depth"].shape[1:3]
    keep = np.ones((len(real), len(range(s // 2, H, s)), len(range(s // 2, W, s))), bool)
    rm = ~pkeep
    if rm.any():
        lut = np.zeros(int(real.max()) + 1, np.int64)
        lut[real] = np.arange(len(real))
        keep[lut[F[rm]], YX[rm, 0], YX[rm, 1]] = False
    if removed is not None:
        for i, mk in removed.items():
            keep[pos_of[i]] &= ~mk[s // 2:H:s, s // 2:W:s]

    # 5. grafo estructural y diagnóstico: original (A) contra estabilizada (B)
    t1 = time.time()
    up = estimate_up(c2w)
    cams = np.zeros((int(real.max()) + 1, 3))
    cams[real] = c2w_new[:, :3, 3]
    cams0 = np.zeros_like(cams)
    cams0[real] = c2w[:, :3, 3]
    V0 = fuse(P0, F, fvox, G_pt, C, Z)
    V1k = subset(V1, vkeep, pkeep)
    F1 = F[pkeep]
    frag_ref = [{"id": n["id"], "frames": n["frames"]} for n in graph.nodes]
    diag, struct_graph, planes_e = {}, None, None
    for name, VV, FF, cc, PP, pv in (("original", V0, F, cams0, P0, pose_in),
                                     ("estable", V1k, F1, cams, P1[pkeep], pose_out)):
        planes, pairs, summ, sg = structure_diagnosis(VV, FF, cc, up, med, p, len(real), session, pv, frag_ref)
        summ["espesor_local"] = local_thickness(PP, p.thick_cell_vox * fvox)
        summ["espesor_local_rel"] = {k: v / med for k, v in summ["espesor_local"].items() if k.startswith("espesor")}
        summ["estados_estructura"] = sg["estados"]
        diag[name] = {"resumen": summ, "pares": pairs}
        if name == "estable":
            struct_graph, planes_e = sg, planes
    # Permanencia del piso, con los mismos puntos en las dos versiones (sin el filtrado, que no es de las poses).
    posF = np.searchsorted(real, F)
    for name, PP in (("original", P0), ("estable", P1)):
        diag[name]["resumen"]["permanencia_piso"] = floor_permanence(PP, posF, G_pt, up, p.floor_cell_rel * med,
                                                                     p.floor_visit_frames)
    times["diagnostico"] = time.time() - t1
    log("superficies paralelas (original -> estable): "
        f"{diag['original']['resumen']['por_clase']} -> {diag['estable']['resumen']['por_clase']}", flush=True)
    fa, fb = (diag[k]["resumen"]["permanencia_piso"] for k in ("original", "estable"))
    if "diferencia_entre_visitas" in fa and "diferencia_entre_visitas" in fb:
        log(f"permanencia del piso (original -> estable, x prof. mediana): diferencia entre visitas p90 "
            f"{fa['diferencia_entre_visitas']['p90'] / med:.4f} -> {fb['diferencia_entre_visitas']['p90'] / med:.4f}, "
            f"celdas separadas {fa['frac_celdas_separadas']:.1%} -> {fb['frac_celdas_separadas']:.1%}, espesor "
            f"mediano {fa['espesor_p10_p90']['mediana'] / med:.4f} -> {fb['espesor_p10_p90']['mediana'] / med:.4f}",
            flush=True)

    # salidas
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    outputs = {"poses_y_mascaras": out + "_estable.npz", "informe": out + "_estable.json",
               "grafo_poses": out + "_grafo.json", "grafo_estructura": out + "_estructura_grafo.json",
               "nube_original": out + "_nube_original.ply", "nube_estable": out + "_nube_estable.ply"}
    if p.registro == "multiway" and graph.nodes:
        # Nube combinada como en el tutorial multiway: cada fragmento transformado por la pose de su nodo,
        # sumados y submuestreados. Los fragmentos comparten frames: cada frame va en el de centro más cercano.
        centers = np.array([nd["centro"] for nd in graph.nodes])
        comb = o3d.geometry.PointCloud()
        for k, nd in enumerate(graph.nodes):
            own = [i for i in nd["frames"] if i in sl and int(np.argmin(np.abs(centers - pos_of[i]))) == k]
            if own:
                idx = np.concatenate([np.arange(*sl[i]) for i in own])
                comb += pcd(P0[idx], C[idx]).transform(nd["pose_optimizada"])
        outputs["nube_combinada"] = out + "_combinada.ply"
        o3d.io.write_point_cloud(outputs["nube_combinada"], comb.voxel_down_sample(fvox))
    # Máscara de consistencia multivista del filtro geométrico: solo si las poses no cambiaron (se calculó con
    # ellas). tsdf_mesh.py --stable la suma a resolución completa.
    fmask, fmask_why = None, None
    if flt is not None and p.use_filter_mask:
        if pose_out != pose_in:
            fmask_why = "no se usa: el grafo cambió las poses y la consistencia del filtro se calculó con las originales"
        elif bool(flt["aligned"]):
            fmask_why = "no se usa: el filtro realineó las poses (--align)"
        else:
            fmask = os.path.abspath(filter_path)
            fmask_why = "se usa: mismas poses que el filtro"
            report["filtrado"]["mascara_filtro_frac_quitada"] = float(1 - flt["keep"].mean())
    report["filtrado"]["mascara_filtro"] = fmask_why
    mask_hash = sha1(np.packbits(keep)) if fmask is None else \
        GP.hashlib.sha1(np.packbits(keep).tobytes() + np.packbits(flt["keep"]).tobytes()).hexdigest()[:16]
    meta = {"version": PIPELINE_VERSION, "sesion": session, "npz": npz, "version_poses_entrada": pose_in,
            "version_poses": pose_out, "huella_mascara": mask_hash, "stride": s, "mascara_filtro": fmask,
            "frames_apartados": sorted(test), "frame_ids": fids.tolist(),
            "tipo_geometria": "observada, registrada y filtrada (Open3D); sin geometría inferida",
            "unidades": report["sesion"]["unidades"], "marco": report["sesion"]["marco"],
            "parametros": dataclasses.asdict(p)}
    np.savez_compressed(outputs["poses_y_mascaras"], real=real, frame_ids=fids, c2w_original=c2w.astype(np.float32),
                        c2w=c2w_new.astype(np.float32), keep=keep, stride=np.array(s), meta=np.array(json.dumps(meta)))
    o3d.io.write_point_cloud(outputs["nube_original"], pcd(V0["centers"], V0["color"]))
    pc1 = pcd(V1k["centers"], V1k["color"])
    pc1.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=p.normals_radius_vox * fvox, max_nn=30))
    pc1.normals = o3d.utility.Vector3dVector(orient_to_cameras(V1k["centers"], np.asarray(pc1.normals).copy(),
                                                               cams[V1k["fmin"]]))
    o3d.io.write_point_cloud(outputs["nube_estable"], pc1)
    dup_pts, dup_col, seen = [], [], set()
    for q in diag["estable"]["pares"]:
        if q["clase"] == "duplicada":
            for k, colr in ((q["a"], (230, 40, 40)), (q["b"], (40, 40, 230))):
                if k in seen:                       # un plano puede estar en varios pares: se escribe una vez
                    continue
                seen.add(k)
                pts = V1k["centers"][planes_e[k]["members"]]
                dup_pts.append(pts)
                dup_col.append(np.tile(colr, (len(pts), 1)))
    if dup_pts:
        outputs["duplicadas"] = out + "_duplicadas.ply"
        o3d.io.write_point_cloud(outputs["duplicadas"], pcd(np.concatenate(dup_pts), np.concatenate(dup_col)))
    # Cámaras estabilizadas en el formato de npz_to_webgl.py: el visor orienta y escala con ellas las capas de
    # exports/estable/ (y la malla estabilizada) y dibuja el recorrido.
    Kc = d["intrinsic"][real].astype(np.float64).copy()
    Kc[:, :2, :] /= max(1, int(d["ds"]) if "ds" in d else 1)
    with open(out + "_cameras.json", "w") as fh:
        json.dump({"npz": os.path.basename(npz), "n_cameras": int(len(real)), "poses": "estabilizadas (o3d_estable.py)",
                   "version_poses": pose_out,
                   "cameras": [{"index": int(i), "c2w": c2w_new[k].tolist(), "intrinsic": Kc[k].tolist()}
                               for k, i in enumerate(real)]}, fh)
    outputs["camaras"] = out + "_cameras.json"
    graph.save(outputs["grafo_poses"], {"version_poses_entrada": pose_in, "version_poses_salida": pose_out,
                                        "npz": npz})
    with open(outputs["grafo_estructura"], "w") as fh:
        json.dump({"sesion": session, "version_poses": pose_out, **struct_graph}, fh, indent=1)
    deps = {"sesion": session, "version_poses": pose_out, "version_poses_entrada": pose_in,
            "huella_mascara": meta["huella_mascara"], "artefactos": outputs,
            "derivados": {"malla TSDF (tsdf_mesh.py --stable)": "se rehace si cambia version_poses o huella_mascara",
                          "gemelo MuJoCo (Pruebas_Macondo, --poses estable)": "se rehace si cambia version_poses",
                          "etiquetas DINOv2 por frame": "no dependen de las poses",
                          "objetos y memoria semántica": "se recalculan desde sus observaciones con la nueva versión"}}
    with open(out + "_dependencias.json", "w") as fh:
        json.dump(deps, fh, indent=1)
    report["diagnostico"] = diag
    report["piso"] = floor_report
    report["tiempos_s"] = {k: round(v, 1) for k, v in times.items()}
    report["tiempo_total_s"] = round(time.time() - t0, 1)
    report["salidas"] = outputs
    report["meta"] = meta
    with open(outputs["informe"], "w") as fh:
        json.dump(report, fh, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    log(f"listo en {report['tiempo_total_s']} s -> {outputs['poses_y_mascaras']}", flush=True)
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("npz")
    ap.add_argument("--out", required=True, help="prefijo de salida: <out>_estable.npz, <out>_grafo.json, ...")
    ap.add_argument("--filter", default=None, help="<..>_filtro.npz de geo_filter.py: su semántica (personas, "
                                                   "cielo) y sus grupos; sus poses deben ser las mismas")
    for f in dataclasses.fields(Params):
        if isinstance(f.default, tuple):
            continue
        if isinstance(f.default, bool) and f.default:
            ap.add_argument("--no_" + f.name, dest=f.name, action="store_false", default=f.default)
        elif isinstance(f.default, bool):
            ap.add_argument("--" + f.name, action="store_true")
        else:
            ap.add_argument("--" + f.name, type=type(f.default), default=f.default)
    a = ap.parse_args()
    p = Params(**{f.name: getattr(a, f.name) for f in dataclasses.fields(Params) if not isinstance(f.default, tuple)})
    run(a.npz, a.out, p, a.filter)


if __name__ == "__main__":
    main()
