"""Filtro geométrico previo a la malla TSDF y al Gaussian Splatting.

No modifica las predicciones del modelo: lee eval/<name>.npz y escribe un archivo aparte,
<out>_filtro.npz, con máscaras y correcciones que tsdf_mesh.py y gsplat_train.py aplican
solo a los frames con los que construyen (los frames apartados para evaluar se miden
siempre contra la profundidad original). La nube fusionada (export_dense_cloud.py) no lo usa.

Etapas, todas por píxel de cada frame real:

  1. Semántica (SegFormer-B0, ADE20K): etiqueta cada píxel. Se descartan personas, animales
     y cielo (objetos que se mueven o sin profundidad real) y se agrupan pared, piso y techo
     para la etapa estructural.
  2. Bordes: píxeles con salto de profundidad grande en su vecindad 3x3 ("flying pixels").
  3. Consistencia multivista: la profundidad de cada frame se proyecta en sus vecinos más
     solapados (cercanos en el tiempo y, sobre todo, lejanos: las revisitas, que es donde
     aparecen las paredes dobles). Cuenta apoyos (el vecino ve la misma superficie) y
     violaciones de espacio libre (el vecino ve más lejos a través del punto: el punto está
     flotando delante de una superficie). Se descarta lo que nadie apoya o lo que varios
     vecinos atraviesan.
  4. Estructura: con los píxeles que sobreviven se buscan planos (paredes verticales por
     RANSAC de 2 puntos con la vertical como restricción, pisos y techos por histograma de
     altura). Las capas paralelas y cercanas vistas desde el mismo lado se fusionan (pared
     doble -> una pared). Cada plano se parte en tramos conexos, cada tramo es un
     rectángulo (4 esquinas) y las esquinas entre tramos que se tocan se calculan como la
     intersección de tres planos (dos paredes y piso/techo). Sale una malla simple,
     <out>_estructura.glb, hecha solo con esas esquinas.
  4b. Realineación por planos (opcional, --align): ICP punto-a-plano de cada frame contra
     la estructura, suavizado en el tiempo. Aplana las paredes, pero en la medición del
     2026-10-04 empeoró las vistas nuevas del splat, por eso está apagada por defecto.
  5. Ajuste a planos: la profundidad de los píxeles que caen cerca de un tramo se reemplaza
     por la intersección del rayo con el plano (depth_snap): las dos capas de una pared doble
     terminan sobre la misma superficie.
  6. Selección de frames: se conserva un frame cuando su solape con el último conservado
     baja de un umbral (y, entre los candidatos, el más nítido). Limita cuántos frames usan
     la malla y el splat.
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

# --- clases ADE20K -----------------------------------------------------------------------
REMOVE_DEPTH = [12, 126, 2]                    # persona, animal, cielo
REMOVE_PHOTO = [12, 126]                       # persona, animal (se mueven entre frames)
WALL_LIKE = [0, 14, 8, 58, 22, 27, 144, 63, 18, 100, 130]   # pared, puerta, ventana, puerta mosquitero,
                                               # cuadro, espejo, cartelera, persiana, cortina, póster, pantalla
FLOOR_LIKE = [3, 28, 6, 11, 52]                # piso, alfombra, calle, andén, sendero
CEIL_LIKE = [5]
G_OTHER, G_WALL, G_FLOOR, G_CEIL, G_REMOVED = 0, 1, 2, 3, 4


def lut(ids, n=256):
    t = np.zeros(n, bool)
    t[ids] = True
    return t


@torch.no_grad()
def segment(images, dev, batch=8):
    """images [N,H,W,3] uint8 -> etiquetas ADE20K [N,H,W] uint8."""
    from transformers import SegformerForSemanticSegmentation
    model = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/segformer-b0-finetuned-ade-512-512").to(dev).eval().half()
    mean = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)
    N, H, W, _ = images.shape
    out = np.zeros((N, H, W), np.uint8)
    for s in range(0, N, batch):
        x = torch.from_numpy(images[s:s + batch]).to(dev).permute(0, 3, 1, 2).float() / 255.0
        x = F.interpolate(x, size=(512, 512), mode="bilinear", align_corners=False)
        x = ((x - mean) / std).half()
        logits = model(pixel_values=x).logits.float()
        logits = F.interpolate(logits, size=(H, W), mode="bilinear", align_corners=False)
        out[s:s + batch] = logits.argmax(1).byte().cpu().numpy()
    del model
    torch.cuda.empty_cache()
    return out


def backproject(D, K, c2w, ys, xs):
    """D [...] sobre la grilla (ys, xs) -> puntos en el mundo [..., 3] (torch)."""
    X = (xs - K[0, 2]) / K[0, 0] * D
    Y = (ys - K[1, 2]) / K[1, 1] * D
    P = torch.stack([X, Y, D], -1)
    return P @ c2w[:3, :3].T + c2w[:3, 3]


def project(P, K, w2c):
    """Puntos del mundo [M,3] -> (u, v, z) en la cámara w2c (torch)."""
    Pc = P @ w2c[:3, :3].T + w2c[:3, 3]
    z = Pc[:, 2]
    zs = z.clamp(min=1e-6)
    return K[0, 0] * Pc[:, 0] / zs + K[0, 2], K[1, 1] * Pc[:, 1] / zs + K[1, 2], z


def sample_nearest(img, u, v):
    """img [H,W], u/v [M] en píxeles -> valor del píxel más cercano (0 si cae afuera)."""
    H, W = img.shape
    ui, vi = torch.round(u).long(), torch.round(v).long()
    inside = (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
    val = torch.zeros_like(u)
    val[inside] = img[vi[inside], ui[inside]]
    return val, inside


def overlap_matrix(D, valid, K, c2w, w2c, stride=16):
    """O[i,j] = fracción de los puntos de i (grilla gruesa) que caen dentro de la imagen de j,
    delante de la cámara y sin quedar tapados por lo que j ve."""
    N, H, W = D.shape
    dev = D.device
    ys, xs = torch.meshgrid(torch.arange(stride // 2, H, stride, device=dev),
                            torch.arange(stride // 2, W, stride, device=dev), indexing="ij")
    ys, xs = ys.float(), xs.float()
    pts, seg = [], []
    for i in range(N):
        d = D[i, ys.long(), xs.long()]
        m = valid[i, ys.long(), xs.long()]
        pts.append(backproject(d[m], K[i], c2w[i], ys[m], xs[m]))
        seg.append(torch.full((int(m.sum()),), i, device=dev, dtype=torch.long))
    pts, seg = torch.cat(pts), torch.cat(seg)
    cnt = torch.bincount(seg, minlength=N).clamp(min=1).float()
    O = torch.zeros(N, N, device=dev)
    for j in range(N):
        u, v, z = project(pts, K[j], w2c[j])
        dj, inside = sample_nearest(D[j], u, v)
        ok = inside & (z > 0) & (dj > 0) & (z < dj * 1.15)
        O[:, j] = torch.bincount(seg, weights=ok.float(), minlength=N) / cnt
    return O


def pick_neighbors(O, k_near, k_far, gap):
    N = O.shape[0]
    idx = torch.arange(N, device=O.device)
    nbrs, far_flags = [], []
    for i in range(N):
        score = torch.minimum(O[i], O[:, i]).clone()   # solape en las dos direcciones
        score[i] = -1
        near = (idx - i).abs() <= gap
        s_near = torch.where(near, score, torch.full_like(score, -1))
        s_far = torch.where(~near, score, torch.full_like(score, -1))
        a = [int(j) for j in torch.topk(s_near, min(k_near, N)).indices if s_near[j] > 0.15]
        b = [int(j) for j in torch.topk(s_far, min(k_far, N)).indices if s_far[j] > 0.15]
        nbrs.append(a + b)
        far_flags.append([False] * len(a) + [True] * len(b))
    return nbrs, far_flags


def edge_mask(D, thr):
    Dp = D.unsqueeze(1)
    mx = F.max_pool2d(Dp, 3, 1, 1)
    mn = -F.max_pool2d(-Dp, 3, 1, 1)
    return ((mx - mn) / D.unsqueeze(1).clamp(min=1e-6) > thr).squeeze(1)


def ear_clip(poly):
    """Triangulación por recorte de orejas de un polígono simple [M,2] -> lista de (a,b,c)."""
    pts = [tuple(p) for p in poly]
    idx = list(range(len(pts)))
    area = sum(pts[i][0] * pts[(i + 1) % len(pts)][1] - pts[(i + 1) % len(pts)][0] * pts[i][1]
               for i in range(len(pts)))
    if area < 0:
        idx.reverse()

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    def inside(p, a, b, c):
        return cross(a, b, p) >= 0 and cross(b, c, p) >= 0 and cross(c, a, p) >= 0

    tris, guard = [], 0
    while len(idx) > 3 and guard < 10000:
        guard += 1
        n = len(idx)
        for k in range(n):
            ia, ib, ic = idx[(k - 1) % n], idx[k], idx[(k + 1) % n]
            a, b, c = pts[ia], pts[ib], pts[ic]
            if cross(a, b, c) <= 1e-12:
                continue
            if any(inside(pts[j], a, b, c) for j in idx if j not in (ia, ib, ic)):
                continue
            tris.append((ia, ib, ic))
            idx.pop(k)
            break
        else:
            break
    if len(idx) == 3:
        tris.append(tuple(idx))
    return tris


# --- estructura ----------------------------------------------------------------------------
def fit_vertical_plane(P, up):
    """Plano vertical por mínimos cuadrados: la normal es la dirección de menor varianza de la
    proyección horizontal de los puntos."""
    e1 = np.cross(up, [1.0, 0, 0]) if abs(up[0]) < 0.9 else np.cross(up, [0, 1.0, 0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(up, e1)
    uv = np.stack([P @ e1, P @ e2], 1)
    c2 = uv.mean(0)
    w, V = np.linalg.eigh(np.cov((uv - c2).T))
    n = V[0, 0] * e1 + V[1, 0] * e2                 # dirección horizontal de menor varianza
    n /= np.linalg.norm(n)
    c = c2[0] * e1 + c2[1] * e2
    return n, float(n @ c)


def ransac_vertical(P, Nrm, up, tol, min_pts, max_planes, iters=600, rng=None):
    """Planos verticales sucesivos. Muestra mínima: 2 puntos (la vertical fija el resto)."""
    rng = rng or np.random.default_rng(0)
    rest = np.arange(len(P))
    planes = []
    while len(rest) >= min_pts and len(planes) < max_planes:
        Q, Qn = P[rest], Nrm[rest]
        i1 = rng.integers(0, len(Q), iters)
        i2 = rng.integers(0, len(Q), iters)
        dvec = Q[i2] - Q[i1]
        n = np.cross(dvec, up)
        nn = np.linalg.norm(n, axis=1)
        good = nn > 1e-9
        n, i1 = n[good] / nn[good, None], i1[good]
        d = np.einsum("ij,ij->i", n, Q[i1])
        best, best_cnt = None, 0
        for s in range(0, len(n), 50):                       # evaluar por lotes
            dist = np.abs(Q @ n[s:s + 50].T - d[s:s + 50])
            al = np.abs(Qn @ n[s:s + 50].T) > math.cos(math.radians(30))
            cnt = ((dist < tol) & al).sum(0)
            k = int(cnt.argmax())
            if cnt[k] > best_cnt:
                best_cnt, best = int(cnt[k]), (n[s + k], d[s + k])
        if best is None or best_cnt < min_pts:
            break
        n0, d0 = best
        inl = (np.abs(Q @ n0 - d0) < tol) & (np.abs(Qn @ n0) > math.cos(math.radians(30)))
        n1, d1 = fit_vertical_plane(Q[inl], up)
        inl = (np.abs(Q @ n1 - d1) < tol) & (np.abs(Qn @ n1) > math.cos(math.radians(30)))
        if inl.sum() < min_pts:
            break
        planes.append({"n": n1, "d": d1, "idx": rest[inl]})
        rest = rest[~inl]
    return planes


def horizontal_levels(h, tol, min_pts, min_gap, min_frac=0.1):
    """Alturas dominantes (pisos o techos) por histograma: picos sucesivos."""
    levels = []
    h = np.sort(h)
    rest = h.copy()
    while len(rest) >= min_pts:
        bins = np.arange(rest.min(), rest.max() + tol, tol)
        if len(bins) < 2:
            bins = np.array([rest.min() - tol, rest.max() + tol])
        hist, edges = np.histogram(rest, bins)
        k = int(hist.argmax())
        c = 0.5 * (edges[k] + edges[k + 1])
        sel = np.abs(rest - c) < tol
        if sel.sum() < min_pts:
            break
        c = float(np.median(rest[sel]))
        sel = np.abs(rest - c) < tol
        levels.append((c, int(sel.sum())))
        rest = rest[~sel]
    # niveles más cercanos que min_gap son el mismo (ruido); niveles chicos se descartan
    levels.sort()
    out = []
    for c, n in levels:
        if out and c - out[-1][0] < min_gap:
            c0, n0 = out[-1]
            out[-1] = ((c0 * n0 + c * n) / (n0 + n), n0 + n)
        else:
            out.append((c, n))
    if out:
        big = max(n for _, n in out)
        out = [(float(c), int(n)) for c, n in out if n >= min_frac * big]
    return out


def components_2d(uv, cell, min_pts, dilate=1):
    """Tramos conexos de una nube 2D (grilla de ocupación). Devuelve una lista de índices."""
    import scipy.ndimage as ndi
    g = np.floor((uv - uv.min(0)) / cell).astype(int)
    shape = g.max(0) + 1
    occ = np.zeros(shape, bool)
    occ[g[:, 0], g[:, 1]] = True
    if dilate:
        occ = ndi.binary_dilation(occ, iterations=dilate)
    lab, n = ndi.label(occ)
    pl = lab[g[:, 0], g[:, 1]]
    out = []
    for k in range(1, n + 1):
        idx = np.flatnonzero(pl == k)
        if len(idx) >= min_pts:
            out.append(idx)
    return out


def align_to_planes(D, K, c2w_np, keep, group, segs, floors, up, med, dev, tol_rel=0.15, lam=0.1, sigma=15.0,
                    max_rot_deg=3.0, max_trans_rel=0.1):
    """Realineación suave de cada frame contra los planos de la estructura.

    Por frame: ICP punto-a-plano rígido (escala fija) de sus píxeles de pared/piso contra los
    tramos y pisos cercanos, con pesos de Cauchy y amortiguación lam. Se descartan las
    correcciones grandes o con un solo plano a la vista (mal condicionadas). Después las
    correcciones (vector de rotación y traslación del mundo) se suavizan en el tiempo con una
    gaussiana de sigma frames: la deriva entre revisitas es lenta, y corregir cada frame por
    separado mete un zigzag de frame a frame. Devuelve las poses c2w corregidas y estadísticas."""
    from scipy.ndimage import gaussian_filter1d
    from scipy.spatial.transform import Rotation as Rot
    N, H, W = D.shape
    Pn = [s["n"] for s in segs] + [up for _ in floors]
    Pd = [s["d"] for s in segs] + [f for f, _ in floors]
    if len(Pn) < 2:
        return c2w_np.copy(), {"aligned_frames": 0}
    ext = [[s["u0"], s["u1"], s["h0"], s["h1"]] for s in segs] + [[-1e9, 1e9, -1e9, 1e9] for _ in floors]
    kind = [G_WALL] * len(segs) + [G_FLOOR] * len(floors)
    Pn = torch.tensor(np.array(Pn), device=dev, dtype=torch.float32)
    Pd = torch.tensor(Pd, device=dev, dtype=torch.float32)
    ext = torch.tensor(ext, device=dev, dtype=torch.float32)
    kind = torch.tensor(kind, device=dev)
    upt = torch.tensor(up, device=dev, dtype=torch.float32)
    T = torch.cross(upt.expand_as(Pn), Pn, dim=1)
    tol = tol_rel * med
    st3 = 3
    ys, xs = torch.meshgrid(torch.arange(0, H, st3, device=dev).float(), torch.arange(0, W, st3, device=dev).float(),
                            indexing="ij")
    rv = np.zeros((N, 3)); bb = np.zeros((N, 3)); valid = np.zeros(N)
    for i in range(N):
        g = torch.from_numpy(group[i, ::st3, ::st3].astype(np.int64)).to(dev)
        kk = torch.from_numpy(keep[i, ::st3, ::st3]).to(dev) & ((g == G_WALL) | (g == G_FLOOR))
        cw = torch.from_numpy(c2w_np[i]).to(dev).float()
        X0 = backproject(D[i, ::st3, ::st3][kk], K[i], cw, ys[kk], xs[kk])
        gg = g[kk]
        if len(X0) < 200:
            continue
        mu = X0.mean(0)
        R = torch.eye(3, device=dev)
        v = torch.zeros(3, device=dev)
        m = None
        for _ in range(5):
            X = mu + (X0 - mu) @ R.T + v
            dist = X @ Pn.T - Pd
            u = X @ T.T
            h = (X @ upt)[:, None]
            inside = (u > ext[:, 0] - tol) & (u < ext[:, 1] + tol) & (h > ext[:, 2] - tol) & (h < ext[:, 3] + tol)
            ok = inside & (dist.abs() < tol) & (gg[:, None] == kind[None, :])
            dd = torch.where(ok, dist.abs(), torch.full_like(dist, 1e9))
            j = dd.argmin(1)
            m = dd.min(1).values < 1e8
            if m.sum() < 100:
                break
            n = Pn[j[m]]
            r = dist[m, j[m]]
            Y = X[m] - mu
            J = torch.cat([torch.cross(Y, n, dim=1), n], 1)          # d r / d[omega, v]
            w_ = 1.0 / (1 + (r / (0.05 * med)) ** 2)                  # Cauchy
            A = (J * w_[:, None]).T @ J + lam * len(r) * torch.eye(6, device=dev)
            x = torch.linalg.solve(A, -(J * w_[:, None]).T @ r)
            dR = torch.from_numpy(Rot.from_rotvec(x[:3].cpu().numpy()).as_matrix()).to(dev).float()
            R = dR @ R
            v = dR @ v + x[3:]
        if m is None or m.sum() < 100 or len(torch.unique(j[m])) < 2:
            continue
        rot = Rot.from_matrix(R.cpu().numpy().astype(np.float64)).as_rotvec()
        if np.degrees(np.linalg.norm(rot)) > max_rot_deg or float(v.norm()) / med > max_trans_rel:
            continue
        rv[i] = rot
        bb[i] = (mu - R @ mu + v).cpu().numpy()
        valid[i] = 1.0
    wsum = gaussian_filter1d(valid, sigma, mode="nearest") + 1e-9
    rvs = gaussian_filter1d(rv * valid[:, None], sigma, axis=0, mode="nearest") / wsum[:, None]
    bbs = gaussian_filter1d(bb * valid[:, None], sigma, axis=0, mode="nearest") / wsum[:, None]
    out = c2w_np.astype(np.float64).copy()
    for k in range(N):
        Rk = Rot.from_rotvec(rvs[k]).as_matrix()
        out[k, :3, :3] = Rk @ c2w_np[k, :3, :3]
        out[k, :3, 3] = Rk @ c2w_np[k, :3, 3] + bbs[k]
    st = {"aligned_frames": int(valid.sum()),
          "rot_deg_median": round(float(np.degrees(np.median(np.linalg.norm(rvs, axis=1)))), 3),
          "trans_rel_median": round(float(np.median(np.linalg.norm(bbs, axis=1)) / med), 4)}
    return out.astype(np.float32), st


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("npz")
    ap.add_argument("--out", required=True, help="prefijo de salida: <out>_filtro.npz, <out>_estructura.glb ...")
    ap.add_argument("--ref_conf_percentile", type=float, default=20.0,
                    help="confianza mínima de la profundidad de los vecinos que votan")
    ap.add_argument("--edge_thr", type=float, default=0.08, help="salto relativo de profundidad en 3x3")
    ap.add_argument("--k_near", type=int, default=4)
    ap.add_argument("--k_far", type=int, default=8)
    ap.add_argument("--gap", type=int, default=24, help="frames: más allá de esto un vecino es 'lejano'")
    ap.add_argument("--tau_sup", type=float, default=0.06, help="tolerancia relativa de apoyo")
    ap.add_argument("--tau_vio", type=float, default=0.15, help="margen relativo de violación de espacio libre")
    ap.add_argument("--min_support", type=int, default=1)
    ap.add_argument("--max_vio_ratio", type=float, default=0.34,
                    help="descartar si más de esta fracción de las observaciones lo atraviesa (y al menos 2)")
    ap.add_argument("--plane_tol_rel", type=float, default=0.02, help="inlier de un plano (rel. a la prof. mediana)")
    ap.add_argument("--merge_rel", type=float, default=0.3,
                    help="dos capas paralelas más cerca que esto (rel. a prof. mediana) son la misma pared")
    ap.add_argument("--corner_rel", type=float, default=0.25, help="distancia para unir tramos en una esquina")
    ap.add_argument("--snap_rel", type=float, default=0.12, help="distancia máxima para llevar un píxel al plano")
    ap.add_argument("--max_planes", type=int, default=60)
    ap.add_argument("--manhattan_deg", type=float, default=25.0,
                    help="girar a los ejes dominantes las paredes a menos de estos grados (0 = no)")
    ap.add_argument("--wall_min_height", type=float, default=0.55,
                    help="alto mínimo de un tramo de pared (fracción de la altura del ambiente)")
    ap.add_argument("--oblique_min_frac", type=float, default=0.15,
                    help="una pared fuera de los ejes de Manhattan se conserva solo con esta fracción de la mayor")
    ap.add_argument("--wall_min_len", type=float, default=0.25, help="largo mínimo (rel. a prof. mediana)")
    ap.add_argument("--level_gap_rel", type=float, default=0.12,
                    help="dos pisos (o techos) más cerca que esto son el mismo nivel")
    ap.add_argument("--select_overlap", type=float, default=0.7,
                    help="conservar un frame cuando su solape con el último conservado baja de esto")
    ap.add_argument("--max_skip", type=int, default=6, help="nunca saltar más de estos frames seguidos")
    ap.add_argument("--align", dest="align", action="store_true", default=False,
                    help="realinear los frames contra los planos (experimental: aplana las paredes pero empeora "
                         "las vistas nuevas del splat; ver la bitácora del 2026-10-04)")
    ap.add_argument("--no_align", dest="align", action="store_false", help="(por defecto) no realinear")
    ap.add_argument("--align_sigma", type=float, default=15.0, help="suavizado temporal de la realineación (frames)")
    ap.add_argument("--no_semantics", action="store_true")
    ap.add_argument("--dump_points", action="store_true", help="guardar la nube de la etapa estructural (depuración)")
    a = ap.parse_args()

    t0 = time.time()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d = np.load(a.npz)
    real = np.flatnonzero(d["is_real"])
    N = len(real)
    depth = d["depth"][real].astype(np.float32)
    conf = d["depth_conf"][real].astype(np.float32)
    imgs = d["images"][real]
    Kn = d["intrinsic"][real].astype(np.float32)
    E = np.tile(np.eye(4, dtype=np.float32), (N, 1, 1))
    E[:, :3, :4] = d["extrinsic"][real]
    c2w_np = np.linalg.inv(E)
    _, H, W = depth.shape
    med = float(np.median(depth[::10][depth[::10] > 0]))
    cref = float(np.percentile(conf[::5], a.ref_conf_percentile))
    print(f"{N} frames {W}x{H} | profundidad mediana {med:.4f} | confianza de referencia > {cref:.3f}", flush=True)
    stats = {"frames": N, "median_depth": med}

    # 1. semántica -----------------------------------------------------------------------
    if a.no_semantics:
        labels = np.zeros((N, H, W), np.uint8) + 255
    else:
        ts = time.time()
        labels = segment(imgs, dev)
        stats["seg_seconds"] = round(time.time() - ts, 1)
        print(f"semántica: {stats['seg_seconds']} s", flush=True)
    rm_depth = lut(REMOVE_DEPTH)[labels]
    rm_photo = lut(REMOVE_PHOTO)[labels]
    group = np.zeros((N, H, W), np.uint8)
    group[lut(WALL_LIKE)[labels]] = G_WALL
    group[lut(FLOOR_LIKE)[labels]] = G_FLOOR
    group[lut(CEIL_LIKE)[labels]] = G_CEIL
    group[rm_depth] = G_REMOVED

    # 2-3. bordes y consistencia multivista ----------------------------------------------------
    D = torch.from_numpy(depth).to(dev)
    Cf = torch.from_numpy(conf).to(dev)
    K = torch.from_numpy(Kn).to(dev)
    c2w = torch.from_numpy(c2w_np).to(dev)
    w2c = torch.linalg.inv(c2w)
    valid_ref = (D > 0) & (Cf >= cref) & ~torch.from_numpy(rm_depth).to(dev)
    Dref = torch.where(valid_ref, D, torch.zeros_like(D))
    ts = time.time()
    O = overlap_matrix(Dref, valid_ref, K, c2w, w2c)
    nbrs, farf = pick_neighbors(O, a.k_near, a.k_far, a.gap)
    stats["overlap_seconds"] = round(time.time() - ts, 1)
    stats["neighbors_mean"] = round(float(np.mean([len(x) for x in nbrs])), 2)
    stats["far_neighbors_mean"] = round(float(np.mean([sum(f) for f in farf])), 2)
    print(f"solapes: {stats['overlap_seconds']} s | vecinos por frame {stats['neighbors_mean']} "
          f"(lejanos {stats['far_neighbors_mean']})", flush=True)

    ys, xs = torch.meshgrid(torch.arange(H, device=dev).float(), torch.arange(W, device=dev).float(), indexing="ij")
    keep = np.zeros((N, H, W), bool)
    reasons = {"invalid": 0, "semantic": 0, "edge": 0, "no_support": 0, "free_space": 0, "kept": 0}
    sup_hist = np.zeros(20, np.int64)
    ts = time.time()
    for i in range(N):
        Di = D[i]
        base = Di > 0
        edge = edge_mask(Di[None], a.edge_thr)[0]
        sem = torch.from_numpy(rm_depth[i]).to(dev)
        cand = base & ~sem & ~edge
        P = backproject(Di[cand], K[i], c2w[i], ys[cand], xs[cand])
        sup = torch.zeros(len(P), device=dev)
        vio = torch.zeros(len(P), device=dev)
        obs = torch.zeros(len(P), device=dev)
        for j in nbrs[i]:
            u, v, z = project(P, K[j], w2c[j])
            dj, inside = sample_nearest(Dref[j], u, v)
            ok = inside & (z > 0) & (dj > 0)
            o = ok & (z < dj * (1 + a.tau_sup))                 # no tapado
            s = o & ((z - dj).abs() < a.tau_sup * dj)
            vv = o & (z < dj * (1 - a.tau_vio))
            obs += o.float(); sup += s.float(); vio += vv.float()
        good = (sup >= a.min_support) & ~((vio >= 2) & (vio > a.max_vio_ratio * obs))
        k = torch.zeros(H, W, dtype=torch.bool, device=dev)
        k[cand] = good
        keep[i] = k.cpu().numpy()
        reasons["invalid"] += int((~base).sum())
        reasons["semantic"] += int((base & sem).sum())
        reasons["edge"] += int((base & ~sem & edge).sum())
        reasons["no_support"] += int((sup < a.min_support).sum())
        reasons["free_space"] += int(((sup >= a.min_support) & ~good).sum())
        reasons["kept"] += int(good.sum())
        sup_hist += np.bincount(sup.clamp(max=19).long().cpu().numpy(), minlength=20)
    tot = N * H * W
    stats["consistency_seconds"] = round(time.time() - ts, 1)
    stats["pixels"] = {k: round(v / tot, 4) for k, v in reasons.items()}
    stats["support_hist"] = (sup_hist / sup_hist.sum()).round(4).tolist()
    print(f"consistencia: {stats['consistency_seconds']} s | fracción de píxeles: {stats['pixels']}", flush=True)

    # 4. estructura ----------------------------------------------------------------------------
    ts = time.time()
    # vertical: autovector menor de la suma de x x^T de los ejes X de las cámaras
    X = c2w_np[:, :3, 0]
    w_, V_ = np.linalg.eigh(X.T @ X)
    up = V_[:, 0]
    if up @ c2w_np[:, :3, 1].mean(0) > 0:      # +Y de la cámara apunta al suelo
        up = -up
    rng = np.random.default_rng(0)
    pts, cols, grp, cam = [], [], [], []
    st = 3
    gy, gx = np.mgrid[0:H:st, 0:W:st]
    for i in range(N):
        m = keep[i, gy, gx] & (conf[i, gy, gx] >= cref)
        z = depth[i, gy, gx][m]
        Pc = np.stack([(gx[m] - Kn[i, 0, 2]) / Kn[i, 0, 0] * z, (gy[m] - Kn[i, 1, 2]) / Kn[i, 1, 1] * z, z], 1)
        pts.append(Pc @ c2w_np[i, :3, :3].T + c2w_np[i, :3, 3])
        cols.append(imgs[i, gy, gx][m])
        grp.append(group[i, gy, gx][m])
        cam.append(np.full(m.sum(), i, np.int32))
    pts = np.concatenate(pts); cols = np.concatenate(cols); grp = np.concatenate(grp); cam = np.concatenate(cam)
    # fusión por vóxel: posición y color medios, grupo por mayoría, una cámara representativa
    vox = med / 120.0
    key = np.floor((pts - pts.min(0)) / vox).astype(np.int64)
    key = (key[:, 0] << 42) | (key[:, 1] << 21) | key[:, 2]
    uk, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
    M = len(uk)
    P = np.stack([np.bincount(inv, pts[:, c], M) for c in range(3)], 1) / cnt[:, None]
    C = np.stack([np.bincount(inv, cols[:, c].astype(np.float64), M) for c in range(3)], 1) / cnt[:, None]
    gv = np.zeros((M, 5))
    for g in range(5):
        gv[:, g] = np.bincount(inv, grp == g, M)
    G = gv.argmax(1)
    first = np.full(M, -1, np.int64)
    first[inv[::-1]] = np.arange(len(inv))[::-1]
    Cam = cam[first]
    Pnt = P[cnt >= 2]; Col = C[cnt >= 2]; Grp = G[cnt >= 2]; Cam = Cam[cnt >= 2]
    import open3d as o3d
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(Pnt))
    pc.estimate_normals(o3d.geometry.KDTreeSearchParamKNN(20))
    Nrm = np.asarray(pc.normals)
    tocam = c2w_np[Cam, :3, 3] - Pnt
    Nrm[np.einsum("ij,ij->i", Nrm, tocam) < 0] *= -1        # normales hacia la cámara que lo vio
    stats["structure_points"] = int(len(Pnt))
    print(f"nube para estructura: {len(Pnt):,} puntos (vóxel {vox:.4f}) | vertical {np.round(up, 3).tolist()}",
          flush=True)

    if a.dump_points:
        np.savez(a.out + "_pts.npz", P=Pnt, N=Nrm, G=Grp, C=Col, cam=Cam, up=up, med=med)
    tol = a.plane_tol_rel * med
    min_pts = max(150, len(Pnt) // 2000)
    horiz = np.abs(Nrm @ up) < 0.35
    wall_sel = np.flatnonzero((Grp == G_WALL) & horiz)
    planes = ransac_vertical(Pnt[wall_sel], Nrm[wall_sel], up, tol, min_pts, a.max_planes, rng=rng)
    for p in planes:
        p["idx"] = wall_sel[p["idx"]]
        side = np.sign(np.einsum("ij,j->i", c2w_np[Cam[p["idx"]], :3, 3], p["n"]) - p["d"])
        if side.mean() < 0:                     # normal hacia las cámaras que lo ven
            p["n"], p["d"] = -p["n"], -p["d"]
    stats["planes_ransac"] = len(planes)
    e1 = np.cross(up, [1.0, 0, 0]) if abs(up[0]) < 0.9 else np.cross(up, [0, 1.0, 0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(up, e1)

    def refit_d(p):
        p["d"] = float(np.median(Pnt[p["idx"]] @ p["n"]))

    # alineación de Manhattan: la orientación dominante de las paredes (módulo 90°), pesada por
    # puntos; las paredes a menos de manhattan_deg de un eje se giran exactamente a ese eje
    n_manh = 0
    if planes and a.manhattan_deg > 0:
        ang = np.array([math.atan2(p["n"] @ e2, p["n"] @ e1) for p in planes])
        wts = np.array([len(p["idx"]) for p in planes], float)
        z = (wts * np.exp(4j * ang)).sum()                    # media circular de 4·ángulo
        th0 = np.angle(z) / 4
        stats["manhattan_axis_deg"] = round(math.degrees(th0), 2)
        for p, an in zip(planes, ang):
            k = round((an - th0) / (math.pi / 2))
            target = th0 + k * math.pi / 2
            if abs(an - target) < math.radians(a.manhattan_deg):
                p["n"] = math.cos(target) * e1 + math.sin(target) * e2
                refit_d(p)
                n_manh += 1
        # paredes oblicuas: solo se conservan si son grandes (en interiores casi todo es Manhattan)
        big = max(len(p["idx"]) for p in planes)
        keep_pl = []
        for p, an in zip(planes, ang):
            k = round((an - th0) / (math.pi / 2))
            aligned = abs(an - (th0 + k * math.pi / 2)) < math.radians(a.manhattan_deg)
            if aligned or len(p["idx"]) >= a.oblique_min_frac * big:
                keep_pl.append(p)
        stats["walls_oblique_dropped"] = len(planes) - len(keep_pl)
        planes = keep_pl
    stats["walls_manhattan"] = n_manh

    # fusión de capas paralelas cercanas vistas desde el mismo lado (paredes dobles)
    merge_d = a.merge_rel * med
    gaps = []
    merged = 0
    changed = True
    while changed:
        changed = False
        order = sorted(range(len(planes)), key=lambda x: -len(planes[x]["idx"]))
        for x in order:
            for y in order:
                if y == x:
                    continue
                p, q = planes[x], planes[y]
                if len(q["idx"]) > len(p["idx"]):
                    continue
                cosang = float(p["n"] @ q["n"])
                if cosang < math.cos(math.radians(12)):        # paralelas y con la misma orientación
                    continue
                t = np.cross(up, p["n"])
                up_ = Pnt[p["idx"]] @ t
                uq_ = Pnt[q["idx"]] @ t
                lo = max(np.percentile(up_, 5), np.percentile(uq_, 5))
                hi = min(np.percentile(up_, 95), np.percentile(uq_, 95))
                shorter = min(np.ptp(np.percentile(up_, [5, 95])), np.ptp(np.percentile(uq_, [5, 95])))
                if hi - lo < 0.3 * shorter:
                    continue
                inq = (uq_ >= lo) & (uq_ <= hi)
                if inq.sum() < 10:
                    continue
                off = abs(float(np.median(Pnt[q["idx"]][inq] @ p["n"])) - p["d"])
                if not off <= merge_d:                  # (también descarta NaN)
                    continue
                gaps.append(off / med)
                p["idx"] = np.concatenate([p["idx"], q["idx"]])
                refit_d(p)                                   # la orientación la fija la pared mayor
                planes.pop(y)
                merged += 1
                changed = True
                break
            if changed:
                break
    stats["walls_merged"] = merged
    stats["merged_gap_rel"] = [round(g, 4) for g in sorted(gaps)]
    print(f"paredes: {stats['planes_ransac']} planos por RANSAC, {n_manh} alineados a Manhattan, {merged} capas "
          f"fusionadas (separación / prof. mediana, mediana {np.median(gaps) if gaps else 0:.3f})", flush=True)

    # pisos y techos
    hgt = Pnt @ up
    vert_n = np.abs(Nrm @ up) > 0.8
    floors = horizontal_levels(hgt[(Grp == G_FLOOR) & vert_n], tol * 1.5, min_pts, a.level_gap_rel * med)
    ceils = horizontal_levels(hgt[(Grp == G_CEIL) & vert_n], tol * 1.5, min_pts, a.level_gap_rel * med)
    stats["floor_levels"] = [round(h / med, 3) for h, _ in floors]
    stats["ceiling_levels"] = [round(h / med, 3) for h, _ in ceils]
    room_h = (min(c for c, _ in ceils) - max(f for f, _ in floors)) if floors and ceils else None
    print(f"pisos {len(floors)} niveles, techos {len(ceils)}"
          + (f", altura del ambiente {room_h / med:.2f} x prof. mediana" if room_h else ""), flush=True)

    # tramos rectangulares de pared: una pared real es alta (llega cerca del piso o del techo),
    # tiene largo y su rectángulo está razonablemente cubierto de puntos
    cell = med / 20.0
    min_h = a.wall_min_height * (room_h if room_h and room_h > 0 else med)
    big_pl = max([len(p["idx"]) for p in planes], default=0)
    segs, rejected = [], 0
    for pi, p in enumerate(planes):
        t = np.cross(up, p["n"])
        uv = np.stack([Pnt[p["idx"]] @ t, hgt[p["idx"]]], 1)
        for comp in components_2d(uv, cell, min_pts // 2, dilate=2):
            sub = p["idx"][comp]
            u = Pnt[sub] @ t
            h = hgt[sub]
            u0, u1 = np.percentile(u, [2, 98])
            h0, h1 = np.percentile(h, [2, 98])
            g = np.floor(np.stack([u - u0, h - h0], 1) / cell).astype(int)
            cover = len(np.unique(g[:, 0] * 100000 + g[:, 1])) / max(1.0, ((u1 - u0) / cell + 1) * ((h1 - h0) / cell + 1))
            if h1 - h0 < min_h or u1 - u0 < a.wall_min_len * med or cover < 0.2 or len(sub) < 0.02 * big_pl:
                rejected += 1
                continue
            hm = 0.5 * (h0 + h1)
            fl = [f for f, _ in floors if f < hm]
            ce = [c for c, _ in ceils if c > hm]
            if fl and h0 - max(fl) < 0.35 * (h1 - h0):
                h0 = max(fl)
            if ce and min(ce) - h1 < 0.35 * (h1 - h0):
                h1 = min(ce)
            segs.append({"plane": pi, "n": p["n"], "d": p["d"], "t": t, "u0": float(u0), "u1": float(u1),
                         "h0": float(h0), "h1": float(h1), "idx": sub, "cover": round(float(cover), 3),
                         "color": (Col[sub].mean(0) / 255.0).tolist()})
    stats["segments_rejected"] = rejected

    # esquinas: intersección de dos paredes (y piso/techo) cerca de los extremos de ambos tramos
    corners = []
    cdist = a.corner_rel * med
    for x in range(len(segs)):
        for y in range(x + 1, len(segs)):
            s, r = segs[x], segs[y]
            if abs(float(s["n"] @ r["n"])) > math.cos(math.radians(30)):
                continue
            # punto de la recta vertical común: n_s·p = d_s, n_r·p = d_r, up·p = 0
            A = np.stack([s["n"], r["n"], up])
            pt = np.linalg.solve(A, np.array([s["d"], r["d"], 0.0]))
            us, ur = pt @ s["t"], pt @ r["t"]
            ds_ = min(abs(us - s["u0"]), abs(us - s["u1"]))
            dr_ = min(abs(ur - r["u0"]), abs(ur - r["u1"]))
            ins = s["u0"] - cdist < us < s["u1"] + cdist and r["u0"] - cdist < ur < r["u1"] + cdist
            if not ins:                                      # esquina (L), unión en T o cruce
                continue
            for seg, uu, dd_ in ((s, us, ds_), (r, ur, dr_)):  # extremos cercanos: llevarlos a la esquina
                if dd_ > cdist:
                    continue
                if abs(uu - seg["u0"]) < abs(uu - seg["u1"]):
                    seg["u0"] = float(uu)
                else:
                    seg["u1"] = float(uu)
            h0, h1 = max(s["h0"], r["h0"]), min(s["h1"], r["h1"])
            corners.append({"segs": [x, y], "bottom": (pt + h0 * up).tolist(), "top": (pt + h1 * up).tolist()})
    stats["wall_segments"] = len(segs)
    stats["corners"] = len(corners)

    # malla simple: un rectángulo (2 triángulos) por tramo + polígono por nivel de piso
    V, Fc, Vc = [], [], []

    def add_quad(q, color):
        b = len(V)
        V.extend(q)
        Vc.extend([color] * 4)
        Fc.extend([(b, b + 1, b + 2), (b, b + 2, b + 3)])

    for s in segs:
        base = s["n"] * s["d"]
        q = [base + s["u0"] * s["t"] + s["h0"] * up, base + s["u1"] * s["t"] + s["h0"] * up,
             base + s["u1"] * s["t"] + s["h1"] * up, base + s["u0"] * s["t"] + s["h1"] * up]
        shade = 0.75 + 0.25 * abs(float(s["n"] @ e1))
        add_quad(q, (np.array(s["color"]) * shade).clip(0, 1).tolist())
    floor_polys = []
    import cv2
    for fh, _ in floors:
        sel = np.flatnonzero((Grp == G_FLOOR) & (np.abs(hgt - fh) < tol * 1.5))
        if len(sel) < min_pts:
            continue
        uv = np.stack([Pnt[sel] @ e1, Pnt[sel] @ e2], 1)
        o = uv.min(0)
        g = np.floor((uv - o) / cell).astype(int)
        occ = np.zeros(g.max(0) + 3, np.uint8)
        occ[g[:, 0] + 1, g[:, 1] + 1] = 255
        occ = cv2.morphologyEx(occ, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        cnts, _ = cv2.findContours(occ, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        fcol = (Col[sel].mean(0) / 255.0).tolist()
        for c in cnts:
            if cv2.contourArea(c) < 20:
                continue
            ap_ = cv2.approxPolyDP(c, 1.5, True)[:, 0, :].astype(np.float64)
            if len(ap_) < 3:
                continue
            uvp = (ap_ - 1 + 0.5) * cell + o
            P3 = [fh * up + p_[0] * e1 + p_[1] * e2 for p_ in uvp]
            b = len(V)
            V.extend(P3)
            Vc.extend([fcol] * len(P3))
            for tri in ear_clip(uvp):
                Fc.append((b + tri[0], b + tri[2], b + tri[1]))
            floor_polys.append({"height": fh, "polygon": np.array(P3).tolist()})
    stats["floor_polygons"] = len(floor_polys)
    stats["mesh_vertices"] = len(V)
    stats["mesh_triangles"] = len(Fc)
    stats["structure_seconds"] = round(time.time() - ts, 1)
    print(f"estructura: {len(segs)} tramos de pared, {len(corners)} esquinas, {len(floor_polys)} polígonos de "
          f"piso -> {len(V)} vértices, {len(Fc)} triángulos ({stats['structure_seconds']} s)", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    if Fc:
        import trimesh
        Vc8 = (np.array(Vc) * 255).astype(np.uint8)
        tm = trimesh.Trimesh(np.array(V), np.array(Fc), vertex_colors=np.c_[Vc8, np.full(len(Vc8), 255)],
                             process=False)
        tm2 = tm.copy()
        tm2.invert()                                        # caras dobles: visibles desde los dos lados
        trimesh.util.concatenate([tm, tm2]).export(a.out + "_estructura.glb")

    # 4b. realineación suave por planos --------------------------------------------------------
    c2w_fix = c2w_np.copy()
    if a.align and (segs or floors):
        ts = time.time()
        c2w_fix, ast = align_to_planes(D, K, c2w_np, keep, group, segs, floors, up, med, dev, sigma=a.align_sigma)
        ast["seconds"] = round(time.time() - ts, 1)
        stats["align"] = ast
        print(f"realineación por planos: {ast}", flush=True)
    c2wf = torch.from_numpy(c2w_fix).to(dev)

    # 5. ajuste a planos ----------------------------------------------------------------------
    ts = time.time()
    snap = np.zeros((N, H, W), np.float16)
    snap_d = a.snap_rel * med
    if segs:
        Sn = torch.tensor(np.array([s["n"] for s in segs]), device=dev, dtype=torch.float32)
        Sd = torch.tensor([s["d"] for s in segs], device=dev, dtype=torch.float32)
        St = torch.tensor(np.array([s["t"] for s in segs]), device=dev, dtype=torch.float32)
        Su = torch.tensor([[s["u0"], s["u1"], s["h0"], s["h1"]] for s in segs], device=dev, dtype=torch.float32)
    upt = torch.tensor(up, device=dev, dtype=torch.float32)
    fl_h = torch.tensor([f for f, _ in floors], device=dev, dtype=torch.float32)
    n_snap = 0
    for i in range(N):
        kk = torch.from_numpy(keep[i]).to(dev)
        g = torch.from_numpy(group[i]).to(dev)
        z_new = torch.zeros(H, W, device=dev)
        dist_best = torch.full((H, W), float("inf"), device=dev)
        P = backproject(D[i], K[i], c2wf[i], ys, xs)          # [H,W,3], con la pose realineada
        ray = torch.stack([(xs - K[i, 0, 2]) / K[i, 0, 0], (ys - K[i, 1, 2]) / K[i, 1, 1], torch.ones_like(xs)], -1)
        rw = ray @ c2wf[i, :3, :3].T
        cc = c2wf[i, :3, 3]
        if segs:
            wm = kk & (g == G_WALL)
            for s in range(len(segs)):
                dist = (P @ Sn[s] - Sd[s]).abs()
                u = P @ St[s]
                h = P @ upt
                m = wm & (dist < snap_d) & (dist < dist_best) & (u > Su[s, 0] - snap_d) & (u < Su[s, 1] + snap_d) \
                    & (h > Su[s, 2] - snap_d) & (h < Su[s, 3] + snap_d)
                den = rw @ Sn[s]
                z = (Sd[s] - cc @ Sn[s]) / torch.where(den.abs() < 1e-6, torch.full_like(den, 1e-6), den)
                m = m & (z > 0) & ((z - D[i]).abs() < 0.3 * D[i])
                z_new = torch.where(m, z, z_new)
                dist_best = torch.where(m, dist, dist_best)
        if len(fl_h):
            fm = kk & (g == G_FLOOR)
            h = P @ upt
            for f in range(len(fl_h)):
                dist = (h - fl_h[f]).abs()
                m = fm & (dist < snap_d) & (dist < dist_best)
                den = rw @ upt
                z = (fl_h[f] - cc @ upt) / torch.where(den.abs() < 1e-6, torch.full_like(den, 1e-6), den)
                m = m & (z > 0) & ((z - D[i]).abs() < 0.3 * D[i])
                z_new = torch.where(m, z, z_new)
                dist_best = torch.where(m, dist, dist_best)
        snap[i] = z_new.cpu().numpy().astype(np.float16)
        n_snap += int((z_new > 0).sum())
    stats["snapped_fraction"] = round(n_snap / tot, 4)
    stats["snap_seconds"] = round(time.time() - ts, 1)
    print(f"ajuste a planos: {stats['snapped_fraction']:.1%} de los píxeles ({stats['snap_seconds']} s)", flush=True)

    # 6. selección de frames -------------------------------------------------------------------
    gray = imgs.mean(-1).astype(np.float32)
    sharp = np.array([cv2.Laplacian(g_, cv2.CV_32F).var() for g_ in gray])
    On = O.cpu().numpy()
    use = np.zeros(N, bool)
    last = 0
    use[0] = True
    i = 1
    while i < N:
        ov = min(On[i, last], On[last, i])
        if ov < a.select_overlap or i - last >= a.max_skip:
            cand_ = [c for c in range(max(last + 1, i - 2), i + 1)]
            pick = max(cand_, key=lambda c: sharp[c])
            use[pick] = True
            last = pick
            i = pick + 1
        else:
            i += 1
    use[N - 1] = True
    stats["frames_selected"] = int(use.sum())
    print(f"selección: {use.sum()} de {N} frames", flush=True)

    use_all = np.zeros(len(d["is_real"]), bool)
    use_all[real] = use
    np.savez_compressed(a.out + "_filtro.npz", real=real, keep=keep, photo_ok=~rm_photo, depth_snap=snap,
                        labels=labels, use_frame=use_all, up=up,
                        c2w_fix=c2w_fix, aligned=np.array(bool(a.align)))
    struct = {"up": up.tolist(), "median_depth": med,
              "walls": [{"n": s["n"].tolist(), "d": s["d"], "u0": s["u0"], "u1": s["u1"], "h0": s["h0"],
                         "h1": s["h1"], "points": int(len(s["idx"]))} for s in segs],
              "floors": [{"height": h, "points": c} for h, c in floors],
              "ceilings": [{"height": h, "points": c} for h, c in ceils],
              "corners": corners, "floor_polygons": floor_polys}
    with open(a.out + "_estructura.json", "w") as fh:
        json.dump(struct, fh)
    stats["seconds"] = round(time.time() - t0, 1)
    stats["args"] = vars(a)
    with open(a.out + "_filtro_info.json", "w") as fh:
        json.dump(stats, fh, indent=1)

    # figura de control
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        sel = [N // 6, N // 2, 5 * N // 6]
        fig, ax = plt.subplots(3, 4, figsize=(16, 12))
        pal = np.array([[.5, .5, .5], [.9, .6, .2], [.2, .6, .9], [.6, .9, .3], [1, 0, 1]])
        for r, i in enumerate(sel):
            ax[r, 0].imshow(imgs[i]); ax[r, 0].set_title(f"frame {i}")
            ax[r, 1].imshow(pal[group[i]]); ax[r, 1].set_title("pared/piso/techo/quitado")
            ov_ = imgs[i].astype(np.float32) / 255
            ov_[~keep[i]] = ov_[~keep[i]] * 0.3 + np.array([0.7, 0, 0])
            ax[r, 2].imshow(ov_.clip(0, 1)); ax[r, 2].set_title(f"descartado (rojo) {1 - keep[i].mean():.0%}")
            ax[r, 3].imshow(np.where(snap[i] > 0, 1.0, 0.0), cmap="gray"); ax[r, 3].set_title("ajustado a plano")
            for c_ in range(4):
                ax[r, c_].axis("off")
        fig.tight_layout(); fig.savefig(a.out + "_filtro.png", dpi=70); plt.close(fig)
        fig, ax = plt.subplots(figsize=(10, 10))
        sub = rng.choice(len(Pnt), min(80000, len(Pnt)), replace=False)
        ax.scatter(Pnt[sub] @ e1, Pnt[sub] @ e2, s=0.3, c=Col[sub] / 255.0)
        for s in segs:
            base = s["n"] * s["d"]
            a0, a1 = base + s["u0"] * s["t"], base + s["u1"] * s["t"]
            ax.plot([a0 @ e1, a1 @ e1], [a0 @ e2, a1 @ e2], "r-", lw=2)
        for c_ in corners:
            b_ = np.array(c_["bottom"])
            ax.plot(b_ @ e1, b_ @ e2, "ko", ms=5)
        ax.set_aspect("equal"); ax.set_title(f"{len(segs)} tramos de pared, {len(corners)} esquinas")
        fig.tight_layout(); fig.savefig(a.out + "_estructura.png", dpi=80); plt.close(fig)
    except Exception as e:                                       # la figura es opcional
        print("figura:", e)
    print(f"listo en {stats['seconds']} s -> {a.out}_filtro.npz", flush=True)


if __name__ == "__main__":
    main()
