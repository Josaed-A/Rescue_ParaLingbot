"""Mapa acumulado con geometría re-registrable (integración Stella-VSLAM, etapas 12 y 13).

Regla del plan: la geometría histórica no puede quedar pegada para siempre a las poses viejas. Por
eso cada frame guarda sus puntos EN COORDENADAS DE SU CÁMARA (salen de la profundidad de LingBot,
que no cambia) y, aparte, la pose con que se registra. El mapa en el mundo se arma en el momento:

    puntos_mundo(frame) = pose(frame) · puntos_cámara(frame)

Una corrección de pose (loop closure, BA, cambio de la pose de referencia) es cambiar una matriz por
frame: no hay que volver a correr el modelo ni recordar dónde quedó cada punto. La fusión espacial
(un punto por vóxel, promedio de posición y color) se hace sobre la versión vigente, así que tampoco
quedan "superficies dobles" de la pose vieja y la nueva.

Memoria: O(puntos guardados por frame), acotada con `max_points_per_frame`. El vóxel se fija en
unidades del modelo o relativo a la profundidad mediana.
"""
import threading

import numpy as np


def backproject(depth, K, conf=None, conf_thr=None, mask=None, max_points=None, rng=None, stride=1):
    """Profundidad (H,W) -> puntos en la cámara (N,3), con filtro de confianza / máscara opcional.
    Devuelve (puntos, índices (v,u)) para poder tomar el color de la imagen."""
    H, W = depth.shape
    v, u = np.mgrid[0:H:stride, 0:W:stride]
    z = depth[v, u].astype(np.float64)
    m = np.isfinite(z) & (z > 0)
    if conf is not None and conf_thr is not None:
        m &= conf[v, u] >= conf_thr
    if mask is not None:
        m &= mask[v, u]
    v, u, z = v[m], u[m], z[m]
    if max_points and len(z) > max_points:
        sel = (rng or np.random.default_rng(0)).choice(len(z), max_points, replace=False)
        v, u, z = v[sel], u[sel], z[sel]
    x = (u - K[0, 2]) / K[0, 0] * z
    y = (v - K[1, 2]) / K[1, 1] * z
    return np.stack([x, y, z], 1).astype(np.float32), (v, u)


class MapAccumulator:
    def __init__(self, voxel=None, max_points_per_frame=None):
        self.voxel = voxel
        self.max_points_per_frame = max_points_per_frame
        self.frames = {}          # id -> dict(pts_cam float32 (N,3), rgb uint8 (N,3), pose 4x4, stamp)
        self.order = []
        self.n_reposes = 0
        self._lock = threading.Lock()

    def add_frame(self, fid, pts_cam, rgb, pose_c2w, stamp=None):
        pts_cam = np.asarray(pts_cam, np.float32)
        rgb = np.asarray(rgb, np.uint8)
        if self.max_points_per_frame and len(pts_cam) > self.max_points_per_frame:
            sel = np.random.default_rng(int(fid) if np.isscalar(fid) else 0).choice(
                len(pts_cam), self.max_points_per_frame, replace=False)
            pts_cam, rgb = pts_cam[sel], rgb[sel]
        with self._lock:
            if fid not in self.frames:
                self.order.append(fid)
            self.frames[fid] = {"pts": pts_cam, "rgb": rgb, "pose": np.asarray(pose_c2w, np.float64).copy(),
                                "stamp": stamp}

    def remove_frame(self, fid):
        with self._lock:
            self.frames.pop(fid, None)
            if fid in self.order:
                self.order.remove(fid)

    def set_pose(self, fid, pose_c2w):
        """Re-registra la geometría de un frame con una pose nueva (corrección histórica)."""
        with self._lock:
            if fid in self.frames:
                self.frames[fid]["pose"] = np.asarray(pose_c2w, np.float64).copy()
                self.n_reposes += 1
                return True
        return False

    def pose(self, fid):
        with self._lock:
            return self.frames[fid]["pose"].copy() if fid in self.frames else None

    def world_points(self, fids=None):
        with self._lock:
            ids = list(self.order if fids is None else fids)
            P, C, F = [], [], []
            for f in ids:
                fr = self.frames.get(f)
                if fr is None or not len(fr["pts"]):
                    continue
                T = fr["pose"]
                P.append(fr["pts"].astype(np.float64) @ T[:3, :3].T + T[:3, 3])
                C.append(fr["rgb"])
                F.append(np.full(len(fr["pts"]), self.order.index(f), np.int32))
        if not P:
            return np.zeros((0, 3)), np.zeros((0, 3), np.uint8), np.zeros(0, np.int32)
        return np.concatenate(P), np.concatenate(C), np.concatenate(F)

    def fused(self, voxel=None):
        """Un punto por vóxel (promedio de posición y color) + cuántos frames distintos aportan a
        cada vóxel. Devuelve (puntos, colores, n_frames_por_voxel, n_puntos_por_voxel)."""
        voxel = voxel or self.voxel
        P, C, F = self.world_points()
        if not len(P):
            return P, C, np.zeros(0, int), np.zeros(0, int)
        lo = P.min(0)
        ijk = np.floor((P - lo) / voxel).astype(np.int64)
        key = (ijk[:, 0] << 42) ^ (ijk[:, 1] << 21) ^ ijk[:, 2]
        uk, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
        out_p = np.zeros((len(uk), 3))
        out_c = np.zeros((len(uk), 3))
        for k in range(3):
            out_p[:, k] = np.bincount(inv, weights=P[:, k], minlength=len(uk)) / cnt
            out_c[:, k] = np.bincount(inv, weights=C[:, k].astype(np.float64), minlength=len(uk)) / cnt
        # frames distintos por vóxel
        pair = np.unique(inv.astype(np.int64) * (F.max() + 1) + F)
        nfr = np.bincount(pair // (F.max() + 1), minlength=len(uk))
        return out_p.astype(np.float32), np.round(out_c).astype(np.uint8), nfr, cnt

    def stats(self, voxel=None):
        P, C, nfr, cnt = self.fused(voxel)
        raw = int(sum(len(f["pts"]) for f in self.frames.values()))
        mem = int(sum(f["pts"].nbytes + f["rgb"].nbytes + 128 for f in self.frames.values()))
        return {"frames": len(self.frames), "puntos_crudos": raw, "voxeles": int(len(P)),
                "duplicacion_crudos_por_voxel": round(raw / max(len(P), 1), 3),
                "voxeles_vistos_por_2_o_mas_frames": round(float((nfr >= 2).mean()) if len(nfr) else 0.0, 4),
                "memoria_mb": round(mem / 2 ** 20, 2), "reposes": self.n_reposes,
                "voxel": float(voxel or self.voxel or 0)}


def write_ply(path, P, C):
    P = np.asarray(P, np.float32)
    C = np.asarray(C, np.uint8)
    dt = np.dtype([("xyz", "<f4", 3), ("rgb", "u1", 3)])
    buf = np.empty(len(P), dtype=dt)
    buf["xyz"], buf["rgb"] = P, C
    with open(path, "wb") as f:
        f.write((f"ply\nformat binary_little_endian 1.0\nelement vertex {len(P)}\n"
                 "property float x\nproperty float y\nproperty float z\n"
                 "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n").encode("ascii"))
        f.write(buf.tobytes())
