"""Corrección histórica con los keyframes de Stella (integración Stella-VSLAM, etapa 12).

Stella corrige su trayectoria con bundle adjustment local y, si detecta un lazo, con loop closure:
mueve las poses de sus KEYFRAMES. Las poses de cámara que publicó en su momento (/stella/camera_pose)
no se vuelven a publicar corregidas. Para que la corrección llegue a la geometría ya registrada:

  1. al registrar cada frame f se lo ancla al keyframe k de Stella más cercano en el tiempo que ya
     exista (por el timestamp de imagen del keyframe, topic ~/keyframes_full del nodo parcheado), y se
     guarda la pose relativa  T_rel = T_k(en ese momento)^-1 · T_f ;
  2. cuando llega una versión nueva de los keyframes, la pose corregida del frame es
     T_f' = T_k(ahora) · T_rel ;
  3. la pose de referencia (modo STELLA: Sim(3) del mapa de Stella al mundo BASIC) y la geometría
     del frame (MapAccumulator.set_pose) se recalculan con T_f'.

Todo en la convención CV del repo (los keyframes llegan en ejes ROS de stella_map y se convierten).
No se toca la pose de los frames anclados a keyframes que Stella borró (culling): quedan como estaban
y se cuentan.
"""
import bisect

import numpy as np

import frames as F


def kf_rows_to_cv(rows):
    """Filas (id, ts, px,py,pz, qx,qy,qz,qw) en ejes ROS -> {id: (ts, c2w_cv 4x4)}."""
    from pose_buffer import quat_to_mat
    out = {}
    for r in rows:
        T = np.eye(4)
        T[:3, :3] = quat_to_mat(np.asarray(r[5:9], np.float64))
        T[:3, 3] = r[2:5]
        out[int(r[0])] = (float(r[1]), F.map_link_to_c2w_cv(T))
    return out


class KeyframeHistory:
    def __init__(self, shift_thr=1e-3):
        self.current = {}            # id -> (ts, c2w_cv)
        self.events = []             # (stamp del mensaje, n keyframes, desplazamiento máximo, ids movidos)
        self.shift_thr = shift_thr
        self.version = 0

    def update(self, stamp, rows):
        """Ingresa un mensaje ~/keyframes_full. Devuelve (máximo desplazamiento, ids movidos)."""
        new = kf_rows_to_cv(rows)
        moved, mx = [], 0.0
        for k, (ts, T) in new.items():
            if k in self.current:
                d = float(np.linalg.norm(T[:3, 3] - self.current[k][1][:3, 3]))
                if d > self.shift_thr:
                    moved.append(k)
                    mx = max(mx, d)
        self.current = new
        self.version += 1
        if moved:
            self.events.append((float(stamp), len(new), mx, moved))
        return mx, moved

    def nearest(self, t, max_dt=2.0):
        """Keyframe vigente con timestamp más cercano a t (<= max_dt). (id, c2w) o None."""
        if not self.current:
            return None
        items = sorted((ts, k) for k, (ts, _) in self.current.items())
        tss = [x[0] for x in items]
        i = bisect.bisect_left(tss, t)
        cands = [j for j in (i - 1, i) if 0 <= j < len(items)]
        j = min(cands, key=lambda j: abs(tss[j] - t))
        if abs(tss[j] - t) > max_dt:
            return None
        k = items[j][1]
        return k, self.current[k][1]


class FrameAnchors:
    """Ancla de cada frame registrado a un keyframe de Stella, y su corrección."""

    def __init__(self):
        self.anchor = {}             # fid -> (kf_id, T_rel)
        self.original = {}           # fid -> pose de Stella del frame al registrarlo (CV)

    def register(self, fid, t, T_stella_cv, history: KeyframeHistory, max_dt=2.0):
        self.original[fid] = np.asarray(T_stella_cv, np.float64)
        n = history.nearest(t, max_dt)
        if n is None:
            return False
        k, Tk = n
        self.anchor[fid] = (k, np.linalg.inv(Tk) @ self.original[fid])
        return True

    def corrected(self, fid, history: KeyframeHistory):
        """Pose de Stella corregida del frame (CV) o None si su keyframe ya no existe / no tiene ancla."""
        a = self.anchor.get(fid)
        if a is None or a[0] not in history.current:
            return None
        return history.current[a[0]][1] @ a[1]

    def corrections(self, history: KeyframeHistory):
        """{fid: (pose original, pose corregida, desplazamiento)} de todos los frames anclados vigentes."""
        out = {}
        for fid in self.anchor:
            c = self.corrected(fid, history)
            if c is not None:
                o = self.original[fid]
                out[fid] = (o, c, float(np.linalg.norm(c[:3, 3] - o[:3, 3])))
        return out


def correct_offline(frame_stamps, frame_poses_cv, kf_messages, max_dt=2.0):
    """Para una grabación: ancla cada frame al keyframe vigente EN SU MOMENTO (el último mensaje de
    keyframes con stamp <= stamp del frame) y lo corrige con la última versión de los keyframes.

    kf_messages: lista ordenada de (stamp, filas) de ~/keyframes_full.
    Devuelve (poses corregidas (N,4,4) con NaN donde no se pudo, desplazamientos (N,), resumen)."""
    hist_at = KeyframeHistory()
    final = KeyframeHistory()
    anchors = FrameAnchors()
    msgs = sorted(kf_messages, key=lambda m: m[0])
    j = 0
    for i, (t, T) in enumerate(zip(frame_stamps, frame_poses_cv)):
        while j < len(msgs) and msgs[j][0] <= t:
            hist_at.update(msgs[j][0], msgs[j][1])
            j += 1
        if np.isfinite(T[0, 0]):
            anchors.register(i, t, T, hist_at, max_dt)
    if msgs:
        final.update(msgs[-1][0], msgs[-1][1])
    out = np.full((len(frame_stamps), 4, 4), np.nan)
    disp = np.full(len(frame_stamps), np.nan)
    n_lost_kf = 0
    for i in anchors.anchor:
        c = anchors.corrected(i, final)
        if c is None:
            n_lost_kf += 1
            continue
        out[i] = c
        disp[i] = float(np.linalg.norm(c[:3, 3] - anchors.original[i][:3, 3]))
    return out, disp, {"frames": len(frame_stamps), "anclados": len(anchors.anchor),
                       "corregidos": int(np.isfinite(disp).sum()), "keyframe_borrado": n_lost_kf,
                       "desplazamiento_max": float(np.nanmax(disp)) if np.isfinite(disp).any() else 0.0,
                       "desplazamiento_mediano": float(np.nanmedian(disp)) if np.isfinite(disp).any() else 0.0}
